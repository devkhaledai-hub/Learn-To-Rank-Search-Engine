"""
train.py — MS MARCO LTR Training Pipeline (runs on Modal cloud)

Usage:
    python train.py

Steps executed on Modal:
    1. download_data   — Fetch MS MARCO dataset from Kaggle into the Modal Volume
    2. prepare_data    — Load data, mine hard negatives, save pairs checkpoint
    3. build_features  — Extract base + semantic features, save features checkpoint
    4. train_model     — Train LightGBM Ranker, evaluate NDCG, save model
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import modal
from config.settings import app, image, volume, MOUNT, kaggle_secret


# ──────────────────────────────────────────────────────────────
# Step 0: Download MS MARCO from Kaggle
# ──────────────────────────────────────────────────────────────
@app.function(
    image=image,
    secrets=[kaggle_secret],
    volumes={MOUNT: volume},
    timeout=60 * 60
)
def download_data():
    import subprocess
    from utils.logging import get_logger
    logger = get_logger(__name__)

    logger.info("Downloading MS MARCO dataset from Kaggle...")
    os.chdir("/data")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", "engkhaledmo/ms-marco", "--unzip", "-p", "/data"],
        check=True,
    )
    logger.info("Download and extraction complete.")
    subprocess.run(["ls", "-la", "/data"])
    volume.commit()


# ──────────────────────────────────────────────────────────────
# Step 1: Load Data + Mine Hard Negatives
# ──────────────────────────────────────────────────────────────
@app.function(
    image=image,
    volumes={MOUNT: volume},
    timeout=60 * 120,
    memory=65536
)
def prepare_data(sample_size: int = 80_000):   #change from 500,000 to 10,000 for testing
    import sys
    sys.path.append("/pkg")
    import numpy as np
    from pathlib import Path
    from datapipeline.data_loader import DataLoader
    from datapipeline.negative_sampler import NegativeSampler
    from config.settings import CHECKPOINT_DIR, RANDOM_SEED
    from utils.logging import get_logger
    logger = get_logger(__name__)

    # Load & merge
    loader = DataLoader()
    df = loader.load_data()
    df = loader.preprocess(df)

    # Sample queries
    unique_queries = df["query_id"].unique()
    rng = np.random.RandomState(RANDOM_SEED)
    sampled = rng.choice(unique_queries, min(sample_size, len(unique_queries)), replace=False)
    df = df[df["query_id"].isin(sampled)].copy()

    queries_df = df[["query_id", "query"]].drop_duplicates()
    collection_df = df[["doc_id", "document"]].drop_duplicates()
    positive_pairs = df[["query_id", "doc_id", "relevance"]].copy()

    # Mine hard negatives
    sampler = NegativeSampler()
    all_pairs = sampler.sample(positive_pairs, queries_df, collection_df)

    # Merge text back
    from datapipeline.feature_pipeline import FeaturePipeline
    fpipe = FeaturePipeline()

    all_df = (
        all_pairs
        .merge(queries_df, on="query_id", how="left")
        .merge(collection_df, on="doc_id", how="left")
    )
    all_df = fpipe.preprocess(all_df)

    # Checkpoint 1
    cp = Path(CHECKPOINT_DIR) / "01_training_pairs.parquet"
    cp.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_parquet(cp, index=False)
    logger.info("Checkpoint 1 saved: %s  (%d rows)", cp, len(all_df))

    volume.commit()
    return str(cp)


# ──────────────────────────────────────────────────────────────
# Step 2: Feature Engineering  (GPU for semantic similarity)
# ──────────────────────────────────────────────────────────────
@app.function(
    image=image,
    volumes={MOUNT: volume},
    timeout=60 * 120,
    gpu="A100",
    memory=65536
)
def build_features():
    import sys
    sys.path.append("/pkg")
    import torch
    from pathlib import Path
    from datapipeline.feature_pipeline import FeaturePipeline
    from config.settings import CHECKPOINT_DIR
    from utils.logging import get_logger
    logger = get_logger(__name__)

    fpipe = FeaturePipeline()

    logger.info("Loading Checkpoint 1...")
    df = fpipe.load_data(str(Path(CHECKPOINT_DIR) / "01_training_pairs.parquet"))
    df = fpipe.preprocess(df)

    # Base features
    df = fpipe.extract_base_features(df)

    # Semantic features (on GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = fpipe.extract_semantic_features(df, device=device)

    # Sort by query_id (required by LightGBM)
    df = df.sort_values("query_id").reset_index(drop=True)

    # Checkpoint 2
    cp = Path(CHECKPOINT_DIR) / "02_features.parquet"
    df.to_parquet(cp, index=False)
    logger.info("Checkpoint 2 saved: %s  (%d rows)", cp, len(df))

    volume.commit()
    return str(cp)


# ──────────────────────────────────────────────────────────────
# Step 3: Train LightGBM Ranker + Evaluate
# ──────────────────────────────────────────────────────────────
@app.function(
    image=image,
    volumes={MOUNT: volume},
    timeout=60 * 30,
    memory=65536
)
def train_model():
    import sys
    sys.path.append("/pkg")
    import numpy as np
    from pathlib import Path
    from sklearn.metrics import ndcg_score

    from datapipeline.feature_pipeline import FeaturePipeline
    from models.lightgbm_ranker import LightGBMRanker
    from config.settings import (
        FEATURE_COLUMNS, CHECKPOINT_DIR, MODEL_DIR,
        TRAIN_RATIO, VALID_RATIO, RANDOM_SEED,
    )
    from utils.logging import get_logger
    logger = get_logger(__name__)

    fpipe = FeaturePipeline()

    logger.info("Loading Checkpoint 2...")
    df = fpipe.load_data(str(Path(CHECKPOINT_DIR) / "02_features.parquet"))

    # Split by query_id
    unique_queries = df["query_id"].unique()
    rng = np.random.RandomState(RANDOM_SEED)
    rng.shuffle(unique_queries)

    # Sub-sample to 15K queries for faster training (same as original notebook)
    if len(unique_queries) > 20_000:
        unique_queries = unique_queries[:15_000]
        logger.info("Sub-sampled to 15,000 queries for evaluation.")

    split1 = int(len(unique_queries) * TRAIN_RATIO)
    split2 = int(len(unique_queries) * (TRAIN_RATIO + VALID_RATIO))

    train_q = set(unique_queries[:split1])
    valid_q = set(unique_queries[split1:split2])
    test_q  = set(unique_queries[split2:])

    train_df = df[df["query_id"].isin(train_q)].copy()
    valid_df = df[df["query_id"].isin(valid_q)].copy()
    test_df  = df[df["query_id"].isin(test_q)].copy()

    g_train = train_df.groupby("query_id").size().tolist()
    g_valid = valid_df.groupby("query_id").size().tolist()

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["relevance"]
    X_valid, y_valid = valid_df[FEATURE_COLUMNS], valid_df["relevance"]
    X_test = test_df[FEATURE_COLUMNS]

    logger.info("Train: %d docs / %d queries", len(X_train), len(train_q))
    logger.info("Valid: %d docs / %d queries", len(X_valid), len(valid_q))
    logger.info("Test:  %d docs / %d queries", len(X_test), len(test_q))

    # Train
    ranker = LightGBMRanker()
    ranker.train(X_train, y_train, g_train, X_valid, y_valid, g_valid)

    # Evaluate on test set
    test_df = test_df.copy()
    test_df["predicted_score"] = ranker.predict(X_test)

    ndcg5, ndcg10, ndcg20 = [], [], []
    for _, group in test_df.groupby("query_id"):
        y_t = np.asarray([group["relevance"].values])
        y_p = np.asarray([group["predicted_score"].values])
        if y_t.shape[1] > 1:
            ndcg5.append(ndcg_score(y_t, y_p, k=5))
            ndcg10.append(ndcg_score(y_t, y_p, k=10))
            ndcg20.append(ndcg_score(y_t, y_p, k=20))

    metrics = {
        "ndcg@5":  np.mean(ndcg5) if ndcg5 else 0.0,
        "ndcg@10": np.mean(ndcg10) if ndcg10 else 0.0,
        "ndcg@20": np.mean(ndcg20) if ndcg20 else 0.0,
    }

    logger.info("======= TEST SET METRICS =======")
    for k, v in metrics.items():
        logger.info("  %s: %.4f", k, v)

    # Feature importances
    importances = ranker.feature_importances(FEATURE_COLUMNS)
    logger.info("Feature Importances:")
    for feat, gain in importances:
        logger.info("  - %s: %.1f", feat, gain)

    # Save model
    model_dir = Path(MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    ranker.save(str(model_dir / "lgbm_ranker.joblib"))

    volume.commit()
    return metrics


# ──────────────────────────────────────────────────────────────
# Step 4: Build Tantivy Search Index
# ──────────────────────────────────────────────────────────────
@app.function(
    image=image,
    volumes={MOUNT: volume},
    timeout=60 * 60,
    memory=65536
)
def build_search_index():
    import sys
    sys.path.append("/pkg")
    import tantivy
    import csv
    import shutil
    import time
    from pathlib import Path
    from config.settings import TANTIVY_INDEX_DIR, COLLECTION_PATH
    from utils.logging import get_logger
    logger = get_logger(__name__)

    index_path = Path(TANTIVY_INDEX_DIR)

    # Check if index already exists and has enough documents
    if index_path.exists():
        try:
            schema_builder = tantivy.SchemaBuilder()
            schema_builder.add_text_field("doc_id", stored=True)
            schema_builder.add_text_field("document", stored=True, tokenizer_name="en_stem")
            existing_schema = schema_builder.build()
            existing_index = tantivy.Index(existing_schema, str(index_path))
            existing_searcher = existing_index.searcher()
            doc_count = existing_searcher.num_docs
            if doc_count > 1_000_000:
                logger.info("Tantivy index already built with %d documents. Skipping.", doc_count)
                return
            else:
                logger.info("Index exists but only has %d docs. Rebuilding...", doc_count)
                del existing_searcher, existing_index
                shutil.rmtree(index_path)
        except Exception as e:
            logger.info("Corrupt index (%s). Rebuilding...", e)
            shutil.rmtree(index_path)

    logger.info("Building Tantivy search index...")
    start = time.time()

    schema_builder = tantivy.SchemaBuilder()
    schema_builder.add_text_field("doc_id", stored=True)
    schema_builder.add_text_field("document", stored=True, tokenizer_name="en_stem")
    schema = schema_builder.build()

    index_path.mkdir(parents=True, exist_ok=True)
    index = tantivy.Index(schema, str(index_path))
    writer = index.writer(heap_size=2_000_000_000)

    count = 0
    with open(COLLECTION_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            doc = tantivy.Document(doc_id=str(row[0]), document=str(row[1]))
            writer.add_document(doc)
            count += 1
            if count % 1_000_000 == 0:
                logger.info("  Indexed %d documents...", count)

    writer.commit()
    logger.info("Tantivy index built in %.1f seconds (%d documents).", time.time() - start, count)
    volume.commit()


# ──────────────────────────────────────────────────────────────
# Main: Run all steps sequentially on Modal
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from utils.logging import get_logger
    logger = get_logger("train")

    # # --- Step 0: Download data from Kaggle (run once) ---
    with app.run():
        logger.info("Step 0: Downloading data from Kaggle...")
        download_data.remote()

    # --- Step 1: Prepare data + mine hard negatives ---
    with app.run():
        logger.info("Step 1: Preparing data + mining hard negatives...")
        data_path = prepare_data.remote(sample_size=25_000) #change from 500,000 to 10,000 for testing

    # --- Step 2: Build features (GPU) ---
    with app.run():
        logger.info("Step 2: Building features (GPU)...")
        feat_path = build_features.remote()

    # --- Step 3: Train model + evaluate ---
    with app.run():
        logger.info("Step 3: Training model + evaluating...")
        scores = train_model.remote()
        logger.info("Final Evaluation Scores on Test Set:")
        for metric, value in scores.items():
            logger.info("  %s: %.4f", metric.upper(), value)

    # --- Step 4: Build Tantivy search index (run once) ---
    with app.run():
        logger.info("Step 4: Building Tantivy search index...")
        build_search_index.remote()

    logger.info("All training steps completed successfully.")

