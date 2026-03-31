import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import ndcg_score

from policy.base_policy import BasePolicy
from datapipeline.data_loader import DataLoader
from datapipeline.negative_sampler import NegativeSampler
from datapipeline.feature_pipeline import FeaturePipeline
from models.lightgbm_ranker import LightGBMRanker
from utils.logging import get_logger
from config.settings import (
    FEATURE_COLUMNS, SAMPLE_SIZE, RANDOM_SEED,
    TRAIN_RATIO, VALID_RATIO,
    CHECKPOINT_DIR, MODEL_DIR,
)

logger = get_logger(__name__)


class TrainingPolicy(BasePolicy):
    """
    Orchestrates the full LTR training pipeline:
      1. Load raw data  (DataLoader)
      2. Sample queries
      3. Mine hard negatives  (NegativeSampler)
      4. Extract features  (FeaturePipeline)
      5. Split train / valid / test by query_id
      6. Train LightGBM Ranker
      7. Evaluate NDCG on held-out test set
      8. Save model + report feature importances
    """

    def __init__(self, sample_size: int = SAMPLE_SIZE, device: str = "cpu"):
        self.sample_size = sample_size
        self.device = device

        self.data_loader = DataLoader()
        self.sampler = NegativeSampler()
        self.feature_pipeline = FeaturePipeline()
        self.ranker = LightGBMRanker()

    # ── BasePolicy interface ─────────────────────────────────

    def train(self, data=None):
        """Run the complete training pipeline."""

        # 1. Load & merge raw data
        logger.info("=== Step 1: Loading Raw Data ===")
        df = self.data_loader.load_data()
        df = self.data_loader.preprocess(df)

        # 2. Sample queries
        logger.info("=== Step 2: Sampling Queries ===")
        unique_queries = df["query_id"].unique()
        rng = np.random.RandomState(RANDOM_SEED)
        sampled = rng.choice(
            unique_queries,
            min(self.sample_size, len(unique_queries)),
            replace=False,
        )
        df = df[df["query_id"].isin(sampled)].copy()

        queries_df = df[["query_id", "query"]].drop_duplicates()
        collection_df = df[["doc_id", "document"]].drop_duplicates()
        positive_pairs = df[["query_id", "doc_id", "relevance"]].copy()

        # 3. Mine hard negatives
        logger.info("=== Step 3: Mining Hard Negatives ===")
        all_pairs = self.sampler.sample(positive_pairs, queries_df, collection_df)

        # Merge text back
        all_df = (
            all_pairs
            .merge(queries_df, on="query_id", how="left")
            .merge(collection_df, on="doc_id", how="left")
        )
        all_df = self.feature_pipeline.preprocess(all_df)

        # Checkpoint 1
        cp1 = Path(CHECKPOINT_DIR) / "01_training_pairs.parquet"
        cp1.parent.mkdir(parents=True, exist_ok=True)
        all_df.to_parquet(cp1, index=False)
        logger.info("Checkpoint 1 saved: %s", cp1)

        # 4. Feature engineering
        logger.info("=== Step 4: Feature Engineering ===")
        all_df = self.feature_pipeline.extract_base_features(all_df)
        all_df = self.feature_pipeline.extract_semantic_features(all_df, device=self.device)

        # Sort by query_id (required by LightGBM)
        all_df = all_df.sort_values("query_id").reset_index(drop=True)

        # Checkpoint 2
        cp2 = Path(CHECKPOINT_DIR) / "02_features.parquet"
        all_df.to_parquet(cp2, index=False)
        logger.info("Checkpoint 2 saved: %s", cp2)

        # 5. Split by query_id
        logger.info("=== Step 5: Train/Valid/Test Split ===")
        uq = all_df["query_id"].unique()
        rng.shuffle(uq)

        split1 = int(len(uq) * TRAIN_RATIO)
        split2 = int(len(uq) * (TRAIN_RATIO + VALID_RATIO))

        train_q = set(uq[:split1])
        valid_q = set(uq[split1:split2])
        test_q = set(uq[split2:])

        train_df = all_df[all_df["query_id"].isin(train_q)].copy()
        valid_df = all_df[all_df["query_id"].isin(valid_q)].copy()
        test_df = all_df[all_df["query_id"].isin(test_q)].copy()

        g_train = train_df.groupby("query_id").size().tolist()
        g_valid = valid_df.groupby("query_id").size().tolist()

        X_train = train_df[FEATURE_COLUMNS]
        y_train = train_df["relevance"]
        X_valid = valid_df[FEATURE_COLUMNS]
        y_valid = valid_df["relevance"]
        X_test = test_df[FEATURE_COLUMNS]

        logger.info(
            "Train: %d docs / %d queries  |  Valid: %d / %d  |  Test: %d / %d",
            len(X_train), len(train_q), len(X_valid), len(valid_q),
            len(X_test), len(test_q),
        )

        # 6. Train
        logger.info("=== Step 6: Training LightGBM Ranker ===")
        self.ranker.train(
            X_train, y_train, g_train,
            X_valid, y_valid, g_valid,
        )

        # 7. Evaluate on test set
        logger.info("=== Step 7: Evaluating on Test Set ===")
        test_df = test_df.copy()
        test_df["predicted_score"] = self.ranker.predict(X_test)

        ndcg5, ndcg10, ndcg20 = [], [], []
        for _, group in test_df.groupby("query_id"):
            y_t = np.asarray([group["relevance"].values])
            y_p = np.asarray([group["predicted_score"].values])
            if y_t.shape[1] > 1:
                ndcg5.append(ndcg_score(y_t, y_p, k=5))
                ndcg10.append(ndcg_score(y_t, y_p, k=10))
                ndcg20.append(ndcg_score(y_t, y_p, k=20))

        metrics = {
            "ndcg@5": np.mean(ndcg5) if ndcg5 else 0.0,
            "ndcg@10": np.mean(ndcg10) if ndcg10 else 0.0,
            "ndcg@20": np.mean(ndcg20) if ndcg20 else 0.0,
        }

        logger.info("======= TEST SET METRICS =======")
        for k, v in metrics.items():
            logger.info("  %s: %.4f", k, v)
        logger.info("================================")

        # Feature importances
        importances = self.ranker.feature_importances(FEATURE_COLUMNS)
        logger.info("Feature Importances:")
        for feat, gain in importances:
            logger.info("  - %s: %.1f", feat, gain)

        # 8. Save
        self.save(model=None, path_to_model=None)

        return metrics

    def predict(self, input_data):
        """Not used during training."""
        pass

    def save(self, model=None, path_to_model=None):
        model_dir = Path(MODEL_DIR)
        model_dir.mkdir(parents=True, exist_ok=True)
        save_path = model_dir / "lgbm_ranker.joblib"
        self.ranker.save(str(save_path))
        logger.info("Ranker model saved to %s", save_path)

    def load(self, path_to_model):
        self.ranker.load(path_to_model)
        logger.info("Ranker model loaded from %s", path_to_model)

    def process(self):
        """Convenience method: train + save."""
        return self.train()
