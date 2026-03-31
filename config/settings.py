import os
import modal

# ──────────────────────────────────────────────────────────────
# Modal Cloud Configuration
# ──────────────────────────────────────────────────────────────
app = modal.App("ltr-search-engine-professional")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy",
        "pandas",
        "scikit-learn",
        "lightgbm",
        "xgboost",
        "tqdm",
        "kaggle",
        "pyarrow",
        "nltk",
        "rank-bm25",
        "torch",
        "sentence-transformers",
        "tantivy",
    )
    .apt_install("unzip")
    .env({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})
    .add_local_dir(BASE_DIR, remote_path="/pkg")
)

volume = modal.Volume.from_name("ltr-ms-data", create_if_missing=True)
MOUNT = "/data"

kaggle_secret = modal.Secret.from_dict(
    {
        "KAGGLE_USERNAME": "engkhaledmo",
        "KAGGLE_KEY": "b2462667c29019e451c6cd329b20ae28",
    }
)

# ──────────────────────────────────────────────────────────────
# Paths  (all relative to the Modal Volume at /data)
# ──────────────────────────────────────────────────────────────

COLLECTION_PATH = "/data/collection/collection.tsv"
QUERIES_PATH = "/data/queries/queries.train.tsv"
QRELS_PATH = "/data/qrels.train.tsv"

CHECKPOINT_DIR = "/data/checkpoints/ms_marco_v2"
MODEL_DIR = "/data/models/ms_marco_v2"
TANTIVY_INDEX_DIR = "/data/tantivy_index"

# ──────────────────────────────────────────────────────────────
# Data Preparation
# ──────────────────────────────────────────────────────────────
SAMPLE_SIZE = 500_000  # Number of unique queries to sample
NEGATIVES_PER_QUERY = 5  # Hard negatives mined per query
TFIDF_CORPUS_SIZE = 250_000  # Documents in the TF-IDF index
TFIDF_MAX_FEATURES = 100_000  # Vocabulary cap for TF-IDF vectorizer
RANDOM_SEED = 42

# ──────────────────────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────────────────────
FEATURE_COLUMNS = [
    "query_len",
    "doc_len",
    "overlap_ratio",
    "bm25_proxy",
    "exact_match_count",
    "tf_idf_proxy",
    "semantic_sim",
]

SEMANTIC_MODEL_NAME = "all-MiniLM-L6-v2"
SEMANTIC_BATCH_SIZE = 1024

# ──────────────────────────────────────────────────────────────
# LightGBM Ranker Hyperparameters
# ──────────────────────────────────────────────────────────────
RANKER_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": 42,
    "importance_type": "gain",
}

EARLY_STOPPING_ROUNDS = 50
EVAL_AT = [5, 10, 20]

# ──────────────────────────────────────────────────────────────
# Train / Validation / Test Split Ratios
# ──────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1  # remainder goes to test

# ──────────────────────────────────────────────────────────────
# Search Engine
# ──────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K = 100  # Stage-1 BM25 retrieval candidates
RERANK_TOP_K = 10  # Stage-2 final results returned
FUZZY_THRESHOLD = 20  # Fall back to fuzzy search below this hit count
