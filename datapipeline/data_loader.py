import pandas as pd
from pathlib import Path

from datapipeline.base_pipeline import BasePipeline
from utils.logging import get_logger
from config.settings import (
    COLLECTION_PATH, QUERIES_PATH, QRELS_PATH, CHECKPOINT_DIR
)

logger = get_logger(__name__)


class DataLoader(BasePipeline):
    """
    Loads the MS MARCO dataset (collection, queries, qrels) from TSV files.
    Supports Parquet caching for fast subsequent reads.
    Merges the three sources into a single DataFrame.
    """

    def load_data(self, file_path=None):
        """
        Load and merge collection, queries, and qrels into a unified DataFrame.

        Returns
        -------
        df : pd.DataFrame
            Columns: query_id, query, doc_id, document, relevance
        """
        collection = self._load_collection()
        queries = self._load_queries()
        qrels = self._load_qrels()

        # Merge: qrels → queries → collection
        logger.info("Merging qrels with queries and collection...")
        df = (
            qrels
            .merge(queries, on="query_id", how="left")
            .merge(collection, on="doc_id", how="left")
        )
        df = df[["query_id", "query", "doc_id", "document", "relevance"]]

        logger.info(
            "Loaded %d query-document pairs  |  %d unique queries  |  %d unique docs",
            len(df), df["query_id"].nunique(), df["doc_id"].nunique()
        )
        return df

    def preprocess(self, X):
        """
        Clean text columns: fill NaN and cast to str.
        """
        X = X.copy()
        X["query"] = X["query"].fillna("").astype(str)
        X["document"] = X["document"].fillna("").astype(str)
        logger.info("Preprocessed text columns (fillna + astype str).")
        return X

    # ── private helpers ──────────────────────────────────────

    def _load_collection(self):
        cache = Path(CHECKPOINT_DIR) / "collection.parquet"
        if cache.exists():
            logger.info("Reading collection from Parquet cache.")
            return pd.read_parquet(cache)

        logger.info("Reading collection TSV: %s", COLLECTION_PATH)
        df = pd.read_csv(
            COLLECTION_PATH, sep="\t",
            names=["doc_id", "document"], engine="pyarrow"
        )
        df["document"] = df["document"].fillna("")
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache)
        return df

    def _load_queries(self):
        cache = Path(CHECKPOINT_DIR) / "queries.parquet"
        if cache.exists():
            logger.info("Reading queries from Parquet cache.")
            return pd.read_parquet(cache)

        logger.info("Reading queries TSV: %s", QUERIES_PATH)
        df = pd.read_csv(
            QUERIES_PATH, sep="\t",
            names=["query_id", "query"], engine="pyarrow"
        )
        df["query"] = df["query"].fillna("")
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache)
        return df

    def _load_qrels(self):
        cache = Path(CHECKPOINT_DIR) / "qrels.parquet"
        if cache.exists():
            logger.info("Reading qrels from Parquet cache.")
            return pd.read_parquet(cache)

        logger.info("Reading qrels TSV: %s", QRELS_PATH)
        df = pd.read_csv(
            QRELS_PATH, sep="\t",
            names=["query_id", "unused", "doc_id", "relevance"], engine="pyarrow"
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache)
        return df
