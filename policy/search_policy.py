import pandas as pd

from policy.base_policy import BasePolicy
from datapipeline.feature_pipeline import FeaturePipeline
from models.lightgbm_ranker import LightGBMRanker
from utils.logging import get_logger
from config.settings import (
    FEATURE_COLUMNS, RETRIEVAL_TOP_K, RERANK_TOP_K, FUZZY_THRESHOLD,
)

logger = get_logger(__name__)


class SearchPolicy(BasePolicy):
    """
    Two-stage search engine policy:
      Stage 1 — Tantivy BM25 retrieval  (fast keyword matching)
      Stage 2 — Feature extraction + LightGBM re-ranking  (ML precision)
    """

    def __init__(self, model_path: str, index_path: str, device: str = "cpu"):
        self.device = device
        self.feature_pipeline = FeaturePipeline()
        self.ranker = LightGBMRanker()
        self.load(model_path)
        self._init_tantivy(index_path)

    # ── BasePolicy interface ─────────────────────────────────

    def train(self, data):
        """Not used for search."""
        pass

    def predict(self, input_data):
        """
        Full two-stage search for a raw query string.

        Parameters
        ----------
        input_data : str   — the user's search query

        Returns
        -------
        dict with 'time' (seconds) and 'results' (list of dicts)
        """
        import time as _time

        query = input_data
        start = _time.time()

        # ── Stage 1: Tantivy retrieval ───────────────────────
        candidates = self._retrieve(query)
        if candidates.empty:
            return {"time": _time.time() - start, "results": []}

        # ── Stage 2: Feature extraction + re-ranking ─────────
        candidates = self.feature_pipeline.extract_inference_features(
            query, candidates, device=self.device
        )

        scores = self.ranker.predict(candidates[FEATURE_COLUMNS])
        candidates["ml_rank_score"] = scores

        final = (
            candidates
            .sort_values("ml_rank_score", ascending=False)
            .head(RERANK_TOP_K)
        )

        elapsed = _time.time() - start

        result_cols = [
            "doc_id", "ml_rank_score", "document",
            "semantic_sim", "tf_idf_proxy",
            "exact_match_count", "overlap_ratio", "bm25_proxy",
        ]
        results = final[result_cols].to_dict(orient="records")
        return {"time": elapsed, "results": results}

    def save(self, model, path_to_model):
        self.ranker.save(path_to_model)

    def load(self, path_to_model):
        logger.info("Loading ranker model from %s", path_to_model)
        self.ranker.load(path_to_model)

    # ── private helpers ──────────────────────────────────────

    def _init_tantivy(self, index_path: str):
        """Open an existing Tantivy index."""
        import tantivy

        logger.info("Connecting to Tantivy index at %s", index_path)
        self.index = tantivy.Index.open(index_path)
        self.searcher = self.index.searcher()
        logger.info("Tantivy searcher ready  (%d docs).", self.searcher.num_docs)

    def _retrieve(self, query: str) -> pd.DataFrame:
        """
        Stage 1: BM25 retrieval via Tantivy.
        Falls back to fuzzy search when exact hits are below threshold.
        """
        query_obj = self.index.parse_query(query, ["document"])
        result = self.searcher.search(query_obj, RETRIEVAL_TOP_K)

        # Fuzzy fallback
        if len(result.hits) < FUZZY_THRESHOLD:
            fuzzy = " ".join(
                f"{w}~1" if len(w) > 3 else w for w in query.split()
            )
            query_obj = self.index.parse_query(fuzzy, ["document"])
            result = self.searcher.search(query_obj, RETRIEVAL_TOP_K)

        doc_ids, documents = [], []
        for _, doc_address in result.hits:
            doc = self.searcher.doc(doc_address)
            doc_ids.append(doc["doc_id"][0])
            documents.append(doc["document"][0])

        if not doc_ids:
            logger.info("Tantivy returned 0 hits for query: '%s'", query)
            return pd.DataFrame()

        return pd.DataFrame({"doc_id": doc_ids, "document": documents})
