import numpy as np
import pandas as pd

from datapipeline.base_pipeline import BasePipeline
from utils.logging import get_logger
from config.settings import (
    FEATURE_COLUMNS, SEMANTIC_MODEL_NAME, SEMANTIC_BATCH_SIZE,
)

logger = get_logger(__name__)


class FeaturePipeline(BasePipeline):
    """
    Extracts ranking features for query-document pairs.

    Features
    --------
    * query_len           – number of tokens in the query
    * doc_len             – number of tokens in the document
    * overlap_ratio       – fraction of query words found in the document
    * exact_match_count   – count of query words that appear in the document
    * bm25_proxy          – lightweight BM25-style score  (tf / (tf + 1.2))
    * tf_idf_proxy        – tf normalised by log(doc_len + 2)
    * semantic_sim        – cosine similarity from SentenceTransformer embeddings
    """

    # ── interface methods ────────────────────────────────────

    def load_data(self, file_path):
        """Load a previously saved feature-enriched parquet checkpoint."""
        logger.info("Loading feature checkpoint from %s", file_path)
        return pd.read_parquet(file_path)

    def preprocess(self, X):
        """
        Clean text and fill NaN before feature extraction.
        """
        X = X.copy()
        X["query"] = X["query"].fillna("").astype(str)
        X["document"] = X["document"].fillna("").astype(str)
        return X

    # ── feature extraction ───────────────────────────────────

    def extract_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute length, overlap, BM25-proxy, TF-IDF-proxy, and exact-match
        features row-wise.
        """
        logger.info("Calculating base features (lengths, overlap, BM25, TF-IDF)...")

        results = df.apply(lambda row: pd.Series(self._calc_base(row)), axis=1)
        results.columns = [
            "query_len", "doc_len", "overlap_ratio",
            "bm25_proxy", "exact_match_count", "tf_idf_proxy",
        ]
        df = pd.concat([df, results], axis=1)
        logger.info("Base feature extraction complete.")
        return df

    def extract_semantic_features(self, df: pd.DataFrame, device: str = "cpu") -> pd.DataFrame:
        """
        Compute cosine similarity between query and document using
        SentenceTransformer embeddings.

        Parameters
        ----------
        df     : DataFrame with 'query' and 'document' text columns.
        device : 'cuda' or 'cpu'.
        """
        import torch
        from sentence_transformers import SentenceTransformer

        logger.info("Loading SentenceTransformer model '%s' on %s...",
                     SEMANTIC_MODEL_NAME, device)
        model = SentenceTransformer(SEMANTIC_MODEL_NAME, device=device)

        similarities = []
        total = len(df)
        for i in range(0, total, SEMANTIC_BATCH_SIZE):
            batch = df.iloc[i : i + SEMANTIC_BATCH_SIZE]
            q_emb = model.encode(
                batch["query"].tolist(),
                convert_to_tensor=True, show_progress_bar=False
            )
            d_emb = model.encode(
                batch["document"].tolist(),
                convert_to_tensor=True, show_progress_bar=False
            )
            cos_sim = torch.nn.functional.cosine_similarity(q_emb, d_emb)
            similarities.extend(cos_sim.cpu().numpy())

            if i % (SEMANTIC_BATCH_SIZE * 5) == 0:
                logger.info("  Semantic similarity: %d / %d rows", i, total)

        df = df.copy()
        df["semantic_sim"] = similarities
        logger.info("Semantic feature extraction complete.")
        return df

    def extract_inference_features(
        self, query: str, candidates: pd.DataFrame, device: str = "cpu"
    ) -> pd.DataFrame:
        """
        Extract features for a single query against a set of candidate
        documents (used at search / inference time).

        Parameters
        ----------
        query      : raw query string
        candidates : DataFrame with 'doc_id' and 'document' columns
        device     : 'cuda' or 'cpu'

        Returns
        -------
        candidates : DataFrame enriched with all FEATURE_COLUMNS
        """
        import torch
        from sentence_transformers import SentenceTransformer

        q_tokens = query.lower().split()
        q_set = set(q_tokens)
        candidates = candidates.copy()

        # Lengths
        candidates["query_len"] = len(q_tokens)
        candidates["doc_len"] = candidates["document"].apply(
            lambda x: len(str(x).split())
        )

        # Overlap
        candidates["overlap_ratio"] = candidates["document"].apply(
            lambda d: (
                len(q_set.intersection(set(str(d).lower().split()))) / len(q_set)
                if q_set else 0
            )
        )

        # Exact match
        candidates["exact_match_count"] = candidates["document"].apply(
            lambda d: sum(1 for q in q_tokens if q in set(str(d).lower().split()))
        )

        # BM25 + TF-IDF proxies
        def _proxies(doc_text):
            d_words = str(doc_text).lower().split()
            d_len = len(d_words)
            freqs = {}
            for w in d_words:
                freqs[w] = freqs.get(w, 0) + 1
            bm25, tfidf = 0.0, 0.0
            for q in q_tokens:
                tf = freqs.get(q, 0)
                bm25 += tf / (tf + 1.2)
                tfidf += tf / np.log(d_len + 2)
            return bm25, tfidf

        proxy_res = candidates["document"].apply(_proxies)
        candidates["bm25_proxy"] = proxy_res.apply(lambda x: x[0])
        candidates["tf_idf_proxy"] = proxy_res.apply(lambda x: x[1])

        # Semantic similarity
        model = SentenceTransformer(SEMANTIC_MODEL_NAME, device=device)
        q_emb = model.encode(
            query, convert_to_tensor=True, show_progress_bar=False
        ).unsqueeze(0)
        d_emb = model.encode(
            candidates["document"].tolist(),
            convert_to_tensor=True, show_progress_bar=False
        )
        if d_emb.dim() == 1:
            d_emb = d_emb.unsqueeze(0)
        cos_sim = torch.nn.functional.cosine_similarity(q_emb, d_emb, dim=1)
        candidates["semantic_sim"] = cos_sim.cpu().numpy()

        return candidates

    # ── private helpers ──────────────────────────────────────

    @staticmethod
    def _calc_base(row):
        q_words = str(row["query"]).lower().split()
        d_words = str(row["document"]).lower().split()

        q_len = len(q_words)
        d_len = len(d_words)

        q_set = set(q_words)
        d_set = set(d_words)

        overlap_ratio = len(q_set.intersection(d_set)) / max(q_len, 1)
        exact_match_count = sum(1 for q in q_words if q in d_set)

        freqs = {}
        for w in d_words:
            freqs[w] = freqs.get(w, 0) + 1

        bm25 = 0.0
        tfidf = 0.0
        for q in q_words:
            tf = freqs.get(q, 0)
            bm25 += tf / (tf + 1.2)
            tfidf += tf / np.log(d_len + 2) if d_len > 0 else 0.0

        return q_len, d_len, overlap_ratio, bm25, exact_match_count, tfidf
