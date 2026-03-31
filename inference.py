"""
inference.py — MS MARCO LTR Search Engine (runs on Modal cloud)

Usage:
    python inference.py

Provides:
    - fetch_sample_data     — Preview feature-engineered data
    - ProductionSearchEngine — Two-stage search (Tantivy + LightGBM re-ranking)
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import modal
from config.settings import app, image, volume, MOUNT


# ──────────────────────────────────────────────────────────────
# Fetch Sample Data (Preview)
# ──────────────────────────────────────────────────────────────
@app.function(
    image=image,
    volumes={MOUNT: volume}
)
def fetch_sample_data():
    import sys
    sys.path.append("/pkg")
    import pandas as pd
    from config.settings import CHECKPOINT_DIR

    df = pd.read_parquet(f"{CHECKPOINT_DIR}/02_features.parquet")

    positives = df[df["relevance"] == 1].sample(3, random_state=42)
    negatives = df[df["relevance"] == 0].sample(3, random_state=42)
    sample = pd.concat([positives, negatives]).reset_index(drop=True)

    return sample.to_dict(orient="records")


# ──────────────────────────────────────────────────────────────
# Production Search Engine (Tantivy + LightGBM Re-ranking)
# ──────────────────────────────────────────────────────────────
@app.cls(
    image=image,
    volumes={MOUNT: volume},
    timeout=60 * 10,
    gpu="A100",
    memory=65536
)
class ProductionSearchEngine:

    @modal.enter()
    def initialize(self):
        import sys
        sys.path.append("/pkg")
        import joblib
        import torch
        from sentence_transformers import SentenceTransformer
        import tantivy
        import time
        from pathlib import Path
        from utils.logging import get_logger

        logger = get_logger(__name__)
        logger.info("Spinning up Search Engine...")
        start = time.time()

        # 1. Load Neural Model (GPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.semantic_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

        # 2. Load LightGBM Ranker
        model_path = Path("/data/models/ms_marco_v2/lgbm_ranker.joblib")
        if not model_path.exists():
            raise FileNotFoundError("Model not found. Run train.py first!")
        self.ranker_model = joblib.load(model_path)

        # 3. Connect to Tantivy
        logger.info("Connecting to Tantivy Database...")
        self.index = tantivy.Index.open("/data/tantivy_index")
        self.searcher = self.index.searcher()

        logger.info("Search Engine READY! Boot time: %.1fs", time.time() - start)

    @modal.method()
    def search(self, query: str, top_k: int = 10):
        import sys
        sys.path.append("/pkg")
        import numpy as np
        import pandas as pd
        import torch
        import time

        from config.settings import FEATURE_COLUMNS

        start = time.time()

        # ── Stage 1: Tantivy Retrieval ───────────────────────
        query_parser = self.index.parse_query(query, ["document"])
        result = self.searcher.search(query_parser, 100)

        # Fuzzy fallback
        if len(result.hits) < 20:
            fuzzy = " ".join(f"{w}~1" if len(w) > 3 else w for w in query.split())
            query_parser = self.index.parse_query(fuzzy, ["document"])
            result = self.searcher.search(query_parser, 100)

        doc_ids, documents = [], []
        for _, doc_address in result.hits:
            doc = self.searcher.doc(doc_address)
            doc_ids.append(doc["doc_id"][0])
            documents.append(doc["document"][0])

        if not doc_ids:
            return {"time": time.time() - start, "results": []}

        candidates = pd.DataFrame({"doc_id": doc_ids, "document": documents})

        # ── Stage 2: Feature Extraction ──────────────────────
        q_tokens = query.lower().split()
        q_set = set(q_tokens)

        candidates["query_len"] = len(q_tokens)
        candidates["doc_len"] = candidates["document"].apply(lambda x: len(str(x).split()))

        candidates["overlap_ratio"] = candidates["document"].apply(
            lambda d: len(q_set.intersection(set(str(d).lower().split()))) / len(q_set) if q_set else 0
        )
        candidates["exact_match_count"] = candidates["document"].apply(
            lambda d: sum(1 for q in q_tokens if q in set(str(d).lower().split()))
        )

        def calc_proxies(doc_text):
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

        proxy_res = candidates["document"].apply(calc_proxies)
        candidates["bm25_proxy"] = proxy_res.apply(lambda x: x[0])
        candidates["tf_idf_proxy"] = proxy_res.apply(lambda x: x[1])

        # Semantic similarity (GPU)
        q_emb = self.semantic_model.encode(query, convert_to_tensor=True, show_progress_bar=False).unsqueeze(0)
        d_emb = self.semantic_model.encode(candidates["document"].tolist(), convert_to_tensor=True, show_progress_bar=False)
        if d_emb.dim() == 1:
            d_emb = d_emb.unsqueeze(0)
        cos_sim = torch.nn.functional.cosine_similarity(q_emb, d_emb, dim=1).cpu().numpy()
        candidates["semantic_sim"] = cos_sim

        # ── Stage 3: LightGBM Re-Ranking ─────────────────────
        ranking_scores = self.ranker_model.predict(candidates[FEATURE_COLUMNS])
        candidates["ml_rank_score"] = ranking_scores

        final = candidates.sort_values("ml_rank_score", ascending=False).head(top_k)
        elapsed = time.time() - start

        result_cols = [
            "doc_id", "ml_rank_score", "document",
            "semantic_sim", "tf_idf_proxy",
            "exact_match_count", "overlap_ratio", "bm25_proxy",
        ]
        results = final[result_cols].to_dict(orient="records")
        return {"time": elapsed, "results": results}


# ──────────────────────────────────────────────────────────────
# Main: Run search queries on Modal
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from utils.logging import get_logger
    logger = get_logger("inference")

    # --- Preview feature data ---
    with app.run():
        import pandas as pd
        logger.info("Fetching sample data preview...")
        sample = fetch_sample_data.remote()
        logger.info("DATA PREVIEW:\n%s", pd.DataFrame(sample).to_string())

    # --- Search ---
    with app.run():
        engine = ProductionSearchEngine()
        query = "what causes high blood pressure?"
        response = engine.search.remote(query)

        logger.info("Search Query: '%s'", query)
        logger.info("Ranked in %.3f seconds", response["time"])
        for i, res in enumerate(response["results"]):
            logger.info(
                "  Rank %d [Score: %.2f] Doc %s: %s",
                i + 1, res["ml_rank_score"], res["doc_id"],
                res["document"][:150],
            )

