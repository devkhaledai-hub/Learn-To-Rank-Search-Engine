import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.logging import get_logger
from config.settings import (
    NEGATIVES_PER_QUERY, TFIDF_CORPUS_SIZE,
    TFIDF_MAX_FEATURES, RANDOM_SEED,
)

logger = get_logger(__name__)


class NegativeSampler:
    """
    Mines hard negatives for each query using TF-IDF dot-product similarity.

    MS MARCO qrels only contain relevance = 1 (positive pairs).
    This class creates relevance = 0 pairs by selecting documents that
    are textually similar (hard negatives) but are NOT the true answer.
    """

    def __init__(
        self,
        negatives_per_query: int = NEGATIVES_PER_QUERY,
        corpus_size: int = TFIDF_CORPUS_SIZE,
        max_features: int = TFIDF_MAX_FEATURES,
        seed: int = RANDOM_SEED,
    ):
        self.negatives_per_query = negatives_per_query
        self.corpus_size = corpus_size
        self.max_features = max_features
        self.rng = np.random.RandomState(seed)

    def sample(
        self,
        positive_pairs: pd.DataFrame,
        queries: pd.DataFrame,
        collection: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build positive + hard-negative pairs.

        Parameters
        ----------
        positive_pairs : DataFrame  (query_id, doc_id, relevance)
        queries        : DataFrame  (query_id, query)
        collection     : DataFrame  (doc_id, document)

        Returns
        -------
        all_pairs : DataFrame  (query_id, doc_id, relevance)
        """
        logger.info("Building positive pairs (relevance = 1)...")
        pos_doc_ids = positive_pairs["doc_id"].unique()

        # Build a corpus subset: all positive docs + random fill
        remaining = max(0, self.corpus_size - len(pos_doc_ids))
        all_doc_ids = collection["doc_id"].values
        random_ids = self.rng.choice(all_doc_ids, size=remaining, replace=True)
        corpus_ids = np.concatenate([pos_doc_ids, random_ids])

        corpus_df = collection[collection["doc_id"].isin(corpus_ids)].copy()
        corpus_df = corpus_df.reset_index(drop=True)

        # Fit TF-IDF on corpus documents
        logger.info("[1/3] Fitting TF-IDF Vectorizer on %d documents...", len(corpus_df))
        vectorizer = TfidfVectorizer(max_features=self.max_features, lowercase=True)
        doc_vectors = vectorizer.fit_transform(corpus_df["document"])

        logger.info("[2/3] Transforming queries...")
        query_vectors = vectorizer.transform(queries["query"])

        logger.info("[3/3] Mining hard negatives via dot-product (chunked)...")
        neg_queries, neg_docs = [], []
        q_ids = queries["query_id"].values
        c_docs = corpus_df["doc_id"].values

        chunk_size = 10_000
        for chunk_start in range(0, query_vectors.shape[0], chunk_size):
            chunk_end = min(chunk_start + chunk_size, query_vectors.shape[0])
            sim_chunk = query_vectors[chunk_start:chunk_end].dot(doc_vectors.T)

            for i in range(sim_chunk.shape[0]):
                row = sim_chunk.getrow(i)
                global_i = chunk_start + i

                if row.nnz == 0:
                    top_indices = self.rng.choice(
                        len(c_docs), self.negatives_per_query, replace=False
                    )
                else:
                    top_idx_in_row = np.argsort(row.data)[-10:][::-1]
                    top_indices = row.indices[top_idx_in_row]
                    if len(top_indices) < self.negatives_per_query:
                        pad = self.rng.choice(
                            len(c_docs),
                            self.negatives_per_query - len(top_indices),
                            replace=False,
                        )
                        top_indices = np.concatenate([top_indices, pad])

                qid = q_ids[global_i]
                actual_pos = positive_pairs.loc[
                    positive_pairs["query_id"] == qid, "doc_id"
                ].values

                selected = []
                for idx in top_indices:
                    doc_id = c_docs[idx]
                    if doc_id not in actual_pos:
                        selected.append(doc_id)
                    if len(selected) == self.negatives_per_query:
                        break

                # Pad with random if not enough hard negatives
                while len(selected) < self.negatives_per_query:
                    selected.append(self.rng.choice(c_docs))

                neg_queries.extend([qid] * self.negatives_per_query)
                neg_docs.extend(selected)

        negative_pairs = pd.DataFrame({
            "query_id": neg_queries,
            "doc_id": neg_docs,
            "relevance": 0,
        })

        all_pairs = pd.concat(
            [positive_pairs[["query_id", "doc_id", "relevance"]], negative_pairs],
            ignore_index=True,
        )

        logger.info(
            "Created %d total pairs  |  Positives: %d  |  Negatives: %d",
            len(all_pairs), len(positive_pairs), len(negative_pairs),
        )
        return all_pairs
