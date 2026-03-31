import numpy as np
from sklearn.metrics import ndcg_score

from policy.base_policy import BasePolicy
from datapipeline.feature_pipeline import FeaturePipeline
from models.lightgbm_ranker import LightGBMRanker
from utils.logging import get_logger
from config.settings import FEATURE_COLUMNS

logger = get_logger(__name__)


class InferencePolicy(BasePolicy):
    """
    Loads a trained LightGBM Ranker and runs predictions on
    query-document pairs, optionally evaluating NDCG.
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        self.feature_pipeline = FeaturePipeline()
        self.ranker = LightGBMRanker()
        self.load(model_path)

    # ── BasePolicy interface ─────────────────────────────────

    def train(self, data):
        """Not used during inference."""
        pass

    def predict(self, input_data):
        """
        Run prediction and NDCG evaluation on a feature-enriched parquet file.

        Parameters
        ----------
        input_data : str
            Path to a parquet file with columns:
            query_id, relevance, and all FEATURE_COLUMNS.

        Returns
        -------
        avg_ndcg : float
        predictions : np.ndarray
        """
        logger.info("Loading data from: %s", input_data)
        df = self.feature_pipeline.load_data(input_data)

        X = df[FEATURE_COLUMNS]
        predictions = self.ranker.predict(X)

        # Evaluate NDCG per query
        groups = df.groupby("query_id")
        ndcg_scores_list = []
        for _, group in groups:
            y_true = group["relevance"].values
            y_pred = predictions[group.index]
            if len(np.unique(y_true)) > 1:
                score = ndcg_score([y_true], [y_pred], k=10)
                ndcg_scores_list.append(score)

        avg_ndcg = np.mean(ndcg_scores_list) if ndcg_scores_list else 0.0
        logger.info("Average NDCG@10: %.4f", avg_ndcg)
        return avg_ndcg, predictions

    def save(self, model, path_to_model):
        """Not typically used during inference."""
        self.ranker.save(path_to_model)

    def load(self, path_to_model):
        logger.info("Loading model from %s", path_to_model)
        self.ranker.load(path_to_model)
