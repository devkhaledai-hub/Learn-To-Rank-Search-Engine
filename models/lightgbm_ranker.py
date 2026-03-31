import joblib
import lightgbm as lgb

from models.base_model import BaseModel
from utils.logging import get_logger
from config.settings import RANKER_PARAMS, EARLY_STOPPING_ROUNDS, EVAL_AT

logger = get_logger(__name__)


class LightGBMRanker(BaseModel):
    """
    LightGBM LambdaRank model for Learning-to-Rank.

    Uses LGBMRanker with early stopping and NDCG evaluation.
    """

    def __init__(self, params: dict = None):
        self.params = params or RANKER_PARAMS.copy()
        self.model = None

    def train(
        self,
        X_train, y_train, groups_train=None,
        X_val=None, y_val=None, groups_val=None,
    ):
        """
        Train the ranker.

        Parameters
        ----------
        groups_train / groups_val : list[int]
            Number of documents per query (required by LGBMRanker).
        """
        logger.info("Initialising LGBMRanker with params: %s", self.params)
        self.model = lgb.LGBMRanker(**self.params)

        fit_kwargs = dict(
            X=X_train,
            y=y_train,
            group=groups_train,
            callbacks=[
                lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
                lgb.log_evaluation(period=50),
            ],
        )

        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["eval_group"] = [groups_val]
            fit_kwargs["eval_names"] = ["Validation"]
            fit_kwargs["eval_at"] = EVAL_AT

        logger.info("Starting LGBMRanker training...")
        self.model.fit(**fit_kwargs)
        logger.info("Training complete.")

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model is not trained or loaded yet.")
        return self.model.predict(X)

    def save(self, path: str):
        if self.model is None:
            raise ValueError("No model to save.")
        joblib.dump(self.model, path)
        logger.info("Model saved to %s", path)

    def load(self, path: str):
        self.model = joblib.load(path)
        logger.info("Model loaded from %s", path)

    def feature_importances(self, feature_names: list) -> list:
        """Return sorted (feature_name, importance) pairs."""
        if self.model is None:
            raise ValueError("Model is not trained or loaded yet.")
        pairs = list(zip(feature_names, self.model.feature_importances_))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs
