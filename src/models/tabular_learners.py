import warnings
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
import lightgbm as lgb


class ElasticNetClassifierWrapper:
    """Multinomial ElasticNet Logistic Regression with class-balanced sample weights."""
    def __init__(self, C: float = 0.5, l1_ratio: float = 0.5, random_state: int = 42):
        self.C = C
        self.l1_ratio = l1_ratio
        self.random_state = random_state

        # In scikit-learn >= 1.8, 'penalty' is deprecated in favor of setting l1_ratio directly
        sk_parts = [int(x) for x in sklearn.__version__.split(".")[:2] if x.isdigit()]
        lr_kwargs = dict(
            solver="saga",
            C=self.C,
            l1_ratio=self.l1_ratio,
            class_weight="balanced",
            max_iter=1500,
            random_state=self.random_state,
        )
        if sk_parts < [1, 8]:
            lr_kwargs["penalty"] = "elasticnet"

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**lr_kwargs)),
        ])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ElasticNetClassifierWrapper":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(X)

    def predict_risk_score(self, X: np.ndarray) -> np.ndarray:
        """Overall adoption risk score 1 - P(None) for concordance evaluation."""
        proba = self.predict_proba(X)
        return 1.0 - proba[:, 0]


class LightGBMClassifierWrapper:
    """Class-weighted LightGBM gradient boosted decision trees for 4-class NBP."""
    def __init__(
        self,
        n_estimators: int = 250,
        learning_rate: float = 0.03,
        num_leaves: int = 15,
        min_child_samples: int = 25,
        subsample_freq: int = 1,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 1.0,
        reg_lambda: float = 2.0,
        random_state: int = 42,
    ):
        self.model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=4,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            subsample=0.8,
            subsample_freq=subsample_freq,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            class_weight="balanced",
            n_jobs=1,  # Prevents OpenMP / libomp thread-pool collision with PyTorch on macOS
            random_state=random_state,
            verbose=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LightGBMClassifierWrapper":
        sw = compute_sample_weight("balanced", y)
        self.model.fit(X, y, sample_weight=sw)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict_risk_score(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return 1.0 - proba[:, 0]
