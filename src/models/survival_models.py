import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.util import Surv
    HAS_SKSURV = True
except ImportError:
    HAS_SKSURV = False


class CauseSpecificCoxCompetingRisks:
    """
    Fits Cause-Specific Cox Proportional Hazards models for each competing adoption class c in {1, 2, 3}.
    Computes cumulative cause-specific hazards and converts them into 4-class adoption probabilities
    over a target evaluation horizon tau (in days).
    """
    def __init__(self, alpha: float = 10.0, horizon_days: float = 365.0):
        self.alpha = alpha
        self.horizon_days = horizon_days
        self.scaler = StandardScaler()
        self.cause_models = {}
        self.baseline_rates = {}

    def fit(self, X: np.ndarray, y_class: np.ndarray, duration_days: np.ndarray) -> "CauseSpecificCoxCompetingRisks":
        X_scaled = self.scaler.fit_transform(X)
        total_time = np.maximum(duration_days.sum(), 1.0)

        for c in [1, 2, 3]:
            event_c = (y_class == c)
            self.baseline_rates[c] = float(event_c.sum()) / total_time
            if HAS_SKSURV and event_c.sum() >= 3:
                y_surv = Surv.from_arrays(event=event_c, time=np.maximum(duration_days, 1.0))
                cox = CoxPHSurvivalAnalysis(alpha=self.alpha)
                cox.fit(X_scaled, y_surv)
                self.cause_models[c] = ("sksurv", cox)
            else:
                # Fallback ridge log-hazard approximator if scikit-survival is unavailable
                from sklearn.linear_model import LogisticRegression
                clf = LogisticRegression(C=1.0 / max(self.alpha, 1e-3), class_weight="balanced", max_iter=500)
                clf.fit(X_scaled, event_c.astype(int))
                self.cause_models[c] = ("fallback", clf)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        n = X_scaled.shape[0]
        cum_hazards = np.zeros((n, 3), dtype=np.float64)

        for idx, c in enumerate([1, 2, 3]):
            mode, model = self.cause_models[c]
            if mode == "sksurv":
                surv_funcs = model.predict_survival_function(X_scaled)
                for i, fn in enumerate(surv_funcs):
                    t_eval = min(self.horizon_days, fn.x[-1])
                    t_eval = max(t_eval, fn.x[0])
                    s_val = np.clip(fn(t_eval), 1e-6, 1.0)
                    cum_hazards[i, idx] = -np.log(s_val)
            else:
                p_c = np.clip(model.predict_proba(X_scaled)[:, 1], 1e-5, 1.0 - 1e-5)
                cum_hazards[:, idx] = -np.log(1.0 - p_c) * (self.baseline_rates[c] * self.horizon_days + 0.25)

        total_H = cum_hazards.sum(axis=1, keepdims=True)
        surv_overall = np.exp(-total_H)  # P(None) up to horizon
        event_prob_total = 1.0 - surv_overall

        cause_shares = cum_hazards / np.maximum(total_H, 1e-8)
        cause_probs = event_prob_total * cause_shares
        return np.hstack([surv_overall, cause_probs])

    def predict_risk_score(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return 1.0 - proba[:, 0]


class RandomSurvivalForestCompetingRisks:
    """
    Non-linear Random Survival Forest wrapper combining right-censored survival ensembles
    with cause-specific risk allocation across (`None`, `Loan_Only`, `Card_Only`, `Both`).
    """
    def __init__(
        self,
        n_estimators: int = 200,
        min_samples_split: int = 15,
        min_samples_leaf: int = 8,
        horizon_days: float = 365.0,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.horizon_days = horizon_days
        self.random_state = random_state
        self.rsf = None
        self.cause_classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y_class: np.ndarray, duration_days: np.ndarray) -> "RandomSurvivalForestCompetingRisks":
        event_any = (y_class > 0)
        # Apply inverse censoring / follow-up weighting to cause classifier
        weights = np.where(event_any, 1.0, np.minimum(duration_days / self.horizon_days, 1.0))
        self.cause_classifier.fit(X, y_class, sample_weight=weights)

        if HAS_SKSURV:
            y_surv = Surv.from_arrays(event=event_any, time=np.maximum(duration_days, 1.0))
            self.rsf = RandomSurvivalForest(
                n_estimators=self.n_estimators,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                n_jobs=-1,
            )
            self.rsf.fit(X, y_surv)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        rf_proba = self.cause_classifier.predict_proba(X)
        if self.rsf is not None:
            surv_funcs = self.rsf.predict_survival_function(X)
            p_none_surv = np.zeros(len(X), dtype=np.float64)
            for i, fn in enumerate(surv_funcs):
                t_eval = min(self.horizon_days, fn.x[-1])
                t_eval = max(t_eval, fn.x[0])
                p_none_surv[i] = np.clip(fn(t_eval), 1e-4, 1.0 - 1e-4)

            p_adopt_surv = 1.0 - p_none_surv
            cause_cond = rf_proba[:, 1:] / np.maximum(rf_proba[:, 1:].sum(axis=1, keepdims=True), 1e-8)
            out = np.zeros_like(rf_proba)
            out[:, 0] = 0.5 * p_none_surv + 0.5 * rf_proba[:, 0]
            out[:, 1:] = (1.0 - out[:, [0]]) * cause_cond
            return out
        return rf_proba

    def predict_risk_score(self, X: np.ndarray) -> np.ndarray:
        if self.rsf is not None:
            return self.rsf.predict(X)
        proba = self.predict_proba(X)
        return 1.0 - proba[:, 0]
