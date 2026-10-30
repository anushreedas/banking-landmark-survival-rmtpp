import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import f1_score

class CostSensitiveThresholdOptimizer:
    """
    Optimizes class-prior multipliers w = (w_0, w_1, w_2, w_3) on out-of-fold validation
    probability predictions to maximize 4-class Macro-F1:
        y_hat = argmax_c (w_c * P(Y = c | x))
    Solves the severe minority-class recall collapse under default argmax rules.
    """
    def __init__(self, n_classes: int = 4, n_restarts: int = 8, random_state: int = 42):
        self.n_classes = n_classes
        self.n_restarts = n_restarts
        self.random_state = random_state
        self.weights_ = np.ones(n_classes, dtype=np.float64)

    def fit(self, oof_proba: np.ndarray, y_true: np.ndarray) -> "CostSensitiveThresholdOptimizer":
        rng = np.random.default_rng(self.random_state)

        # Initialize inverse class frequency prior as starting candidate
        counts = np.bincount(y_true, minlength=self.n_classes).astype(np.float64)
        inv_prior = (len(y_true) / np.maximum(counts, 1.0)) ** 0.5
        inv_prior /= inv_prior[0]

        best_w = np.ones(self.n_classes, dtype=np.float64)
        best_score = f1_score(y_true, np.argmax(oof_proba, axis=1), average="macro", zero_division=0)

        candidates = [np.ones(self.n_classes), inv_prior]
        for _ in range(self.n_restarts):
            candidates.append(inv_prior * rng.uniform(0.4, 2.5, size=self.n_classes))

        def neg_macro_f1(w_free):
            w = np.concatenate([[1.0], np.maximum(w_free, 0.05)])
            scaled = oof_proba * w[None, :]
            preds = np.argmax(scaled, axis=1)
            return -f1_score(y_true, preds, average="macro", zero_division=0)

        for init_w in candidates:
            res = minimize(
                neg_macro_f1,
                x0=init_w[1:],
                method="Nelder-Mead",
                options={"maxiter": 300, "xatol": 1e-2, "fatol": 1e-4},
            )
            score = -float(res.fun)
            if score > best_score:
                best_score = score
                best_w = np.concatenate([[1.0], np.maximum(res.x, 0.05)])

        self.weights_ = best_w
        return self

    def predict(self, proba: np.ndarray) -> np.ndarray:
        scaled = proba * self.weights_[None, :]
        return np.argmax(scaled, axis=1)
