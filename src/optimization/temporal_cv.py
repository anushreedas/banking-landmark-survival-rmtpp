import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from typing import Tuple, Dict

def chronological_cohort_split(
    df: pd.DataFrame,
    train_years: Tuple[int, ...] = (1993, 1994, 1995, 1996),
    test_years: Tuple[int, ...] = (1997,),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Enforces the Global Chronological Cohort Split:
    - 1993-1996 account opening cohorts -> Training & Cross-Validation
    - 1997 account opening cohort -> Held-Out Evaluation Test Set
    """
    train_mask = df["cohort_year"].isin(train_years).values
    test_mask = df["cohort_year"].isin(test_years).values
    return np.where(train_mask)[0], np.where(test_mask)[0]

def get_stratified_cv_splits(y_train: np.ndarray, n_splits: int = 5, random_state: int = 42):
    """Returns stratified folds within the 1993-1996 training cohorts."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(skf.split(np.zeros(len(y_train)), y_train))

def compute_harrell_c_index(
    event_observed: np.ndarray,
    duration_days: np.ndarray,
    risk_scores: np.ndarray,
) -> float:
    """
    Computes Harrell's Concordance Index (C-index) for right-censored time-to-event data.
    Higher risk_scores should correspond to shorter time-to-adoption among uncensored events.
    """
    n = len(duration_days)
    concordant = 0.0
    permissible = 0.0

    for i in range(n):
        if event_observed[i] != 1:
            continue
        t_i = duration_days[i]
        r_i = risk_scores[i]
        # Compare against all j whose follow-up time exceeds t_i
        mask_j = (duration_days > t_i) | ((duration_days == t_i) & (event_observed == 0))
        n_j = mask_j.sum()
        if n_j == 0:
            continue
        r_j = risk_scores[mask_j]
        permissible += n_j
        concordant += (r_i > r_j).sum() + 0.5 * (r_i == r_j).sum()

    return float(concordant / permissible) if permissible > 0 else 0.5

def evaluate_multiclass_and_survival(
    y_true: np.ndarray,
    proba: np.ndarray,
    y_pred_tuned: np.ndarray,
    event_observed: np.ndarray,
    duration_days: np.ndarray,
    risk_scores: np.ndarray,
) -> Dict[str, float]:
    """Computes One-vs-Rest Macro ROC-AUC, PR-AUC, Default vs. Tuned Macro-F1, and Harrell's C-index."""
    y_pred_default = np.argmax(proba, axis=1)
    f1_default = f1_score(y_true, y_pred_default, average="macro", zero_division=0)
    f1_tuned = f1_score(y_true, y_pred_tuned, average="macro", zero_division=0)

    roc_aucs, pr_aucs = [], []
    for c in range(4):
        y_bin = (y_true == c).astype(int)
        if 0 < y_bin.sum() < len(y_bin):
            roc_aucs.append(roc_auc_score(y_bin, proba[:, c]))
            pr_aucs.append(average_precision_score(y_bin, proba[:, c]))

    c_idx = compute_harrell_c_index(event_observed, duration_days, risk_scores)
    return {
        "macro_roc_auc": float(np.mean(roc_aucs)) if roc_aucs else 0.5,
        "macro_pr_auc": float(np.mean(pr_aucs)) if pr_aucs else 0.25,
        "macro_f1_default": float(f1_default),
        "macro_f1_tuned": float(f1_tuned),
        "c_index": float(c_idx),
    }
