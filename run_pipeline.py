import os
# Prevent macOS OpenMP (libomp) thread-pool segfaults
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
import numpy as np
import pandas as pd

from data.download_berka import download_and_extract_berka
from src.preprocessing.landmark_spine import load_raw_tables, build_90d_landmark_spine
from src.preprocessing.tabular_features import build_landmark_tabular_features
from src.preprocessing.sequence_builder import BerkaEventSequenceBuilder
from src.models.tabular_learners import ElasticNetClassifierWrapper, LightGBMClassifierWrapper
from src.models.survival_models import (
    CauseSpecificCoxCompetingRisks,
    RandomSurvivalForestCompetingRisks,
)
from src.optimization.temporal_cv import (
    chronological_cohort_split,
    get_stratified_cv_splits,
    evaluate_multiclass_and_survival,
)
from src.optimization.threshold_tuning import CostSensitiveThresholdOptimizer


def run_benchmark(data_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    print("[1/5] Ensuring Berka ASCII tables are downloaded & extracted, then building 90-day spine...")
    download_and_extract_berka(data_dir=data_dir)
    tables = load_raw_tables(data_dir)
    spine, trans_90d = build_90d_landmark_spine(tables)

    print("[2/5] Engineering leakage-free 90-day tabular features & event sequences...")
    feat_df, feature_cols = build_landmark_tabular_features(
        trans_90d, spine, district_df=tables["district"]
    )
    seq_builder = BerkaEventSequenceBuilder(max_seq_len=64).fit(trans_90d)
    marks, cont_feats, seq_lengths = seq_builder.transform(
        trans_90d, feat_df["account_id"].tolist()
    )

    train_idx, test_idx = chronological_cohort_split(feat_df)
    X = feat_df[feature_cols].values.astype(np.float32)
    y = feat_df["target_class"].values.astype(np.int64)
    ev = feat_df["event_observed"].values.astype(np.int64)
    dur = feat_df["duration_days"].values.astype(np.float32)

    X_tr, y_tr, ev_tr, dur_tr = X[train_idx], y[train_idx], ev[train_idx], dur[train_idx]
    X_te, y_te, ev_te, dur_te = X[test_idx], y[test_idx], ev[test_idx], dur[test_idx]

    print(f"      Train cohorts (1993-1996): {len(train_idx)} accounts | Test cohort (1997): {len(test_idx)} accounts")

    models = {
        "ElasticNet": lambda: ElasticNetClassifierWrapper(C=0.5, l1_ratio=0.5),
        "LightGBM": lambda: LightGBMClassifierWrapper(),
        "CauseSpecificCox": lambda: CauseSpecificCoxCompetingRisks(alpha=10.0),
        "RandomSurvivalForest": lambda: RandomSurvivalForestCompetingRisks(n_estimators=200),
    }

    results = []
    splits = get_stratified_cv_splits(y_tr, n_splits=5)

    print("[3/5] Training & cross-validating Tabular and Competing-Risk Survival models...")
    for name, ctor in models.items():
        print(f"      -> Fitting & cross-validating {name}...")
        oof_proba = np.zeros((len(train_idx), 4), dtype=np.float64)
        for tr_f, val_f in splits:
            m = ctor()
            if name in ("CauseSpecificCox", "RandomSurvivalForest"):
                m.fit(X_tr[tr_f], y_tr[tr_f], dur_tr[tr_f])
            else:
                m.fit(X_tr[tr_f], y_tr[tr_f])
            oof_proba[val_f] = m.predict_proba(X_tr[val_f])

        tuner = CostSensitiveThresholdOptimizer().fit(oof_proba, y_tr)

        full_model = ctor()
        if name in ("CauseSpecificCox", "RandomSurvivalForest"):
            full_model.fit(X_tr, y_tr, dur_tr)
        else:
            full_model.fit(X_tr, y_tr)

        test_proba = full_model.predict_proba(X_te)
        test_risk = full_model.predict_risk_score(X_te)
        test_pred_tuned = tuner.predict(test_proba)

        metrics = evaluate_multiclass_and_survival(
            y_te, test_proba, test_pred_tuned, ev_te, dur_te, test_risk
        )
        metrics["model"] = name
        results.append(metrics)

    print("[4/5] Training PyTorch RMTPP-GRU continuous-time sequence model...")
    # Lazy import PyTorch AFTER LightGBM/sksurv finish so libomp.dylib and libiomp5.dylib never collide
    from src.models.rmtpp_gru import RMTPPTrainerWrapper

    rmtpp_oof = np.zeros((len(train_idx), 4), dtype=np.float64)
    for tr_f, val_f in splits:
        rmtpp = RMTPPTrainerWrapper(num_marks=len(seq_builder.mark_vocab), epochs=25)
        rmtpp.fit(
            marks[train_idx][tr_f],
            cont_feats[train_idx][tr_f],
            seq_lengths[train_idx][tr_f],
            X_tr[tr_f],
            y_tr[tr_f],
            dur_tr[tr_f],
        )
        p_val, _ = rmtpp.predict_proba_and_hazard(
            marks[train_idx][val_f],
            cont_feats[train_idx][val_f],
            seq_lengths[train_idx][val_f],
            X_tr[val_f],
        )
        rmtpp_oof[val_f] = p_val

    rmtpp_tuner = CostSensitiveThresholdOptimizer().fit(rmtpp_oof, y_tr)
    rmtpp_full = RMTPPTrainerWrapper(num_marks=len(seq_builder.mark_vocab), epochs=30)
    rmtpp_full.fit(
        marks[train_idx], cont_feats[train_idx], seq_lengths[train_idx], X_tr, y_tr, dur_tr
    )
    te_proba, te_haz = rmtpp_full.predict_proba_and_hazard(
        marks[test_idx], cont_feats[test_idx], seq_lengths[test_idx], X_te
    )
    te_pred_tuned = rmtpp_tuner.predict(te_proba)
    rmtpp_metrics = evaluate_multiclass_and_survival(
        y_te, te_proba, te_pred_tuned, ev_te, dur_te, te_haz
    )
    rmtpp_metrics["model"] = "RMTPP-GRU (PyTorch)"
    results.append(rmtpp_metrics)

    print("[5/5] Saving benchmark summary table...")
    res_df = pd.DataFrame(results)[
        ["model", "macro_roc_auc", "macro_pr_auc", "macro_f1_default", "macro_f1_tuned", "c_index"]
    ]
    out_csv = os.path.join(output_dir, "benchmark_results_1997_cohort.csv")
    res_df.to_csv(out_csv, index=False)
    print("\n" + res_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/raw")
    parser.add_argument("--output_dir", type=str, default="assets")
    args = parser.parse_args()
    run_benchmark(args.data_dir, args.output_dir)
