# Continuous-Time Next-Best-Product Recommendation (`banking-landmark-survival-rmtpp`)

A leakage-free experimental framework benchmarking **Static Tabular Classifiers (ElasticNet, LightGBM)**, **Competing-Risk Survival Models (Cause-Specific Cox PH, Random Survival Forest)**, and **Continuous-Time Neural Sequence Models (PyTorch RMTPP-GRU)** on the PKDD'99 Czech Financial Dataset (Berka).

---

## Repository Architecture

```text
banking-landmark-survival-rmtpp/
├── assets/                          # Equation cards & diagnostic report plots
│   ├── eq1_landmark_window.png
│   ├── eq2_competing_risks.png
│   ├── eq3_rmtpp_objective.png
│   ├── fig_horizon_cohort_effect.png
│   ├── fig_cumulative_incidence.png
│   ├── fig_features_by_class.png
│   ├── fig_roc_pr.png
│   └── fig_recall_before_after.png
├── data/                            # Auto-downloader / extractor for Berka ASCII tables (*.asc)
│   └── download_berka.py
├── src/
│   ├── __init__.py
│   ├── preprocessing/               # Strictly causal data & feature pipelines
│   │   ├── __init__.py
│   │   ├── landmark_spine.py        # 90-day cohort windowing & right-censoring logic
│   │   ├── tabular_features.py      # Leakage-free behavioral & demographic aggregates
│   │   └── sequence_builder.py      # Event mark tokenization & log(1 + delta_t) padding
│   ├── models/                      # Standardized estimators across 3 paradigms
│   │   ├── __init__.py
│   │   ├── tabular_learners.py      # Class-weighted ElasticNet & LightGBM wrappers
│   │   ├── survival_models.py       # Cause-Specific Cox PH & Random Survival Forest
│   │   └── rmtpp_gru.py             # PyTorch RMTPP-style GRU encoder & multi-task head
│   └── optimization/                # Evaluation & post-processing
│       ├── __init__.py
│       ├── temporal_cv.py           # Chronological cohort splits (1993-1996 vs. 1997)
│       └── threshold_tuning.py      # Out-of-fold cost-sensitive prior & decision rules
├── run_pipeline.py                  # End-to-end training, evaluation & plotting entrypoint
├── requirements.txt
└── README.md
```

---

## Key Methodological Highlights

1. **Strict 90-Day Landmark Observation Window:** Every account opened at $t_{0,u}$ is observed over $\mathcal{W}_u = [t_{0,u}, t_{0,u} + 90\text{ days}]$. Accounts with product adoption inside the first 90 days are excluded to prevent look-ahead bias, and `order.asc` (which lacks creation timestamps) is excluded from landmark features.
2. **Global Chronological Split:** Models are trained and cross-validated on the **1993–1996 account opening cohorts** and evaluated on the held-out **1997 cohort**.
3. **Competing-Risk & Censoring Calibration:** Post-landmark outcomes (`None`, `Loan_Only`, `Card_Only`, `Both`) account for exact right-censoring durations up to `1998-12-31`.
4. **Cost-Sensitive Threshold Optimization:** Decouples probability ranking from classification rules, lifting 4-class Macro-F1 from `0.217` (default `argmax`) to `0.432` (Random Survival Forest with tuned thresholds, Macro ROC-AUC `0.821`).

---

## Quickstart

```bash
pip install -r requirements.txt
python data/download_berka.py --data_dir data/raw
python run_pipeline.py --data_dir data/raw --output_dir assets/
```
