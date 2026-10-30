import pandas as pd
import numpy as np
from typing import Tuple, List

def clean_district_table(district_df: pd.DataFrame) -> pd.DataFrame:
    """Clean Berka district demographic table (A1..A16) and handle '?' missing tokens."""
    d = district_df.copy()
    rename_map = {
        "A1": "district_id",
        "A4": "inhabitants",
        "A10": "urban_ratio",
        "A11": "avg_salary",
        "A12": "unemp_95",
        "A13": "unemp_96",
        "A14": "entrepreneurs_per_1k",
    }
    d = d.rename(columns={k: v for k, v in rename_map.items() if k in d.columns})
    keep_cols = [c for c in rename_map.values() if c in d.columns]
    d = d[keep_cols].copy()

    for col in keep_cols:
        if col != "district_id":
            d[col] = pd.to_numeric(d[col].replace("?", np.nan), errors="coerce")
            d[col] = d[col].fillna(d[col].median())
    return d

def build_landmark_tabular_features(
    trans_90d: pd.DataFrame,
    spine: pd.DataFrame,
    district_df: pd.DataFrame = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Aggregate 90-day transaction history strictly within [t_0, t_0 + 90].
    Captures volume, balance volatility, cash-flow ratios, temporal inter-arrival gaps,
    account metadata, and district socio-economic context.
    """
    df = trans_90d.sort_values(["account_id", "trans_date", "trans_id"]).copy()

    # Inter-transaction waiting times (delta_t in days)
    df["prev_date"] = df.groupby("account_id")["trans_date"].shift(1)
    df["delta_t"] = (df["trans_date"] - df["prev_date"]).dt.days.fillna(0.0).clip(lower=0.0)

    df["is_credit"] = (df["type"] == "PRIJEM").astype(int)
    df["is_debit"] = (df["type"].isin(["VYDAJ", "VYBER"])).astype(int)
    df["is_cash"] = (df["operation"].isin(["VKLAD", "VYBER"])).astype(int)
    df["is_bank_transfer"] = (df["operation"].isin(["PREVOD Z UCTU", "PREVOD NA UCET"])).astype(int)

    agg = (
        df.groupby("account_id")
        .agg(
            tx_count=("trans_id", "count"),
            credit_count=("is_credit", "sum"),
            debit_count=("is_debit", "sum"),
            cash_tx_count=("is_cash", "sum"),
            transfer_tx_count=("is_bank_transfer", "sum"),
            amt_sum=("amount", "sum"),
            amt_mean=("amount", "mean"),
            amt_std=("amount", "std"),
            amt_max=("amount", "max"),
            bal_min=("balance", "min"),
            bal_max=("balance", "max"),
            bal_mean=("balance", "mean"),
            bal_first=("balance", "first"),
            bal_last=("balance", "last"),
            bal_std=("balance", "std"),
            gap_mean=("delta_t", "mean"),
            gap_max=("delta_t", "max"),
            gap_std=("delta_t", "std"),
            active_days=("trans_date", "nunique"),
        )
        .reset_index()
    )

    # Momentum, velocity & ratio features
    agg["credit_debit_ratio"] = agg["credit_count"] / np.maximum(agg["debit_count"], 1.0)
    agg["bal_drawdown"] = agg["bal_last"] - agg["bal_min"]
    agg["bal_net_change"] = agg["bal_last"] - agg["bal_first"]
    agg["bal_range"] = agg["bal_max"] - agg["bal_min"]
    agg["log_amt_sum"] = np.log1p(np.maximum(agg["amt_sum"], 0.0))

    merged = spine.merge(agg, on="account_id", how="left")

    # Encode account & owner demographics
    merged["is_female"] = (merged["gender"] == "F").astype(float)
    freq_dummies = pd.get_dummies(merged["frequency"], prefix="freq", dtype=float)
    merged = pd.concat([merged, freq_dummies], axis=1)

    if district_df is not None:
        clean_dist = clean_district_table(district_df)
        merged = merged.merge(clean_dist, on="district_id", how="left")

    exclude_cols = {
        "account_id", "district_id", "client_id", "owner_disp_id",
        "frequency", "open_date", "landmark_date", "cohort_year",
        "gender", "birth_date", "loan_date", "card_date",
        "target_name", "target_class", "event_observed", "event_date", "duration_days",
    }
    feature_cols = [c for c in merged.columns if c not in exclude_cols]
    merged[feature_cols] = merged[feature_cols].fillna(0.0)

    return merged, feature_cols
