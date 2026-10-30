import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict

STUDY_END_DATE = pd.Timestamp("1998-12-31")
LANDMARK_DAYS = 90

LABEL_MAP = {
    "None": 0,
    "Loan_Only": 1,
    "Card_Only": 2,
    "Both": 3,
}

def parse_berka_date(s: pd.Series) -> pd.Series:
    """Convert YYMMDD integer/string column into standard pandas datetime."""
    cleaned = s.astype(str).str.slice(0, 6).str.zfill(6)
    return pd.to_datetime("19" + cleaned, format="%Y%m%d", errors="coerce")

def extract_client_demographics(client_df: pd.DataFrame) -> pd.DataFrame:
    """Decode gender and birth_date from Czech birth_number (YYMMDD / YY[MM+50]DD)."""
    bn = client_df["birth_number"].astype(str).str.zfill(6)
    year = 1900 + bn.str.slice(0, 2).astype(int)
    month = bn.str.slice(2, 4).astype(int)
    day = bn.str.slice(4, 6).astype(int)

    is_female = month > 50
    month = np.where(is_female, month - 50, month)

    out = client_df.copy()
    out["gender"] = np.where(is_female, "F", "M")
    out["birth_date"] = pd.to_datetime(
        {"year": year, "month": month, "day": day}, errors="coerce"
    )
    return out

def load_raw_tables(data_dir: str) -> Dict[str, pd.DataFrame]:
    """Load relational Berka tables (excluding order.asc to prevent leakage)."""
    tables = {}
    for name in ["account", "client", "disp", "trans", "loan", "card", "district"]:
        fpath = os.path.join(data_dir, f"{name}.asc")
        tables[name] = pd.read_csv(fpath, sep=";", low_memory=False)
    return tables

def build_90d_landmark_spine(
    tables: Dict[str, pd.DataFrame],
    landmark_days: int = LANDMARK_DAYS,
    study_end: pd.Timestamp = STUDY_END_DATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Constructs the leakage-free 90-day landmark cohort spine and filters the transaction
    table strictly to [t_0, t_0 + 90 days].

    Returns
    -------
    spine : pd.DataFrame
        One row per valid account with:
        - account_id, district_id, frequency, open_date, landmark_date, cohort_year
        - owner demographics (gender, age_at_open, has_disponent)
        - target_class (0..3), target_name, event_observed (0/1), duration_days (post-landmark)
    trans_90d : pd.DataFrame
        Transactions occurring strictly within [open_date, landmark_date] for valid cohort accounts.
    """
    account = tables["account"].copy()
    client = extract_client_demographics(tables["client"])
    disp = tables["disp"].copy()
    loan = tables["loan"].copy()
    card = tables["card"].copy()
    trans = tables["trans"].copy()

    account["open_date"] = parse_berka_date(account["date"])
    account["landmark_date"] = account["open_date"] + pd.Timedelta(days=landmark_days)
    account["cohort_year"] = account["open_date"].dt.year

    # Link OWNER and DISPONENT dispositions
    owners = disp[disp["type"] == "OWNER"][["account_id", "client_id", "disp_id"]].rename(
        columns={"disp_id": "owner_disp_id"}
    )
    disponents = (
        disp[disp["type"] != "OWNER"]
        .groupby("account_id")["disp_id"]
        .count()
        .gt(0)
        .astype(int)
        .rename("has_disponent")
        .reset_index()
    )

    spine = (
        account[["account_id", "district_id", "frequency", "open_date", "landmark_date", "cohort_year"]]
        .merge(owners, on="account_id", how="left")
        .merge(client[["client_id", "gender", "birth_date"]], on="client_id", how="left")
        .merge(disponents, on="account_id", how="left")
    )
    spine["has_disponent"] = spine["has_disponent"].fillna(0).astype(int)
    spine["age_at_open"] = (spine["open_date"] - spine["birth_date"]).dt.days / 365.25

    # Map loan dates
    loan["loan_date"] = parse_berka_date(loan["date"])
    first_loan = loan.groupby("account_id")["loan_date"].min().reset_index()

    # Map credit card dates via disp_id -> account_id
    card["card_date"] = parse_berka_date(card["issued"])
    card_acct = card.merge(disp[["disp_id", "account_id"]], on="disp_id", how="left")
    first_card = card_acct.groupby("account_id")["card_date"].min().reset_index()

    spine = spine.merge(first_loan, on="account_id", how="left").merge(
        first_card, on="account_id", how="left"
    )

    # Filter out accounts where landmark_date >= study_end or where adoption happened <= landmark_date
    early_loan = spine["loan_date"].notna() & (spine["loan_date"] <= spine["landmark_date"])
    early_card = spine["card_date"].notna() & (spine["card_date"] <= spine["landmark_date"])
    valid_horizon = spine["landmark_date"] < study_end

    spine = spine[valid_horizon & (~early_loan) & (~early_card)].copy()

    # Assign 4-class outcome over (landmark_date, study_end]
    has_loan = spine["loan_date"].notna() & (spine["loan_date"] > spine["landmark_date"])
    has_card = spine["card_date"].notna() & (spine["card_date"] > spine["landmark_date"])

    conditions = [
        has_loan & has_card,
        has_loan & (~has_card),
        (~has_loan) & has_card,
    ]
    choices = ["Both", "Loan_Only", "Card_Only"]
    spine["target_name"] = np.select(conditions, choices, default="None")
    spine["target_class"] = spine["target_name"].map(LABEL_MAP).astype(int)

    # Post-landmark time-to-first-adoption (or censoring duration)
    first_event_date = spine[["loan_date", "card_date"]].min(axis=1)
    spine["event_observed"] = (spine["target_class"] > 0).astype(int)
    spine["event_date"] = first_event_date.fillna(study_end)
    spine["duration_days"] = (spine["event_date"] - spine["landmark_date"]).dt.days.clip(lower=1)

    # Filter transactions strictly to [open_date, landmark_date]
    trans["trans_date"] = parse_berka_date(trans["date"])
    trans_merged = trans.merge(
        spine[["account_id", "open_date", "landmark_date"]], on="account_id", how="inner"
    )
    mask_90d = (trans_merged["trans_date"] >= trans_merged["open_date"]) & (
        trans_merged["trans_date"] <= trans_merged["landmark_date"]
    )
    trans_90d = trans_merged[mask_90d].copy()

    return spine.reset_index(drop=True), trans_90d.reset_index(drop=True)
