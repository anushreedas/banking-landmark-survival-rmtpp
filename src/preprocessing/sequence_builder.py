import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class BerkaEventSequenceBuilder:
    """
    Converts 90-day transaction streams into padded Marked Temporal Point Process (MTPP)
    event sequences: (mark_id, log_amount, scaled_balance, log_delta_t).
    Pure NumPy/Pandas implementation so PyTorch is only imported when RMTPP training begins.
    """
    def __init__(self, max_seq_len: int = 64):
        self.max_seq_len = max_seq_len
        self.mark_vocab: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.bal_mean: float = 0.0
        self.bal_std: float = 1.0

    def _make_mark_token(self, df: pd.DataFrame) -> pd.Series:
        t_type = df["type"].fillna("UNK_TYPE").astype(str)
        op = df["operation"].fillna("UNK_OP").astype(str)
        k_sym = df["k_symbol"].fillna("NONE").replace(" ", "NONE").astype(str)
        return t_type + "__" + op + "__" + k_sym

    def fit(self, trans_90d: pd.DataFrame) -> "BerkaEventSequenceBuilder":
        tokens = self._make_mark_token(trans_90d)
        for tok in sorted(tokens.unique()):
            if tok not in self.mark_vocab:
                self.mark_vocab[tok] = len(self.mark_vocab)
        self.bal_mean = float(trans_90d["balance"].mean())
        self.bal_std = float(max(trans_90d["balance"].std(), 1.0))
        return self

    def transform(
        self,
        trans_90d: pd.DataFrame,
        account_ids: List[int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns
        -------
        marks : np.ndarray of shape (N, max_seq_len), int64
        cont_feats : np.ndarray of shape (N, max_seq_len, 3), float32
            [log1p(amount), z_score(balance), log1p(delta_t)]
        seq_lengths : np.ndarray of shape (N,), int64
        """
        df = trans_90d.sort_values(["account_id", "trans_date", "trans_id"]).copy()
        df["mark_token"] = self._make_mark_token(df)
        df["mark_id"] = df["mark_token"].map(lambda x: self.mark_vocab.get(x, 1)).astype(np.int64)

        df["prev_date"] = df.groupby("account_id")["trans_date"].shift(1)
        df["delta_t"] = (df["trans_date"] - df["prev_date"]).dt.days.fillna(0.0).clip(lower=0.0)

        df["log_amount"] = np.log1p(np.maximum(df["amount"].fillna(0.0), 0.0))
        df["norm_balance"] = (df["balance"].fillna(self.bal_mean) - self.bal_mean) / self.bal_std
        df["log_delta_t"] = np.log1p(df["delta_t"])

        grouped = {
            acct_id: grp for acct_id, grp in df.groupby("account_id", sort=False)
        }

        n_accts = len(account_ids)
        marks = np.zeros((n_accts, self.max_seq_len), dtype=np.int64)
        cont_feats = np.zeros((n_accts, self.max_seq_len, 3), dtype=np.float32)
        seq_lengths = np.ones(n_accts, dtype=np.int64)

        for i, acct_id in enumerate(account_ids):
            if acct_id not in grouped:
                continue
            g = grouped[acct_id].tail(self.max_seq_len)
            L = len(g)
            if L == 0:
                continue
            seq_lengths[i] = L
            marks[i, :L] = g["mark_id"].values
            cont_feats[i, :L, 0] = g["log_amount"].values
            cont_feats[i, :L, 1] = g["norm_balance"].values
            cont_feats[i, :L, 2] = g["log_delta_t"].values

        return marks, cont_feats, seq_lengths
