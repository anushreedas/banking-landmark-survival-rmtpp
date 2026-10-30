import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

torch.set_num_threads(1)


class BerkaRMTPPDataset(Dataset):
    """PyTorch Dataset wrapping event sequences, static context, class labels, and survival times."""
    def __init__(
        self,
        marks: np.ndarray,
        cont_feats: np.ndarray,
        seq_lengths: np.ndarray,
        static_feats: np.ndarray,
        y_class: np.ndarray,
        event_observed: np.ndarray,
        duration_days: np.ndarray,
    ):
        self.marks = torch.tensor(marks.tolist(), dtype=torch.long)
        self.cont_feats = torch.tensor(cont_feats.tolist(), dtype=torch.float32)
        self.seq_lengths = torch.tensor(seq_lengths.tolist(), dtype=torch.long)
        self.static_feats = torch.tensor(static_feats.tolist(), dtype=torch.float32)
        self.y_class = torch.tensor(y_class.tolist(), dtype=torch.long)
        self.event_observed = torch.tensor(event_observed.tolist(), dtype=torch.float32)
        self.duration_days = torch.tensor(duration_days.tolist(), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y_class)

    def __getitem__(self, idx: int):
        return (
            self.marks[idx],
            self.cont_feats[idx],
            self.seq_lengths[idx],
            self.static_feats[idx],
            self.y_class[idx],
            self.event_observed[idx],
            self.duration_days[idx],
        )


class BerkaRMTPPGRU(nn.Module):
    """
    RMTPP-style GRU encoder with a multi-task head for 4-class portfolio adoption
    and parametric continuous-time hazard/arrival intensity estimation.
    """
    def __init__(
        self,
        num_marks: int,
        emb_dim: int = 16,
        cont_dim: int = 3,
        static_dim: int = 20,
        hidden_dim: int = 64,
        num_classes: int = 4,
    ):
        super().__init__()
        self.mark_emb = nn.Embedding(num_marks, emb_dim, padding_idx=0)

        # Input per step: mark embedding + continuous tx features (log_amount, balance, log_delta_t)
        self.gru = nn.GRU(
            input_size=emb_dim + cont_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # Combine final sequence hidden state h_T with static demographic/district context
        combined_dim = hidden_dim + static_dim
        self.shared_proj = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

        # Head 1: 4-class portfolio propensity logits (None, Loan_Only, Card_Only, Both)
        self.class_head = nn.Linear(hidden_dim, num_classes)

        # Head 2: Continuous-time log-intensity / hazard scale parameter for time-to-adoption
        self.intensity_head = nn.Linear(hidden_dim, 1)

    def forward(self, marks, cont_feats, seq_lengths, static_feats):
        emb = self.mark_emb(marks)
        x = torch.cat([emb, cont_feats], dim=-1)

        packed = nn.utils.rnn.pack_padded_sequence(
            x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)
        h_last = h_n[-1]  # [batch_size, hidden_dim]

        context = self.shared_proj(torch.cat([h_last, static_feats], dim=-1))
        class_logits = self.class_head(context)
        log_hazard = self.intensity_head(context).squeeze(-1)

        return class_logits, log_hazard


def rmtpp_multitask_loss(
    class_logits: torch.Tensor,
    log_hazard: torch.Tensor,
    y_class: torch.Tensor,
    event_observed: torch.Tensor,
    duration_days: torch.Tensor,
    class_weights: torch.Tensor,
    tpp_weight: float = 0.25,
) -> torch.Tensor:
    """
    Equation 3: Joint Multi-Task RMTPP-GRU Objective combining class-weighted
    cross-entropy with continuous-time point-process Negative Log-Likelihood.
    """
    ce_loss = nn.functional.cross_entropy(class_logits, y_class, weight=class_weights)

    # Scale duration to years so hazard rates stay numerically well-conditioned
    t_years = torch.clamp(duration_days / 365.25, min=1e-3)
    hazard = torch.exp(torch.clamp(log_hazard, min=-10.0, max=5.0))

    # Point-process NLL: - [ delta_u * log lambda_u(T_u) - int_0^{T_u} lambda_u(tau) d_tau ]
    nll_tpp = -(event_observed * torch.log(hazard + 1e-8) - hazard * t_years).mean()
    return ce_loss + tpp_weight * nll_tpp


class RMTPPTrainerWrapper:
    """High-level scikit-learn style wrapper for training and evaluating BerkaRMTPPGRU."""
    def __init__(
        self,
        num_marks: int,
        emb_dim: int = 16,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 30,
        batch_size: int = 64,
        tpp_weight: float = 0.25,
        random_state: int = 42,
    ):
        self.num_marks = num_marks
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.tpp_weight = tpp_weight
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.device = torch.device("cpu")

    def fit(
        self,
        marks: np.ndarray,
        cont_feats: np.ndarray,
        seq_lengths: np.ndarray,
        static_feats: np.ndarray,
        y_class: np.ndarray,
        duration_days: np.ndarray,
    ) -> "RMTPPTrainerWrapper":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        static_scaled = self.scaler.fit_transform(static_feats)
        event_obs = (y_class > 0).astype(np.float32)

        classes = np.unique(y_class)
        cw = [1.0, 1.0, 1.0, 1.0]
        computed = compute_class_weight("balanced", classes=classes, y=y_class)
        for c, w in zip(classes, computed):
            cw[int(c)] = float(w)
        cw_tensor = torch.tensor(cw, dtype=torch.float32, device=self.device)

        dataset = BerkaRMTPPDataset(
            marks, cont_feats, seq_lengths, static_scaled, y_class, event_obs, duration_days
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = BerkaRMTPPGRU(
            num_marks=self.num_marks,
            emb_dim=self.emb_dim,
            cont_dim=cont_feats.shape[-1],
            static_dim=static_scaled.shape[-1],
            hidden_dim=self.hidden_dim,
            num_classes=4,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        self.model.train()
        for _ in range(self.epochs):
            for b_marks, b_cont, b_len, b_stat, b_y, b_ev, b_dur in loader:
                b_marks = b_marks.to(self.device)
                b_cont = b_cont.to(self.device)
                b_stat = b_stat.to(self.device)
                b_y = b_y.to(self.device)
                b_ev = b_ev.to(self.device)
                b_dur = b_dur.to(self.device)

                optimizer.zero_grad()
                logits, log_haz = self.model(b_marks, b_cont, b_len, b_stat)
                loss = rmtpp_multitask_loss(
                    logits, log_haz, b_y, b_ev, b_dur, cw_tensor, self.tpp_weight
                )
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
                optimizer.step()

        return self

    @torch.no_grad()
    def predict_proba_and_hazard(
        self,
        marks: np.ndarray,
        cont_feats: np.ndarray,
        seq_lengths: np.ndarray,
        static_feats: np.ndarray,
    ):
        self.model.eval()
        static_scaled = self.scaler.transform(static_feats)
        dummy_y = np.zeros(len(marks), dtype=np.int64)
        dummy_ev = np.zeros(len(marks), dtype=np.float32)
        dummy_dur = np.ones(len(marks), dtype=np.float32)

        dataset = BerkaRMTPPDataset(
            marks, cont_feats, seq_lengths, static_scaled, dummy_y, dummy_ev, dummy_dur
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_probs, all_haz = [], []
        for b_marks, b_cont, b_len, b_stat, _, _, _ in loader:
            b_marks = b_marks.to(self.device)
            b_cont = b_cont.to(self.device)
            b_stat = b_stat.to(self.device)
            logits, log_haz = self.model(b_marks, b_cont, b_len, b_stat)
            probs = np.array(torch.softmax(logits, dim=-1).cpu().tolist(), dtype=np.float64)
            haz = np.array(torch.exp(log_haz).cpu().tolist(), dtype=np.float64)
            all_probs.append(probs)
            all_haz.append(haz)

        return np.vstack(all_probs), np.concatenate(all_haz)
