import numpy as np
import polars as pl
import torch
from torch import nn

from evaluate import compute_pr_auc
from models import BaseSpikeModel, TargetConfig, register_model
from models.ofi import OFILinearGLM

DIRS = ("buyR", "sellR")


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _Net(nn.Module):
    def __init__(self, n_in: int, hidden: int):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
        self.head = nn.Linear(hidden, len(DIRS))  # one logit per direction

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


@register_model("lstm")
class LSTMModel(BaseSpikeModel):
    """Single-layer LSTM over the last `seq_len` clusters of the OFI feature set.

    Same features as ofi.py (union of the buyR and sellR columns), windows never
    cross a chunk boundary (zero-padded at chunk start), one shared network with
    a buyR and a sellR logit, masked BCE so each direction only learns from its
    own valid rows. The epoch with the best pooled val PR-AUC is kept.
    """

    def __init__(
        self,
        seq_len: int = 32,
        hidden: int = 32,
        epochs: int = 3,
        batch_size: int = 1024,
        lr: float = 1e-3,
        seed: int = 0,
    ):
        super().__init__()
        self.seq_len, self.hidden, self.epochs = seq_len, hidden, epochs
        self.batch_size, self.lr, self.seed = batch_size, lr, seed
        self.ofi = OFILinearGLM()
        self.cols = list(
            dict.fromkeys(c for d in DIRS for c in OFILinearGLM._feature_cols(d))
        )
        self.offsets = torch.arange(-seq_len + 1, 1)
        self.device = pick_device()

    def _raw(self, feat: pl.DataFrame) -> np.ndarray:
        X = feat.select(self.cols).to_numpy().astype(np.float32)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    def _windows(self, feat: pl.DataFrame) -> tuple[torch.Tensor, np.ndarray]:
        """Normalised features, zero-padded by seq_len-1 rows before every chunk;
        pos[i] is row i's position, so its window is Xp[pos[i]-L+1 : pos[i]+1]."""
        X = np.clip((self._raw(feat) - self.mean) / self.std, -10.0, 10.0)
        pad = self.seq_len - 1
        new_chunk = feat["chunk_id"] != feat["chunk_id"].shift(1)
        rank = new_chunk.fill_null(True).cum_sum().to_numpy().astype(np.int64)
        pos = np.arange(len(X)) + rank * pad
        Xp = np.zeros((len(X) + rank[-1] * pad, X.shape[1]), np.float32)
        Xp[pos] = X
        return torch.from_numpy(Xp), pos

    def _batch(self, Xp: torch.Tensor, pos: np.ndarray) -> torch.Tensor:
        return Xp[torch.from_numpy(pos)[:, None] + self.offsets].to(self.device)

    def _targets(self, feat, cfg, split):
        y = np.stack([feat[cfg.target_col(d)].fill_null(0).to_numpy() for d in DIRS], 1)
        m = np.stack(
            [feat.select(self.valid_mask_expr(cfg, d, split)).to_series().to_numpy() for d in DIRS],
            1,
        )
        return y.astype(np.float32), m

    @torch.no_grad()
    def _predict(self, Xp: torch.Tensor, pos: np.ndarray) -> np.ndarray:
        self.net.eval()
        bs = self.batch_size * 4
        out = [
            torch.sigmoid(self.net(self._batch(Xp, pos[i : i + bs]))).cpu()
            for i in range(0, len(pos), bs)
        ]
        return torch.cat(out).numpy()

    def fit(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        cfg: TargetConfig,
    ) -> "LSTMModel":
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        tr = self.ofi._build_features(train_df, cfg)
        va = self.ofi._build_features(val_df, cfg)
        y_tr, m_tr = self._targets(tr, cfg, "train")
        y_va, m_va = self._targets(va, cfg, "val")
        rows, va_rows = np.flatnonzero(m_tr.any(1)), np.flatnonzero(m_va.any(1))

        X = self._raw(tr)[rows]
        self.mean = X.mean(0)
        self.std = np.where(X.std(0) > 1e-8, X.std(0), 1.0).astype(np.float32)
        Xp_tr, pos_tr = self._windows(tr)
        Xp_va, pos_va = self._windows(va)

        self.net = _Net(len(self.cols), self.hidden).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        best, state = -1.0, None
        for ep in range(self.epochs):
            self.net.train()
            for b in np.array_split(rng.permutation(rows), max(1, len(rows) // self.batch_size)):
                yb = torch.from_numpy(y_tr[b]).to(self.device)
                mb = torch.from_numpy(m_tr[b]).float().to(self.device)
                logits = self.net(self._batch(Xp_tr, pos_tr[b]))
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, yb, reduction="none"
                )
                loss = (loss * mb).sum() / mb.sum().clamp(min=1.0)
                opt.zero_grad()
                loss.backward()
                opt.step()

            p, m = self._predict(Xp_va, pos_va[va_rows]), m_va[va_rows]
            score = compute_pr_auc(y_va[va_rows][m], p[m])
            print(f"    [lstm] epoch {ep + 1}/{self.epochs}: val PR-AUC={score:.4f}")
            if score > best:
                best = score
                state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
        if state is not None:
            self.net.load_state_dict(state)
        return self

    def predict_proba(self, df: pl.DataFrame, cfg: TargetConfig) -> pl.DataFrame:
        feat = self.ofi._build_features(df, cfg)
        p = self._predict(*self._windows(feat))
        return pl.DataFrame(
            {
                "chunk_id": feat["chunk_id"],
                "event_seq": feat["event_seq"],
                "prob_buyR": p[:, 0],
                "prob_sellR": p[:, 1],
            }
        )
