import numpy as np
import polars as pl
import torch
from torch import nn
from torch.nn import functional as F

from evaluate import compute_pr_auc
from models import BaseSpikeModel, TargetConfig, register_model

DIRS = ("buyR", "sellR")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class _CTGRUNet(nn.Module):
    """Continuous-Time Decay GRU with input projection and instantaneous skip connection."""

    def __init__(self, n_in: int, hidden: int = 32, dropout: float = 0.2):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.gru_cell = nn.GRUCell(hidden, hidden)
        # Learnable per-channel continuous-time decay rates beta > 0 (in 1/sec)
        self.log_beta = nn.Parameter(torch.linspace(-2.0, 2.0, hidden))
        self.head = nn.Sequential(
            nn.Linear(hidden + n_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_seq: torch.Tensor, dt_sec: torch.Tensor) -> torch.Tensor:
        # x_seq: (B, L, F), dt_sec: (B, L)
        e_seq = self.in_proj(x_seq)
        decay_seq = torch.exp(-F.softplus(self.log_beta) * dt_sec.unsqueeze(-1))

        h = torch.zeros(
            x_seq.size(0), e_seq.size(-1), device=x_seq.device, dtype=x_seq.dtype
        )
        for t in range(x_seq.size(1)):
            h = self.gru_cell(e_seq[:, t], h * decay_seq[:, t])

        out = torch.cat([h, x_seq[:, -1]], dim=-1)
        return self.head(out).squeeze(-1)


@register_model("ct_gru")
class CTGRUModel(BaseSpikeModel):
    """Protocol-Aware Continuous-Time GRU over raw single-step event & snapshot sequences."""

    def __init__(
        self,
        seq_len: int = 32,
        hidden: int = 32,
        dropout: float = 0.2,
        epochs: int = 4,
        batch_size: int = 2048,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        seed: int = 0,
    ):
        super().__init__()
        self.seq_len, self.hidden, self.dropout = seq_len, hidden, dropout
        self.epochs, self.batch_size = epochs, batch_size
        self.lr, self.weight_decay, self.seed = lr, weight_decay, seed
        self.offsets = torch.arange(-seq_len + 1, 1)
        self.device = get_device()

    def _build_features(self, df: pl.DataFrame, cfg: TargetConfig) -> pl.DataFrame:
        H, eps = cfg.horizon, 1e-6

        # 1. Single-step L1 & Top-5 OFI + price steps within each continuous chunk
        prev_bid = pl.col("r_best_bid").shift(1).over("chunk_id")
        prev_ask = pl.col("r_best_ask").shift(1).over("chunk_id")
        prev_bsz = pl.col("r_bid_sz_1").shift(1).over("chunk_id")
        prev_asz = pl.col("r_ask_sz_1").shift(1).over("chunk_id")

        d_bid = (
            pl.when(pl.col("r_best_bid") >= prev_bid)
            .then(pl.col("r_bid_sz_1"))
            .otherwise(0.0)
            - pl.when(pl.col("r_best_bid") <= prev_bid).then(prev_bsz).otherwise(0.0)
        ).fill_null(0.0)
        d_ask = (
            pl.when(pl.col("r_best_ask") <= prev_ask)
            .then(pl.col("r_ask_sz_1"))
            .otherwise(0.0)
            - pl.when(pl.col("r_best_ask") >= prev_ask).then(prev_asz).otherwise(0.0)
        ).fill_null(0.0)

        step_ofi_l1 = d_bid - d_ask
        step_ofi_top5 = (
            pl.col("r_bid_sz_top5").diff().over("chunk_id").fill_null(0.0)
            - pl.col("r_ask_sz_top5").diff().over("chunk_id").fill_null(0.0)
            + pl.col("v_buyR")
            - pl.col("v_sellR")
        )

        dt_ms = pl.col("dt_cluster_ms").fill_null(0.0).clip(lower_bound=0.0)
        base = df.with_columns(
            step_ofi_l1.alias("_step_ofi_l1"),
            step_ofi_top5.alias("_step_ofi_top5"),
            pl.col("r_best_bid")
            .diff()
            .over("chunk_id")
            .fill_null(0.0)
            .alias("_d_best_bid"),
            pl.col("r_best_ask")
            .diff()
            .over("chunk_id")
            .fill_null(0.0)
            .alias("_d_best_ask"),
            (
                (pl.col("r_bid_sz_1") - pl.col("r_ask_sz_1"))
                / (pl.col("r_bid_sz_1") + pl.col("r_ask_sz_1") + eps)
            ).alias("_obi_l1"),
            (
                (pl.col("r_bid_sz_top3") - pl.col("r_ask_sz_top3"))
                / (pl.col("r_bid_sz_top3") + pl.col("r_ask_sz_top3") + eps)
            ).alias("_obi_l3"),
            (
                (pl.col("r_bid_sz_top5") - pl.col("r_ask_sz_top5"))
                / (pl.col("r_bid_sz_top5") + pl.col("r_ask_sz_top5") + eps)
            ).alias("_obi_l5"),
            (
                (pl.col("r_bid_sz_total") - pl.col("r_ask_sz_total"))
                / (pl.col("r_bid_sz_total") + pl.col("r_ask_sz_total") + eps)
            ).alias("_obi_tot"),
            (pl.col("r_bid_sz_1") / (pl.col("r_bid_sz_top5") + eps)).alias("_bid_conc"),
            (pl.col("r_ask_sz_1") / (pl.col("r_ask_sz_top5") + eps)).alias("_ask_conc"),
            pl.col("spread_prob").fill_null(0.01).alias("_spread"),
            dt_ms.log1p().alias("_log_dt"),
            (dt_ms / 1000.0).clip(upper_bound=3600.0).alias("_dt_sec"),
        )

        # 2. Canonical attacker-perspective features (symmetric across buyR and sellR)
        exprs = []
        for d, sign in (("buyR", 1.0), ("sellR", -1.0)):
            opp = "sellR" if d == "buyR" else "buyR"
            thr = pl.col(f"thr_h{H}_{d}").fill_null(50.0).clip(lower_bound=1.0)
            own_top5 = (
                pl.col("r_bid_sz_top5") if d == "buyR" else pl.col("r_ask_sz_top5")
            )
            opp_top5 = (
                pl.col("r_ask_sz_top5") if d == "buyR" else pl.col("r_bid_sz_top5")
            )
            opp_lvls = (
                pl.col("r_n_ask_levels") if d == "buyR" else pl.col("r_n_bid_levels")
            )

            exprs.extend(
                [
                    (pl.col(f"v_{d}") / thr).alias(f"v_own_{d}"),
                    (pl.col(f"v_{opp}") / thr).alias(f"v_opp_{d}"),
                    pl.col(f"n_{d}").cast(pl.Float32).log1p().alias(f"log_n_own_{d}"),
                    (sign * pl.col("_step_ofi_l1") / thr).alias(f"step_ofi_l1_{d}"),
                    (sign * pl.col("_step_ofi_top5") / thr).alias(f"step_ofi_top5_{d}"),
                    (
                        sign
                        * (
                            pl.col("_d_best_bid")
                            if d == "buyR"
                            else pl.col("_d_best_ask")
                        )
                    ).alias(f"d_best_own_{d}"),
                    (
                        sign
                        * (
                            pl.col("_d_best_ask")
                            if d == "buyR"
                            else pl.col("_d_best_bid")
                        )
                    ).alias(f"d_best_opp_{d}"),
                    (sign * pl.col("_obi_l1")).alias(f"obi_l1_{d}"),
                    (sign * pl.col("_obi_l3")).alias(f"obi_l3_{d}"),
                    (sign * pl.col("_obi_l5")).alias(f"obi_l5_{d}"),
                    (sign * pl.col("_obi_tot")).alias(f"obi_tot_{d}"),
                    (pl.col("_bid_conc") if d == "buyR" else pl.col("_ask_conc")).alias(
                        f"own_conc_{d}"
                    ),
                    (pl.col("_ask_conc") if d == "buyR" else pl.col("_bid_conc")).alias(
                        f"opp_conc_{d}"
                    ),
                    ((opp_top5 + 1.0) / thr).log().alias(f"opp_depth_thr_{d}"),
                    ((own_top5 + 1.0) / thr).log().alias(f"own_depth_thr_{d}"),
                    opp_lvls.cast(pl.Float32).log1p().alias(f"log_opp_lvls_{d}"),
                    pl.col("_spread").alias(f"spread_{d}"),
                    pl.col("_log_dt").alias(f"log_dt_{d}"),
                ]
            )

        return base.with_columns(exprs)

    @staticmethod
    def _feature_cols(direction: str) -> list[str]:
        bases = [
            "v_own",
            "v_opp",
            "log_n_own",
            "step_ofi_l1",
            "step_ofi_top5",
            "d_best_own",
            "d_best_opp",
            "obi_l1",
            "obi_l3",
            "obi_l5",
            "obi_tot",
            "own_conc",
            "opp_conc",
            "opp_depth_thr",
            "own_depth_thr",
            "log_opp_lvls",
            "spread",
            "log_dt",
        ]
        return [f"{b}_{direction}" for b in bases]

    def _raw_pair(self, feat: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        mats = [
            feat.select(self._feature_cols(d)).to_numpy().astype(np.float32)
            for d in DIRS
        ]
        X = np.nan_to_num(
            np.stack(mats, axis=1), nan=0.0, posinf=0.0, neginf=0.0
        )  # (N, 2, F)
        dt = np.nan_to_num(feat["_dt_sec"].to_numpy().astype(np.float32), nan=0.0)
        return X, dt

    def _windows(
        self, feat: pl.DataFrame
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        X, dt = self._raw_pair(feat)
        X = np.clip((X - self.mean) / self.std, -10.0, 10.0)

        pad = self.seq_len - 1
        new_chunk = (feat["chunk_id"] != feat["chunk_id"].shift(1)).fill_null(True)
        rank = new_chunk.cum_sum().to_numpy().astype(np.int64)
        pos = np.arange(len(X)) + rank * pad

        Xp = np.zeros((len(X) + rank[-1] * pad, 2, X.shape[2]), dtype=np.float32)
        dt_p = np.zeros(len(Xp), dtype=np.float32)
        Xp[pos], dt_p[pos] = X, dt
        return torch.from_numpy(Xp), torch.from_numpy(dt_p), pos

    def _targets(self, feat: pl.DataFrame, cfg: TargetConfig, split: str):
        y = np.stack(
            [feat[cfg.target_col(d)].fill_null(0).to_numpy() for d in DIRS], axis=1
        )
        m = np.stack(
            [
                feat.select(self.valid_mask_expr(cfg, d, split)).to_series().to_numpy()
                for d in DIRS
            ],
            axis=1,
        )
        return y.astype(np.float32), m

    @torch.no_grad()
    def _predict_dir(
        self, Xp: torch.Tensor, dt_p: torch.Tensor, pos: np.ndarray, dir_idx: int
    ) -> np.ndarray:
        self.net.eval()
        bs = self.batch_size * 4
        out = []
        for i in range(0, len(pos), bs):
            idx = torch.from_numpy(pos[i : i + bs])[:, None] + self.offsets
            x_b = Xp[idx, dir_idx].to(self.device)
            dt_b = dt_p[idx].to(self.device)
            out.append(torch.sigmoid(self.net(x_b, dt_b)).cpu())
        return torch.cat(out).numpy()

    def fit(
        self, train_df: pl.DataFrame, val_df: pl.DataFrame, cfg: TargetConfig
    ) -> "CTGRUModel":
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        tr = self._build_features(train_df, cfg)
        va = self._build_features(val_df, cfg)
        y_tr, m_tr = self._targets(tr, cfg, "train")
        y_va, m_va = self._targets(va, cfg, "val")

        # Fit shared scaler on valid canonical samples across both directions
        X_tr_raw, _ = self._raw_pair(tr)
        valid_flat = X_tr_raw[m_tr]
        self.mean = valid_flat.mean(axis=0)
        self.std = np.where(
            valid_flat.std(axis=0) > 1e-8, valid_flat.std(axis=0), 1.0
        ).astype(np.float32)

        Xp_tr, dt_tr, pos_tr = self._windows(tr)
        Xp_va, dt_va, pos_va = self._windows(va)

        tr_rows, tr_dirs = np.nonzero(m_tr)
        va_rows, va_dirs = np.nonzero(m_va)
        y_tr_flat = y_tr[tr_rows, tr_dirs]
        y_va_flat = y_va[va_rows, va_dirs]

        self.net = _CTGRUNet(X_tr_raw.shape[2], self.hidden, self.dropout).to(
            self.device
        )
        opt = torch.optim.AdamW(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        best_score, best_state = -1.0, None
        n_samples = len(tr_rows)
        for ep in range(self.epochs):
            self.net.train()
            perm = rng.permutation(n_samples)
            for b in np.array_split(perm, max(1, n_samples // self.batch_size)):
                r_b, d_b = tr_rows[b], tr_dirs[b]
                idx = torch.from_numpy(pos_tr[r_b])[:, None] + self.offsets
                x_b = Xp_tr[idx, torch.from_numpy(d_b)[:, None]].to(self.device)
                dt_b = dt_tr[idx].to(self.device)
                y_b = torch.from_numpy(y_tr_flat[b]).to(self.device)

                logits = self.net(x_b, dt_b)
                loss = F.binary_cross_entropy_with_logits(logits, y_b)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()

            p_buy = self._predict_dir(Xp_va, dt_va, pos_va, 0)
            p_sell = self._predict_dir(Xp_va, dt_va, pos_va, 1)
            p_va_pool = np.stack([p_buy, p_sell], axis=1)[va_rows, va_dirs]
            score = compute_pr_auc(y_va_flat, p_va_pool)
            print(f"    [ct_gru] epoch {ep + 1}/{self.epochs}: val PR-AUC={score:.4f}")
            if score > best_score:
                best_score = score
                best_state = {
                    k: v.detach().clone() for k, v in self.net.state_dict().items()
                }

        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def predict_proba(self, df: pl.DataFrame, cfg: TargetConfig) -> pl.DataFrame:
        feat = self._build_features(df, cfg)
        Xp, dt_p, pos = self._windows(feat)
        return pl.DataFrame(
            {
                "chunk_id": feat["chunk_id"],
                "event_seq": feat["event_seq"],
                "prob_buyR": self._predict_dir(Xp, dt_p, pos, 0),
                "prob_sellR": self._predict_dir(Xp, dt_p, pos, 1),
            }
        )
