import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from evaluate import compute_pr_auc
from models import BaseSpikeModel, TargetConfig, register_model

DIRS = (("buyR", 1.0), ("sellR", -1.0))
SPANS_TFI = (10, 50, 200, 500)
SPANS_OFI = (10, 50, 200)
LEVELS = ("l1", "top5")
EPS = 1e-6

# per-direction feature columns
COLS = {
    d: [f"hawkes_v_ewma{sp}_{d}" for sp in SPANS_TFI]
    + [
        f"hawkes_opp_ewma50_{d}",
        f"hawkes_n_ewma50_{d}",
        f"bwd_thr_ratio_{d}",
        f"bwd_z_{d}",
        f"lb_trade_density_{d}",
        "_log_dt",
        "clock_accel",
    ]
    + [
        f"ofi_{k}_{lvl}_ewma{sp}_{d}"
        for lvl in LEVELS
        for k in ("P", "N")
        for sp in SPANS_OFI
    ]
    + [f"ofi_ratio_{lvl}_{d}" for lvl in LEVELS]
    for d, _ in DIRS
}


@register_model("ofi_hawkes")
class OFIHawkesModel(BaseSpikeModel):
    """OFI-Hawkes = TFI-Hawkes + limit-only OFI excitation streams.

    One L2 logistic regression per direction on 25 event-time features.
    """

    def __init__(self, c_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)):
        super().__init__()
        self.c_grid = c_grid
        self.params: dict[str, dict] = {}

    def _build_features(self, df: pl.DataFrame, cfg: TargetConfig) -> pl.DataFrame:
        H = cfg.horizon

        # ---- stage 1: limit-only OFI atoms (R-frame) + slow same-level depth norms
        prev_bid_px = pl.col("r_best_bid").shift(1).over("chunk_id")
        prev_ask_px = pl.col("r_best_ask").shift(1).over("chunk_id")
        prev_bsz = pl.col("r_bid_sz_1").shift(1).over("chunk_id")
        prev_asz = pl.col("r_ask_sz_1").shift(1).over("chunk_id")

        # Cont-Kukanov-Stoikov L1 flow
        d_bid = (
            pl.when(pl.col("r_best_bid") >= prev_bid_px)
            .then(pl.col("r_bid_sz_1"))
            .otherwise(0.0)
            - pl.when(pl.col("r_best_bid") <= prev_bid_px).then(prev_bsz).otherwise(0.0)
        ).fill_null(0.0)
        d_ask = (
            pl.when(pl.col("r_best_ask") <= prev_ask_px)
            .then(pl.col("r_ask_sz_1"))
            .otherwise(0.0)
            - pl.when(pl.col("r_best_ask") >= prev_ask_px).then(prev_asz).otherwise(0.0)
        ).fill_null(0.0)
        # trade-driven depletion capped by the previous L1 queue (buyR eats r_ask, sellR eats r_bid)
        tr_buy = (
            pl.when(pl.col("v_buyR") < prev_asz)
            .then(pl.col("v_buyR"))
            .otherwise(prev_asz)
            .fill_null(0.0)
        )
        tr_sell = (
            pl.when(pl.col("v_sellR") < prev_bsz)
            .then(pl.col("v_sellR"))
            .otherwise(prev_bsz)
            .fill_null(0.0)
        )

        out = df.with_columns(
            pl.col("dt_cluster_ms").clip(lower_bound=0.0).log1p().alias("_log_dt"),
            (d_bid - d_ask - tr_buy + tr_sell).alias("_lo_l1"),
            (
                pl.col("r_bid_sz_top5").diff().over("chunk_id").fill_null(0.0)
                - pl.col("r_ask_sz_top5").diff().over("chunk_id").fill_null(0.0)
                - pl.col("v_buyR")
                + pl.col("v_sellR")
            ).alias("_lo_top5"),
            (
                (pl.col("r_bid_sz_1") + pl.col("r_ask_sz_1"))
                .ewm_mean(span=500, adjust=False)
                .over("chunk_id")
                + 1.0
            ).alias("_dep_l1"),
            (
                (pl.col("r_bid_sz_top5") + pl.col("r_ask_sz_top5"))
                .ewm_mean(span=500, adjust=False)
                .over("chunk_id")
                + 1.0
            ).alias("_dep_top5"),
        )

        # ---- stage 2: Block A (trade excitation) + Block B (OFI excitation EWMAs)
        exprs = [
            (
                pl.col("_log_dt").ewm_mean(span=20, adjust=False).over("chunk_id")
                / (
                    pl.col("_log_dt").ewm_mean(span=500, adjust=False).over("chunk_id")
                    + EPS
                )
            ).alias("clock_accel")
        ]
        for d, sign in DIRS:
            opp = "sellR" if d == "buyR" else "buyR"
            sfx = f"h{H}_{d}"
            thr = pl.col(f"thr_{sfx}").fill_null(50.0).clip(lower_bound=1.0)
            mu = pl.col(f"mu_{sfx}").fill_null(0.0)
            sd = pl.col(f"sd_{sfx}").fill_null(1.0).clip(lower_bound=1.0)

            # Block A
            exprs += [
                (
                    pl.col(f"v_{d}").ewm_mean(span=sp, adjust=False).over("chunk_id")
                    * H
                    / thr
                ).alias(f"hawkes_v_ewma{sp}_{d}")
                for sp in SPANS_TFI
            ]
            exprs += [
                (
                    pl.col(f"v_{opp}").ewm_mean(span=50, adjust=False).over("chunk_id")
                    * H
                    / thr
                ).alias(f"hawkes_opp_ewma50_{d}"),
                pl.col(f"n_{d}")
                .cast(pl.Float64)
                .ewm_mean(span=50, adjust=False)
                .over("chunk_id")
                .alias(f"hawkes_n_ewma50_{d}"),
                (pl.col(f"bwd_vol_{sfx}") / thr).alias(f"bwd_thr_ratio_{d}"),
                ((pl.col(f"bwd_vol_{sfx}") - mu) / sd).alias(f"bwd_z_{d}"),
                (pl.col(f"lookback_n_{d}").cast(pl.Float64) / 2000.0).alias(
                    f"lb_trade_density_{d}"
                ),
            ]

            # Block B: supportive (P) / opposing (N) pressure, depth-normalised, asinh, EWMA
            for lvl in LEVELS:
                for kind, sg in (("P", sign), ("N", -sign)):
                    x = (sg * pl.col(f"_lo_{lvl}")).clip(lower_bound=0.0) / pl.col(
                        f"_dep_{lvl}"
                    )
                    mark = (x + (x * x + 1.0).sqrt()).log()  # asinh(x)
                    exprs += [
                        mark.ewm_mean(span=sp, adjust=False)
                        .over("chunk_id")
                        .alias(f"ofi_{kind}_{lvl}_ewma{sp}_{d}")
                        for sp in SPANS_OFI
                    ]
        out = out.with_columns(exprs)

        # ---- stage 3: P / (P + N) ratios from the span-50 streams
        return out.with_columns(
            [
                (
                    pl.col(f"ofi_P_{lvl}_ewma50_{d}")
                    / (
                        pl.col(f"ofi_P_{lvl}_ewma50_{d}")
                        + pl.col(f"ofi_N_{lvl}_ewma50_{d}")
                        + EPS
                    )
                ).alias(f"ofi_ratio_{lvl}_{d}")
                for d, _ in DIRS
                for lvl in LEVELS
            ]
        )

    def fit(
        self, train_df: pl.DataFrame, val_df: pl.DataFrame, cfg: TargetConfig
    ) -> "OFIHawkesModel":
        tr_feat = self._build_features(train_df, cfg)
        va_feat = self._build_features(val_df, cfg)

        for d, _ in DIRS:
            cols, y_col = COLS[d], cfg.target_col(d)
            tr_sub = tr_feat.filter(self.valid_mask_expr(cfg, d, split="train"))
            va_sub = va_feat.filter(self.valid_mask_expr(cfg, d, split="val"))

            X_tr = np.nan_to_num(
                tr_sub.select(cols).to_numpy().astype(np.float64),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            y_tr = tr_sub[y_col].to_numpy().astype(np.int32)
            mean = np.mean(X_tr, axis=0)
            std = np.where(np.std(X_tr, axis=0) > 1e-8, np.std(X_tr, axis=0), 1.0)
            X_tr = np.clip((X_tr - mean) / std, -10.0, 10.0)

            X_va = np.nan_to_num(
                va_sub.select(cols).to_numpy().astype(np.float64),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            y_va = va_sub[y_col].to_numpy().astype(np.int32)
            X_va = np.clip((X_va - mean) / std, -10.0, 10.0)

            best_score, best = -1.0, None
            for c_val in self.c_grid:
                clf = LogisticRegression(
                    l1_ratio=0, C=c_val, solver="lbfgs", max_iter=200
                )
                clf.fit(X_tr, y_tr)
                w, b = clf.coef_.ravel().astype(np.float64), float(clf.intercept_[0])
                val_probs = 1.0 / (1.0 + np.exp(-np.clip(X_va @ w + b, -30.0, 30.0)))
                score = compute_pr_auc(y_va, val_probs)
                if np.isnan(score) or score > best_score:
                    best_score = score
                    best = {"w": w, "b": b, "mean": mean, "std": std, "C": c_val}
            self.params[d] = best
            print(
                f"    [ofi_hawkes | {d}] n_feat={len(cols)} best val PR-AUC={best_score:.4f} (C={best['C']})"
            )
        return self

    def predict_proba(self, df: pl.DataFrame, cfg: TargetConfig) -> pl.DataFrame:
        feat = self._build_features(df, cfg)
        out_cols = {"chunk_id": feat["chunk_id"], "event_seq": feat["event_seq"]}
        for d, _ in DIRS:
            p = self.params[d]
            X = np.nan_to_num(
                feat.select(COLS[d]).to_numpy().astype(np.float64),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            X = np.clip((X - p["mean"]) / p["std"], -10.0, 10.0)
            out_cols[f"prob_{d}"] = 1.0 / (
                1.0 + np.exp(-np.clip(X @ p["w"] + p["b"], -30.0, 30.0))
            )
        return pl.DataFrame(out_cols)
