import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from evaluate import compute_pr_auc
from models import BaseSpikeModel, TargetConfig, register_model


@register_model("ofi")
class OFILinearGLM(BaseSpikeModel):
    """Linear Family: L1/L2-Penalized Microstructure Logistic GLM.

    Combines multi-level OBI, multi-scale Cont-Kukanov-Stoikov OFI (L1 & Top-5),
    multi-horizon TFI, relative spread, book slope, microprice deviation,
    log-odds price, and log inter-event duration.
    """

    def __init__(
        self,
        c_grid: tuple[float, ...] = (0.01, 0.1, 1.0),
        penalties: tuple[str, ...] = ("l2", "l1"),
    ):
        super().__init__()
        self.c_grid = c_grid
        self.penalties = penalties
        self.params: dict[str, dict] = {}

    def _build_features(self, df: pl.DataFrame, cfg: TargetConfig) -> pl.DataFrame:
        H = cfg.horizon
        eps = 1e-6

        # 1. Step-level Cont-Kukanov-Stoikov L1 OFI & Trade-Restored Top-5 OFI inside each chunk
        prev_bid_px = pl.col("r_best_bid").shift(1).over("chunk_id")
        prev_ask_px = pl.col("r_best_ask").shift(1).over("chunk_id")
        prev_bid_sz1 = pl.col("r_bid_sz_1").shift(1).over("chunk_id")
        prev_ask_sz1 = pl.col("r_ask_sz_1").shift(1).over("chunk_id")

        delta_bid_l1 = (
            pl.when(pl.col("r_best_bid") >= prev_bid_px)
            .then(pl.col("r_bid_sz_1"))
            .otherwise(0.0)
            - pl.when(pl.col("r_best_bid") <= prev_bid_px)
            .then(prev_bid_sz1)
            .otherwise(0.0)
        ).fill_null(0.0)

        delta_ask_l1 = (
            pl.when(pl.col("r_best_ask") <= prev_ask_px)
            .then(pl.col("r_ask_sz_1"))
            .otherwise(0.0)
            - pl.when(pl.col("r_best_ask") >= prev_ask_px)
            .then(prev_ask_sz1)
            .otherwise(0.0)
        ).fill_null(0.0)

        step_ofi_l1 = delta_bid_l1 - delta_ask_l1
        step_ofi_top5 = (
            pl.col("r_bid_sz_top5").diff().over("chunk_id").fill_null(0.0)
            - pl.col("r_ask_sz_top5").diff().over("chunk_id").fill_null(0.0)
            + pl.col("v_buyR")
            - pl.col("v_sellR")
        )

        # 2. Multi-level OBI & Book Geometry in R-frame
        obi_l1 = (pl.col("r_bid_sz_1") - pl.col("r_ask_sz_1")) / (
            pl.col("r_bid_sz_1") + pl.col("r_ask_sz_1") + eps
        )
        obi_l3 = (pl.col("r_bid_sz_top3") - pl.col("r_ask_sz_top3")) / (
            pl.col("r_bid_sz_top3") + pl.col("r_ask_sz_top3") + eps
        )
        obi_l5 = (pl.col("r_bid_sz_top5") - pl.col("r_ask_sz_top5")) / (
            pl.col("r_bid_sz_top5") + pl.col("r_ask_sz_top5") + eps
        )
        obi_tot = (pl.col("r_bid_sz_total") - pl.col("r_ask_sz_total")) / (
            pl.col("r_bid_sz_total") + pl.col("r_ask_sz_total") + eps
        )

        # Microprice deviation relative to half-spread: turns equal to obi_l1 in normalized units,
        # and weighted by spread_prob in probability space
        micro_dev = obi_l1 * (0.5 * pl.col("spread_prob").fill_null(0.01))

        # Log-odds transformed mid price: stretches the 0.90+ boundary regime on SENATEGA-26
        r_mid_clip = pl.col("r_mid").fill_null(0.5).clip(0.001, 0.999)
        log_odds_mid = (r_mid_clip / (1.0 - r_mid_clip)).log()

        # Book slope / L1-to-Top5 concentration ratios
        bid_conc = pl.col("r_bid_sz_1") / (pl.col("r_bid_sz_top5") + eps)
        ask_conc = pl.col("r_ask_sz_1") / (pl.col("r_ask_sz_top5") + eps)

        out = df.with_columns(
            step_ofi_l1.alias("_step_ofi_l1"),
            step_ofi_top5.alias("_step_ofi_top5"),
            obi_l1.alias("_obi_l1"),
            obi_l3.alias("_obi_l3"),
            obi_l5.alias("_obi_l5"),
            obi_tot.alias("_obi_tot"),
            micro_dev.alias("_micro_dev"),
            log_odds_mid.alias("_log_odds_mid"),
            bid_conc.alias("_bid_conc"),
            ask_conc.alias("_ask_conc"),
            (pl.col("r_bid_sz_1") + pl.col("r_ask_sz_1") + 1.0).alias("_l1_depth"),
            (pl.col("r_bid_sz_top5") + pl.col("r_ask_sz_top5") + 1.0).alias(
                "_l5_depth"
            ),
            pl.col("dt_cluster_ms").clip(lower_bound=0.0).log1p().alias("log_dt"),
        )

        # 3. Rolling multi-horizon OFI, TFI, and Clock Acceleration over chunk_id
        roll_exprs = [
            (
                pl.col("log_dt").ewm_mean(span=20, adjust=False).over("chunk_id")
                / (
                    pl.col("log_dt").ewm_mean(span=500, adjust=False).over("chunk_id")
                    + eps
                )
            ).alias("clock_accel")
        ]

        for w in (20, 100, 500):
            ofi_l1_w = pl.col("_step_ofi_l1").rolling_sum(
                window_size=w, min_samples=1
            ).over("chunk_id") / pl.col("_l1_depth").rolling_mean(
                window_size=w, min_samples=1
            ).over("chunk_id")
            ofi_l5_w = pl.col("_step_ofi_top5").rolling_sum(
                window_size=w, min_samples=1
            ).over("chunk_id") / pl.col("_l5_depth").rolling_mean(
                window_size=w, min_samples=1
            ).over("chunk_id")
            vb_w = (
                pl.col("v_buyR")
                .rolling_sum(window_size=w, min_samples=1)
                .over("chunk_id")
            )
            vs_w = (
                pl.col("v_sellR")
                .rolling_sum(window_size=w, min_samples=1)
                .over("chunk_id")
            )
            tfi_w = (vb_w - vs_w) / (vb_w + vs_w + eps)

            roll_exprs.extend(
                [
                    ofi_l1_w.alias(f"_ofi_l1_w{w}"),
                    ofi_l5_w.alias(f"_ofi_l5_w{w}"),
                    tfi_w.alias(f"_tfi_w{w}"),
                ]
            )

        out = out.with_columns(roll_exprs)

        # 4. Orient directional features symmetrically for buyR (sign = +1) and sellR (sign = -1)
        dir_exprs = []
        for d, sign in (("buyR", 1.0), ("sellR", -1.0)):
            sfx = f"h{H}_{d}"
            thr_col = pl.col(f"thr_{sfx}").fill_null(50.0).clip(lower_bound=1.0)
            opp_top5 = (
                pl.col("r_ask_sz_top5") if d == "buyR" else pl.col("r_bid_sz_top5")
            )
            own_conc = pl.col("_bid_conc") if d == "buyR" else pl.col("_ask_conc")
            opp_conc = pl.col("_ask_conc") if d == "buyR" else pl.col("_bid_conc")

            dir_exprs.extend(
                [
                    (sign * pl.col("_obi_l1")).alias(f"obi_l1_{d}"),
                    (sign * pl.col("_obi_l3")).alias(f"obi_l3_{d}"),
                    (sign * pl.col("_obi_l5")).alias(f"obi_l5_{d}"),
                    (sign * pl.col("_obi_tot")).alias(f"obi_tot_{d}"),
                    (sign * pl.col("_micro_dev")).alias(f"micro_dev_{d}"),
                    (sign * pl.col("_log_odds_mid")).alias(f"log_odds_mid_{d}"),
                    (pl.col("_log_odds_mid").abs()).alias(f"abs_log_odds_mid_{d}"),
                    own_conc.alias(f"own_conc_{d}"),
                    opp_conc.alias(f"opp_conc_{d}"),
                    ((opp_top5 + 1.0) / thr_col).log().alias(f"opp_depth_to_thr_{d}"),
                    (pl.col(f"bwd_vol_{sfx}") / thr_col).alias(f"bwd_thr_ratio_{d}"),
                ]
            )
            for w in (20, 100, 500):
                dir_exprs.extend(
                    [
                        (sign * pl.col(f"_ofi_l1_w{w}")).alias(f"ofi_l1_w{w}_{d}"),
                        (sign * pl.col(f"_ofi_l5_w{w}")).alias(f"ofi_l5_w{w}_{d}"),
                        (sign * pl.col(f"_tfi_w{w}")).alias(f"tfi_w{w}_{d}"),
                    ]
                )

        return out.with_columns(dir_exprs)

    @staticmethod
    def _feature_cols(direction: str) -> list[str]:
        cols = [
            f"obi_l1_{direction}",
            f"obi_l3_{direction}",
            f"obi_l5_{direction}",
            f"obi_tot_{direction}",
            f"micro_dev_{direction}",
            f"log_odds_mid_{direction}",
            f"abs_log_odds_mid_{direction}",
            f"own_conc_{direction}",
            f"opp_conc_{direction}",
            f"opp_depth_to_thr_{direction}",
            f"bwd_thr_ratio_{direction}",
            "spread_prob",
            "log_dt",
            "clock_accel",
        ]
        for w in (20, 100, 500):
            cols.extend(
                [
                    f"ofi_l1_w{w}_{direction}",
                    f"ofi_l5_w{w}_{direction}",
                    f"tfi_w{w}_{direction}",
                ]
            )
        return cols

    def fit(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        cfg: TargetConfig,
    ) -> "OFILinearGLM":
        tr_feat = self._build_features(train_df, cfg)
        va_feat = self._build_features(val_df, cfg)

        for d in ("buyR", "sellR"):
            fcols = self._feature_cols(d)
            y_col = cfg.target_col(d)

            tr_sub = tr_feat.filter(self.valid_mask_expr(cfg, d, split="train"))
            va_sub = va_feat.filter(self.valid_mask_expr(cfg, d, split="val"))

            X_tr = np.nan_to_num(
                tr_sub.select(fcols).to_numpy().astype(np.float64),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            y_tr = tr_sub[y_col].to_numpy().astype(np.int32)

            mean = np.mean(X_tr, axis=0)
            std = np.where(np.std(X_tr, axis=0) > 1e-8, np.std(X_tr, axis=0), 1.0)
            X_tr_norm = np.clip((X_tr - mean) / std, -10.0, 10.0)

            X_va = np.nan_to_num(
                va_sub.select(fcols).to_numpy().astype(np.float64),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            y_va = va_sub[y_col].to_numpy().astype(np.int32)
            X_va_norm = np.clip((X_va - mean) / std, -10.0, 10.0)

            best_score = -1.0
            best_cfg = None
            for pen in self.penalties:
                for c_val in self.c_grid:
                    solver = "saga" if pen == "l1" else "lbfgs"
                    l1_ratio = 1 if pen == "l1" else 0
                    clf = LogisticRegression(
                        l1_ratio=l1_ratio, C=c_val, solver=solver, max_iter=200
                    )
                    clf.fit(X_tr_norm, y_tr)
                    w, b = (
                        clf.coef_.ravel().astype(np.float64),
                        float(clf.intercept_[0]),
                    )
                    val_probs = 1.0 / (
                        1.0 + np.exp(-np.clip(X_va_norm @ w + b, -30.0, 30.0))
                    )
                    score = compute_pr_auc(y_va, val_probs)
                    if np.isnan(score) or score > best_score:
                        best_score = score
                        best_cfg = {
                            "w": w,
                            "b": b,
                            "mean": mean,
                            "std": std,
                            "penalty": pen,
                            "C": c_val,
                        }
            if best_cfg is not None:
                self.params[d] = best_cfg
                print(
                    f"    [ofi | {d}] Best Val PR-AUC={best_score:.4f} "
                    f"(penalty={best_cfg['penalty']}, C={best_cfg['C']})"
                )

        return self

    def predict_proba(self, df: pl.DataFrame, cfg: TargetConfig) -> pl.DataFrame:
        feat_df = self._build_features(df, cfg)
        out_cols = {
            "chunk_id": feat_df["chunk_id"],
            "event_seq": feat_df["event_seq"],
        }

        for d in ("buyR", "sellR"):
            fcols = self._feature_cols(d)
            p = self.params[d]
            X = np.nan_to_num(
                feat_df.select(fcols).to_numpy().astype(np.float64),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            X_norm = np.clip((X - p["mean"]) / p["std"], -10.0, 10.0)
            logits = np.clip(X_norm @ p["w"] + p["b"], -30.0, 30.0)
            out_cols[f"prob_{d}"] = 1.0 / (1.0 + np.exp(-logits))

        return pl.DataFrame(out_cols)
