import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from evaluate import compute_pr_auc
from models import BaseSpikeModel, TargetConfig, register_model


@register_model("tfi_hawkes")
class TFIHawkesModel(BaseSpikeModel):
    """Trade-Only Self-Exciting Baseline

    Uses multi-scale EWMAs of directional taker volumes, backward H-block
    realized ratios, and physical inter-event waiting time acceleration (zero L2
    order book information).
    """

    def __init__(self, c_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)):
        super().__init__()
        self.c_grid = c_grid
        self.params: dict[str, dict] = {}

    def _build_features(self, df: pl.DataFrame, cfg: TargetConfig) -> pl.DataFrame:
        H = cfg.horizon
        eps = 1e-6

        # Event-clock waiting time features strictly within each continuous chunk
        out = df.with_columns(
            pl.col("dt_cluster_ms").clip(lower_bound=0.0).log1p().alias("_log_dt")
        ).with_columns(
            pl.col("_log_dt")
            .ewm_mean(span=20, adjust=False)
            .over("chunk_id")
            .alias("_dt_ewma_fast"),
            pl.col("_log_dt")
            .ewm_mean(span=500, adjust=False)
            .over("chunk_id")
            .alias("_dt_ewma_slow"),
        )

        exprs = [
            (pl.col("_dt_ewma_fast") / (pl.col("_dt_ewma_slow") + eps)).alias(
                "clock_accel"
            )
        ]

        # Directional Hawkes / EWMA self-exciting intensities for buyR and sellR
        for d in ("buyR", "sellR"):
            opp = "sellR" if d == "buyR" else "buyR"
            sfx = f"h{H}_{d}"
            thr_col = pl.col(f"thr_{sfx}").fill_null(50.0).clip(lower_bound=1.0)
            mu_col = pl.col(f"mu_{sfx}").fill_null(0.0)
            sd_col = pl.col(f"sd_{sfx}").fill_null(1.0).clip(lower_bound=1.0)

            for span in (10, 50, 200, 500):
                ewma_v = (
                    pl.col(f"v_{d}").ewm_mean(span=span, adjust=False).over("chunk_id")
                )
                # Scale by H and normalize by trailing threshold thr_{sfx} for regime invariance
                exprs.append(((ewma_v * H) / thr_col).alias(f"hawkes_v_ewma{span}_{d}"))

            # Opposing flow excitation & trade count arrival intensity
            ewma_opp = (
                pl.col(f"v_{opp}").ewm_mean(span=50, adjust=False).over("chunk_id")
            )
            ewma_n = (
                pl.col(f"n_{d}")
                .cast(pl.Float64)
                .ewm_mean(span=50, adjust=False)
                .over("chunk_id")
            )

            exprs.extend(
                [
                    ((ewma_opp * H) / thr_col).alias(f"hawkes_opp_ewma50_{d}"),
                    ewma_n.alias(f"hawkes_n_ewma50_{d}"),
                    (pl.col(f"bwd_vol_{sfx}") / thr_col).alias(f"bwd_thr_ratio_{d}"),
                    ((pl.col(f"bwd_vol_{sfx}") - mu_col) / sd_col).alias(f"bwd_z_{d}"),
                    (pl.col(f"lookback_n_{d}").cast(pl.Float64) / 2000.0).alias(
                        f"lb_trade_density_{d}"
                    ),
                ]
            )

        return out.with_columns(exprs)

    @staticmethod
    def _feature_cols(direction: str) -> list[str]:
        return [
            f"hawkes_v_ewma10_{direction}",
            f"hawkes_v_ewma50_{direction}",
            f"hawkes_v_ewma200_{direction}",
            f"hawkes_v_ewma500_{direction}",
            f"hawkes_opp_ewma50_{direction}",
            f"hawkes_n_ewma50_{direction}",
            f"bwd_thr_ratio_{direction}",
            f"bwd_z_{direction}",
            f"lb_trade_density_{direction}",
            "_log_dt",
            "clock_accel",
        ]

    def fit(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        cfg: TargetConfig,
    ) -> "HawkesModel":
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

            best_score, best_w, best_b, best_c = -1.0, None, 0.0, 1.0
            for c_val in self.c_grid:
                clf = LogisticRegression(
                    l1_ratio=0, C=c_val, solver="lbfgs", max_iter=200
                )
                clf.fit(X_tr_norm, y_tr)
                w, b = clf.coef_.ravel().astype(np.float64), float(clf.intercept_[0])
                val_probs = 1.0 / (
                    1.0 + np.exp(-np.clip(X_va_norm @ w + b, -30.0, 30.0))
                )
                score = compute_pr_auc(y_va, val_probs)
                if np.isnan(score) or score > best_score:
                    best_score, best_w, best_b, best_c = score, w, b, c_val

            self.params[d] = {
                "w": best_w,
                "b": best_b,
                "mean": mean,
                "std": std,
                "C": best_c,
            }

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
