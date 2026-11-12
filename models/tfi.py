import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from evaluate import compute_pr_auc
from models import BaseSpikeModel, TargetConfig, register_model


@register_model("tfi")
class TFILinearModel(BaseSpikeModel):
    """Microstructure Linear Baseline (L1 Queue Imbalance + Trade Flow Imbalance):

    Tests how much predictive power comes solely from top-of-book L1 OBI and
    directional TFI without deep L2 levels or order flow cancellation dynamics.
    """

    def __init__(self, c_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)):
        super().__init__()
        self.c_grid = c_grid
        self.params: dict[str, dict] = {}

    def _build_features(self, df: pl.DataFrame, cfg: TargetConfig) -> pl.DataFrame:
        H = cfg.horizon
        eps = 1e-6

        # R-frame L1 OBI: positive when R-Bid > R-Ask (upward pressure favoring buyR)
        obi_l1 = (pl.col("r_bid_sz_1") - pl.col("r_ask_sz_1")) / (
            pl.col("r_bid_sz_1") + pl.col("r_ask_sz_1") + eps
        )

        v_buy_50 = (
            pl.col("v_buyR").rolling_sum(window_size=50, min_samples=1).over("chunk_id")
        )
        v_sell_50 = (
            pl.col("v_sellR")
            .rolling_sum(window_size=50, min_samples=1)
            .over("chunk_id")
        )
        tfi_50 = (v_buy_50 - v_sell_50) / (v_buy_50 + v_sell_50 + eps)

        v_buy_H = (
            pl.col("v_buyR").rolling_sum(window_size=H, min_samples=1).over("chunk_id")
        )
        v_sell_H = (
            pl.col("v_sellR").rolling_sum(window_size=H, min_samples=1).over("chunk_id")
        )
        tfi_H = (v_buy_H - v_sell_H) / (v_buy_H + v_sell_H + eps)

        return df.with_columns(
            obi_l1.alias("obi_l1_buyR"),
            (-obi_l1).alias("obi_l1_sellR"),
            tfi_50.alias("tfi_50_buyR"),
            (-tfi_50).alias("tfi_50_sellR"),
            tfi_H.alias("tfi_H_buyR"),
            (-tfi_H).alias("tfi_H_sellR"),
        )

    @staticmethod
    def _feature_cols(direction: str) -> list[str]:
        return [
            f"obi_l1_{direction}",
            f"tfi_50_{direction}",
            f"tfi_H_{direction}",
        ]

    def fit(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        cfg: TargetConfig,
    ) -> "TFILinearModel":
        tr_feat = self._build_features(train_df, cfg)
        va_feat = self._build_features(val_df, cfg)

        for d in ("buyR", "sellR"):
            fcols = self._feature_cols(d)
            y_col = cfg.target_col(d)

            tr_sub = tr_feat.filter(self.valid_mask_expr(cfg, d, split="train"))
            va_sub = va_feat.filter(self.valid_mask_expr(cfg, d, split="val"))

            X_tr = np.nan_to_num(
                tr_sub.select(fcols).to_numpy().astype(np.float64), nan=0.0
            )
            y_tr = tr_sub[y_col].to_numpy().astype(np.int32)

            X_va = np.nan_to_num(
                va_sub.select(fcols).to_numpy().astype(np.float64), nan=0.0
            )
            y_va = va_sub[y_col].to_numpy().astype(np.int32)

            best_score, best_w, best_b, best_c = -1.0, None, 0.0, 1.0
            for c_val in self.c_grid:
                clf = LogisticRegression(
                    l1_ratio=0, C=c_val, solver="lbfgs", max_iter=200
                )
                clf.fit(X_tr, y_tr)
                w, b = clf.coef_.ravel().astype(np.float64), float(clf.intercept_[0])
                val_probs = 1.0 / (1.0 + np.exp(-np.clip(X_va @ w + b, -30.0, 30.0)))
                score = compute_pr_auc(y_va, val_probs)
                if np.isnan(score) or score > best_score:
                    best_score, best_w, best_b, best_c = score, w, b, c_val

            self.params[d] = {"w": best_w, "b": best_b, "C": best_c}

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
                feat_df.select(fcols).to_numpy().astype(np.float64), nan=0.0
            )
            logits = np.clip(X @ p["w"] + p["b"], -30.0, 30.0)
            out_cols[f"prob_{d}"] = 1.0 / (1.0 + np.exp(-logits))

        return pl.DataFrame(out_cols)
