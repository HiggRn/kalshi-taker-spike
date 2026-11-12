import itertools

import numpy as np
import polars as pl

from models import TargetConfig


def compute_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Exact Average Precision (area under PR curve) without external sklearn dependency."""
    if len(y_true) == 0 or np.sum(y_true) == 0:
        return float("nan")
    order = np.argsort(-y_prob, kind="mergesort")
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted)
    ranks = np.arange(1, len(y_sorted) + 1, dtype=np.float64)
    precisions = tp / ranks
    return float(np.sum(precisions[y_sorted == 1]) / np.sum(y_true))


def compute_f1_at_topk(
    y_true: np.ndarray, y_prob: np.ndarray, k_pct: float
) -> tuple[float, float, float]:
    """Return (F1, Precision, Recall) when alerting on the top k_pct% highest probabilities."""
    n = len(y_true)
    if n == 0 or np.sum(y_true) == 0:
        return float("nan"), float("nan"), float("nan")
    k_count = max(1, round(n * (k_pct / 100.0)))
    idx = np.argpartition(-y_prob, k_count - 1)[:k_count]
    tp = float(np.sum(y_true[idx] == 1))
    prec = tp / k_count
    rec = tp / float(np.sum(y_true == 1))
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return f1, prec, rec


def compute_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean((y_prob - y_true) ** 2))


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error across `n_bins` uniform probability intervals."""
    if len(y_true) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for b0, b1 in itertools.pairwise(bins):
        mask = (
            (y_prob >= b0) & (y_prob <= b1)
            if b1 == 1.0
            else (y_prob >= b0) & (y_prob < b1)
        )
        cnt = np.sum(mask)
        if cnt > 0:
            avg_prob = np.mean(y_prob[mask])
            avg_true = np.mean(y_true[mask])
            ece += (cnt / n) * abs(avg_prob - avg_true)
    return float(ece)


def compute_markout_avoidance_bps(
    y_prob: np.ndarray,
    r_mid_now: np.ndarray,
    r_mid_fwd: np.ndarray,
    spread_prob: np.ndarray,
    direction: str,
    alert_pct: float = 2.0,
) -> float:
    """Simulated passive L1 Market Maker Adverse Selection Avoidance (in bps of $1 par):

    - When a `buyR` spike occurs, R-frame mid price tends to jump up (r_mid_fwd
    > r_mid_now),
      inflicting an adverse post-trade markout on a resting R-Ask quote.
    - When a `sellR` spike occurs, R-frame mid price tends to drop (r_mid_fwd <
    r_mid_now),
      inflicting an adverse post-trade markout on a resting R-Bid quote.
    - Canceling/widening on the top `alert_pct`% predicted spike probabilities
    avoids the
      adverse directional markout move, net of the forfeited half-spread
      opportunity cost.
    """
    valid = ~np.isnan(r_mid_now) & ~np.isnan(r_mid_fwd) & ~np.isnan(spread_prob)
    if np.sum(valid) < 10:
        return float("nan")

    prob = y_prob[valid]
    mid_now = r_mid_now[valid]
    mid_fwd = r_mid_fwd[valid]

    # Directional adverse move against passive L1 quote
    if direction == "buyR":
        adverse_move = mid_fwd - mid_now  # Ask gets run over when R mid rises
    else:
        adverse_move = mid_now - mid_fwd  # Bid gets run over when R mid falls

    # Passive L1 fill loss when hit = adverse_move - half_spread
    # Avoiding the quote on top alert_pct% saves (adverse_move - half_spread)
    k_count = max(1, round(len(prob) * (alert_pct / 100.0)))
    alert_idx = np.argpartition(-prob, k_count - 1)[:k_count]

    # Average adverse selection avoided per alerted event vs baseline unconditional markout, in bps (1e4)
    net_avoided_per_alert = np.mean(adverse_move[alert_idx] - np.mean(adverse_move))
    return float(net_avoided_per_alert * 10_000.0)


def evaluate_predictions(
    contract_df: pl.DataFrame,
    pred_df: pl.DataFrame,
    cfg: TargetConfig,
    split: str = "test",
) -> dict[str, float]:
    """Evaluate a model's predicted probabilities on a single contract's `split`."""
    joined = contract_df.join(
        pred_df.select(["chunk_id", "event_seq", "prob_buyR", "prob_sellR"]),
        on=["chunk_id", "event_seq"],
        how="inner",
    ).filter(pl.col("split") == split)

    # Convert mid_now and mid_fwd to R-frame (on D book, R_mid = 1.0 - D_mid)
    fwd_col = (
        cfg.markout_col
        if cfg.markout_col in joined.columns
        else f"mid_fwd_h{cfg.horizon}"
    )
    joined = joined.with_columns(
        pl.when(pl.col("party") == "R")
        .then(pl.col("mid_now"))
        .otherwise(1.0 - pl.col("mid_now"))
        .alias("_r_mid_now"),
        pl.when(pl.col("party") == "R")
        .then(pl.col(fwd_col))
        .otherwise(1.0 - pl.col(fwd_col))
        .alias("_r_mid_fwd"),
    )

    all_y, all_p = [], []
    markouts_bps = []
    dir_metrics = {}

    for d in ("buyR", "sellR"):
        y_col = cfg.target_col(d)
        p_col = f"prob_{d}"
        cond = pl.col(y_col).is_not_null() & pl.col(p_col).is_not_null()
        if cfg.leader_only:
            cond = cond & pl.col(cfg.mask_col(d)).fill_null(False)

        sub = joined.filter(cond)
        if sub.is_empty():
            continue

        y_arr = sub[y_col].to_numpy().astype(np.int32)
        p_arr = np.clip(sub[p_col].to_numpy().astype(np.float64), 0.0, 1.0)

        all_y.append(y_arr)
        all_p.append(p_arr)

        dir_metrics[f"pr_auc_{d}"] = compute_pr_auc(y_arr, p_arr)
        dir_metrics[f"base_rate_{d}"] = float(np.mean(y_arr))

        mk_bps = compute_markout_avoidance_bps(
            p_arr,
            sub["_r_mid_now"].to_numpy(),
            sub["_r_mid_fwd"].to_numpy(),
            sub["spread_prob"].to_numpy(),
            direction=d,
            alert_pct=2.0,
        )
        if not np.isnan(mk_bps):
            markouts_bps.append(mk_bps)

    if not all_y:
        return {}

    y_pool = np.concatenate(all_y)
    p_pool = np.concatenate(all_p)

    base_rate = float(np.mean(y_pool))
    pr_auc = compute_pr_auc(y_pool, p_pool)

    res = {
        "n_eval": len(y_pool),
        "base_rate (%)": round(base_rate * 100.0, 3),
        "PR-AUC": round(pr_auc, 4),
        "PR-Lift (x)": (
            round(pr_auc / base_rate, 2) if base_rate > 0 else float("nan")
        ),
        "Brier": round(compute_brier(y_pool, p_pool), 5),
        "ECE": round(compute_ece(y_pool, p_pool), 5),
        "Markout@2% (bps)": (
            round(float(np.mean(markouts_bps)), 2) if markouts_bps else float("nan")
        ),
    }

    for k in cfg.top_k_pcts:
        f1, prec, _rec = compute_f1_at_topk(y_pool, p_pool, k)
        res[f"F1@{k:g}%"] = round(f1, 4)
        res[f"Prec@{k:g}%"] = round(prec, 4)

    for k, v in dir_metrics.items():
        res[k] = round(v, 4)

    return res


def format_comparison_table(rows: list[dict[str, object]], cfg: TargetConfig) -> str:
    """Format a clean console comparison table across models and contracts."""
    if not rows:
        return "No evaluation results produced."
    df = pl.DataFrame(rows)
    display_cols = [
        "Model",
        "Contract",
        "Regime",
        "n_eval",
        "base_rate (%)",
        "PR-AUC",
        "PR-Lift (x)",
        f"F1@{cfg.top_k_pcts[0]:g}%",
        f"F1@{cfg.top_k_pcts[1]:g}%",
        "Brier",
        "ECE",
        "Markout@2% (bps)",
    ]
    cols = [c for c in display_cols if c in df.columns]
    with pl.Config(
        tbl_rows=100,
        tbl_cols=20,
        tbl_width_chars=160,
        fmt_str_lengths=30,
    ):
        return str(df.select(cols))
