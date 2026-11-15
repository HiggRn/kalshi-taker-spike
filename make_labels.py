from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

IN_DIR, OUT_DIR = Path("data/processed"), Path("data/labelled")
PRICE_SCALE, SIZE_SCALE, MS_NS = 1000, 100, 1_000_000

W = 2000  # lookback, in clusters
H_GRID = [50, 100, 200]  # forward horizons, in clusters
LAT_MS = 1  # forward window starts at ts >= t + LAT_MS (>= 1 => strictly later ms)
ALPHA, C = (
    0.05,
    2.5,
)  # quantile rule: 1-ALPHA; sd rule: mu + C*sd (calibrate on train only)
V_MIN = 50.0  # contracts; floor on threshold and "big print" cutoff for the arb mask
N_MIN_TRADES, MAX_SPAN_MIN = 10, 240  # lookback validity
ARB_S = 5.0  # follower window vs. the paired book
SPLIT_FRAC = (0.70, 0.85)  # global chronological train / val / test
MS_OFFSETS = [1_000, 5_000, 30_000]  # fixed-time forward mids for markout

# taker_side -> R-frame direction, per book (R yes == D no)
DIRS = {
    "R": {"yes_taker": "buyR", "no_taker": "sellR"},
    "D": {"no_taker": "buyR", "yes_taker": "sellR"},
}

files = sorted(IN_DIR.glob("*.parquet"))
OUT_DIR.mkdir(exist_ok=True)
lf = pl.scan_parquet(files)

# global split timestamps
t0, t1 = (
    lf.select(
        pl.col("timestamp_ns").min().alias("a"), pl.col("timestamp_ns").max().alias("b")
    )
    .collect()
    .row(0)
)
T_VAL, T_TEST = (t0 + f * (t1 - t0) for f in SPLIT_FRAC)

# big prints per (ticker, R-frame direction), used by the paired-book mask
big = (
    lf.filter(
        (pl.col("event_type") == "trade") & (pl.col("trade_size") >= V_MIN * SIZE_SCALE)
    )
    .select(
        pl.col("ticker").cast(pl.Utf8),
        pl.col("taker_side").cast(pl.Utf8),
        "timestamp_ns",
    )
    .collect()
)
big_ts = {
    (tk, DIRS[tk[-1]][side]): np.sort(g["timestamp_ns"].to_numpy())
    for (tk, side), g in big.group_by(["ticker", "taker_side"])
}

# one chunk per verified segment (continuous book between reconnects)
chunks = [
    g
    for f in files
    for g in pl.read_parquet(
        f,
        columns=[
            "event_seq",
            "timestamp_ns",
            "ticker",
            "event_type",
            "best_bid",
            "best_ask",
            "trade_size",
            "taker_side",
            "segment",
        ],
    )
    .cast({c: pl.Utf8 for c in ("ticker", "event_type", "taker_side", "segment")})
    .filter(pl.col("event_type") != "snapshot")
    .partition_by("segment", maintain_order=True)
]
labelled = defaultdict(list)

for df in chunks:
    tk, seg = df["ticker"][0], df["segment"][0]
    contract, party = tk.rsplit("-", 1)
    pair = f"{contract}-{'D' if party == 'R' else 'R'}"
    side_dir = DIRS[party]

    # one row per ms cluster (decision point = last row of the cluster)
    c = (
        df.with_columns(
            *[
                pl.when(pl.col("taker_side") == s)
                .then(pl.col("trade_size") / SIZE_SCALE)
                .otherwise(0.0)
                .alias(f"v_{d}")
                for s, d in side_dir.items()
            ],
            *[
                (pl.col("taker_side") == s)
                .fill_null(False)
                .cast(pl.Int32)
                .alias(f"n_{d}")
                for s, d in side_dir.items()
            ],
        )
        .group_by("timestamp_ns", maintain_order=True)
        .agg(
            pl.col("event_seq").last(),
            pl.col("v_buyR", "v_sellR", "n_buyR", "n_sellR").sum(),
            pl.col("best_bid", "best_ask").last(),
        )
    )
    ts = c["timestamp_ns"].to_numpy()
    n = len(ts)
    if n < W + max(
        H_GRID
    ):  # reconnect fragment: no row can have lookback + forward window
        print(f"{seg}: skipped ({n} clusters)")
        continue
    mid = (
        (c["best_bid"].cast(pl.Float64) + c["best_ask"].cast(pl.Float64)).to_numpy()
        / 2
        / PRICE_SCALE
    )
    part = (ts >= T_VAL).astype(int) + (ts >= T_TEST)  # 0 train, 1 val, 2 test
    s = np.searchsorted(ts, ts + LAT_MS * MS_NS, side="left")  # forward window start

    span = np.full(n, np.nan)
    if n >= W:
        span[W - 1 :] = (ts[W - 1 :] - ts[: n - W + 1]) / 6e10

    out = {
        "ticker": [tk] * n,
        "contract": [contract] * n,
        "party": [party] * n,
        "chunk_id": [seg] * n,
        "event_seq": c["event_seq"].to_numpy(),
        "cluster_id": np.arange(n),
        "timestamp_ns": ts,
        "split": np.array(["train", "val", "test"])[part],
        "lookback_span_min": span,
        "mid_now": mid,
    }

    # forward window must exist and stay inside the same split (purge)
    fwd_ok, fwd_end = {}, {}
    for H in H_GRID:
        e = s + H - 1
        fwd_end[H] = np.minimum(e, n - 1)
        fwd_ok[H] = (e < n) & (part[fwd_end[H]] == part)
        out[f"mid_fwd_h{H}"] = np.where(fwd_ok[H], mid[fwd_end[H]], np.nan)
    for off in MS_OFFSETS:
        j = np.searchsorted(ts, ts + off * MS_NS, side="right") - 1
        out[f"mid_fwd_ms{off}"] = np.where(ts + off * MS_NS <= ts[-1], mid[j], np.nan)

    for d in side_dir.values():
        v = c[f"v_{d}"]
        cs = np.concatenate([[0.0], np.cumsum(v.to_numpy())])
        lb_n = c[f"n_{d}"].rolling_sum(W).to_numpy()
        out[f"lookback_n_{d}"] = lb_n
        base_ok = (lb_n >= N_MIN_TRADES) & (span <= MAX_SPAN_MIN)

        # follower flag: equivalent leg in the paired book had a big print strictly before t
        pts = np.concatenate(
            [np.zeros(1, np.int64), big_ts.get((pair, d), np.empty(0, np.int64))]
        )
        i = np.searchsorted(pts, ts, side="left") - 1
        age_ms = np.where(i > 0, (ts - pts[i]) / MS_NS, np.nan)
        out[f"paired_big_age_ms_{d}"] = age_ms
        out[f"mask_leader_{d}"] = ~(age_ms <= ARB_S * 1000)

        for H in H_GRID:
            B = v.rolling_sum(H)  # overlapping backward H-sums
            win = W - H + 1
            bwd = B.to_numpy()
            mu = B.rolling_mean(win).to_numpy()
            sd = B.rolling_std(win).to_numpy()
            q = B.rolling_quantile(1 - ALPHA, window_size=win).to_numpy()
            fwd = cs[np.minimum(s + H, n)] - cs[np.minimum(s, n)]
            ok = base_ok & fwd_ok[H]
            thr_q = np.maximum(q, V_MIN)
            thr_sd = np.maximum(mu + C * sd, V_MIN)
            spike = fwd > thr_q
            sfx = f"h{H}_{d}"
            out[f"bwd_vol_{sfx}"] = bwd
            out[f"mu_{sfx}"] = mu
            out[f"sd_{sfx}"] = sd
            out[f"thr_{sfx}"] = thr_q
            out[f"fwd_vol_{sfx}"] = np.where(ok, fwd, np.nan)
            out[f"z_{sfx}"] = np.divide(
                fwd - mu, sd, out=np.full(n, np.nan), where=ok & (sd > 0)
            )
            out[f"y_{sfx}"] = np.where(ok, spike, np.nan)
            out[f"y_sd_{sfx}"] = np.where(ok, fwd > thr_sd, np.nan)
            out[f"y_onset_{sfx}"] = np.where(ok, spike & (bwd < thr_q), np.nan)

    labelled[tk].append(
        pl.DataFrame(out)
        .with_columns(pl.col(pl.Float32, pl.Float64).fill_nan(None))
        .with_columns(pl.col("^y.*$").cast(pl.Int8))
    )
    print(f"{seg}: {n} clusters")

for tk, parts in labelled.items():
    pl.concat(parts).write_parquet(OUT_DIR / f"{tk}.parquet", compression="zstd")
