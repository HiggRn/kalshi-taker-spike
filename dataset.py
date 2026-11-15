from pathlib import Path

import polars as pl

DATA_DIR = Path("data/processed")
LABEL_DIR = Path("data/labelled")
PRICE_SCALE = 1000.0
SIZE_SCALE = 100.0

PRIMARY_CONTRACT = "CONTROLS-2026"
TRANSFER_CONTRACTS = ["SENATETX-26", "SENATEGA-26"]
ALL_CONTRACTS = [PRIMARY_CONTRACT] + TRANSFER_CONTRACTS

# taker_side -> R-frame direction
DIRS = {
    "R": {"yes_taker": "buyR", "no_taker": "sellR"},
    "D": {"no_taker": "buyR", "yes_taker": "sellR"},
}


def _load_single_chunk(label_path: Path, raw_dir: Path = DATA_DIR) -> pl.DataFrame:
    """Join one labelled cluster parquet with its raw L2+trade parquet from data/,

    aligning all book levels and trade flows into the unified R-frame.
    """
    raw_path = raw_dir / label_path.name
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw parquet for labelled file: {raw_path}")

    lbl_df = pl.read_parquet(label_path)
    if lbl_df.is_empty():
        return lbl_df

    party = lbl_df["party"][0]
    side_dir = DIRS[party]

    raw_df = (
        pl.read_parquet(
            raw_path,
            columns=[
                "event_seq",
                "timestamp_ns",
                "event_type",
                "bid_prices",
                "bid_sizes",
                "ask_prices",
                "ask_sizes",
                "best_bid",
                "best_ask",
                "spread",
                "trade_size",
                "taker_side",
            ],
        )
        .cast({"event_type": pl.Utf8, "taker_side": pl.Utf8})
        .filter(pl.col("event_type") != "snapshot")
    )

    # Aggregate cluster-level directional volumes and take the end-of-cluster L2 state
    cluster_raw = (
        raw_df.with_columns(
            *[
                pl.when(pl.col("taker_side") == s)
                .then(pl.col("trade_size").cast(pl.Float64) / SIZE_SCALE)
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
            pl.col(
                "bid_prices",
                "bid_sizes",
                "ask_prices",
                "ask_sizes",
                "best_bid",
                "best_ask",
                "spread",
            ).last(),
        )
    )

    # Join on (timestamp_ns, event_seq)
    df = lbl_df.join(
        cluster_raw,
        on=["timestamp_ns", "event_seq"],
        how="inner",
        maintain_order="left",
    )

    # Convert L2 book into unified R-frame:
    # - r_bid_*: resting liquidity supporting R (depleted by sellR takers)
    # - r_ask_*: resting liquidity opposing R (depleted by buyR takers)
    # On the D book, D_yes == R_no, so D bids (at price p) are R asks (at price 1000 - p),
    # and D asks (at price p) are R bids (at price 1000 - p).
    if party == "R":
        df = df.with_columns(
            (pl.col("best_bid").cast(pl.Float64) / PRICE_SCALE).alias("r_best_bid"),
            (pl.col("best_ask").cast(pl.Float64) / PRICE_SCALE).alias("r_best_ask"),
            pl.col("bid_prices").alias("r_bid_prices"),
            (pl.col("bid_sizes").cast(pl.List(pl.Float64)) / SIZE_SCALE).alias(
                "r_bid_sizes"
            ),
            pl.col("ask_prices").alias("r_ask_prices"),
            (pl.col("ask_sizes").cast(pl.List(pl.Float64)) / SIZE_SCALE).alias(
                "r_ask_sizes"
            ),
        )
    else:
        df = df.with_columns(
            ((PRICE_SCALE - pl.col("best_ask").cast(pl.Float64)) / PRICE_SCALE).alias(
                "r_best_bid"
            ),
            ((PRICE_SCALE - pl.col("best_bid").cast(pl.Float64)) / PRICE_SCALE).alias(
                "r_best_ask"
            ),
            pl.col("ask_prices")
            .list.eval(PRICE_SCALE - pl.element())
            .cast(pl.List(pl.Int32))
            .alias("r_bid_prices"),
            (pl.col("ask_sizes").cast(pl.List(pl.Float64)) / SIZE_SCALE).alias(
                "r_bid_sizes"
            ),
            pl.col("bid_prices")
            .list.eval(PRICE_SCALE - pl.element())
            .cast(pl.List(pl.Int32))
            .alias("r_ask_prices"),
            (pl.col("bid_sizes").cast(pl.List(pl.Float64)) / SIZE_SCALE).alias(
                "r_ask_sizes"
            ),
        )

    # Extract commonly used R-frame top-of-book & full-book scalars for convenience
    df = df.with_columns(
        ((pl.col("r_best_bid") + pl.col("r_best_ask")) / 2.0).alias("r_mid"),
        (pl.col("spread").cast(pl.Float64) / PRICE_SCALE).alias("spread_prob"),
        pl.col("timestamp_ns")
        .diff()
        .over("chunk_id")
        .fill_null(0)
        .truediv(1_000_000.0)
        .alias("dt_cluster_ms"),
        pl.col("r_bid_sizes")
        .list.get(0, null_on_oob=True)
        .fill_null(0.0)
        .alias("r_bid_sz_1"),
        pl.col("r_ask_sizes")
        .list.get(0, null_on_oob=True)
        .fill_null(0.0)
        .alias("r_ask_sz_1"),
        pl.col("r_bid_sizes")
        .list.slice(0, 3)
        .list.sum()
        .fill_null(0.0)
        .alias("r_bid_sz_top3"),
        pl.col("r_ask_sizes")
        .list.slice(0, 3)
        .list.sum()
        .fill_null(0.0)
        .alias("r_ask_sz_top3"),
        pl.col("r_bid_sizes")
        .list.slice(0, 5)
        .list.sum()
        .fill_null(0.0)
        .alias("r_bid_sz_top5"),
        pl.col("r_ask_sizes")
        .list.slice(0, 5)
        .list.sum()
        .fill_null(0.0)
        .alias("r_ask_sz_top5"),
        pl.col("r_bid_sizes").list.sum().fill_null(0.0).alias("r_bid_sz_total"),
        pl.col("r_ask_sizes").list.sum().fill_null(0.0).alias("r_ask_sz_total"),
        pl.col("r_bid_sizes").list.len().cast(pl.Int32).alias("r_n_bid_levels"),
        pl.col("r_ask_sizes").list.len().cast(pl.Int32).alias("r_n_ask_levels"),
    )

    return df.drop(["bid_prices", "bid_sizes", "ask_prices", "ask_sizes"])


class MarketDataset:
    """Container holding aligned cluster-level data across all 3 contracts."""

    def __init__(self, data_dir: Path = DATA_DIR, label_dir: Path = LABEL_DIR):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.by_contract: dict[str, pl.DataFrame] = {}

    def load(
        self, contracts: list[str] | None = None, verbose: bool = True
    ) -> "MarketDataset":
        target_contracts = contracts or ALL_CONTRACTS
        files = sorted(self.label_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(
                f"No labelled parquet files found in {self.label_dir}. Run make_labels.py first."
            )

        grouped: dict[str, list[pl.DataFrame]] = {c: [] for c in target_contracts}
        for f in files:
            tk = f.stem.split("_")[0]  # e.g., CONTROLS-2026-R
            contract = tk.rsplit("-", 1)[0]
            if contract not in grouped:
                continue
            chunk_df = _load_single_chunk(f, self.data_dir)
            if not chunk_df.is_empty():
                grouped[contract].append(chunk_df)

        for contract, dfs in grouped.items():
            if not dfs:
                continue
            combined = pl.concat(dfs, how="vertical").sort(["chunk_id", "cluster_id"])
            self.by_contract[contract] = combined
            if verbose:
                splits = (
                    combined.group_by("split")
                    .len()
                    .sort("split")
                    .to_dict(as_series=False)
                )
                split_str = ", ".join(
                    f"{s}={n:,}" for s, n in zip(splits["split"], splits["len"])
                )
                print(
                    f"[Dataset] {contract:<14} | chunks={combined['chunk_id'].n_unique():2d} | total={len(combined):9,d} ({split_str})"
                )

        return self

    def get_primary_train_val(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Return (train_df, val_df) on CONTROLS-2026.

        Note: We pass chunks containing train/val so models can compute rolling
        features cleanly across chunk boundaries, while filtering `split ==
        'train'` or `'val'` for loss evaluation.
        """
        df = self.by_contract[PRIMARY_CONTRACT]
        train_df = df.filter(pl.col("split") == "train")
        # Include train+val rows so rolling windows at the start of 'val' have causal warm-up
        val_df = df.filter(pl.col("split").is_in(["train", "val"]))
        return train_df, val_df

    def get_contract_full(self, contract: str) -> pl.DataFrame:
        """Return full DataFrame for a contract (so causal rolling features warm up before test split)."""
        return self.by_contract[contract]
