import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import polars as pl

from dataset import ALL_CONTRACTS, PRIMARY_CONTRACT, MarketDataset
from evaluate import evaluate_predictions, format_comparison_table
from models import TargetConfig, discover_models, get_model

REGIME_LABELS = {
    "CONTROLS-2026": "In-Domain (Liquid)",
    "SENATETX-26": "Homo-Transfer (Liquid)",
    "SENATEGA-26": "Hetero-Transfer (Boundary)",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=("Kalshi Taker Flow Spike Prediction & Zero-Shot Transfer Pipeline")
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help=(
            "Model names to run (e.g. --models hawkes_ewma single_l1"
            " linear_glm) or 'all'"
        ),
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=100,
        choices=[50, 100, 200],
        help="Forward cluster horizon H (default: 100)",
    )
    parser.add_argument(
        "--label-type",
        type=str,
        default="y",
        choices=["y", "y_sd", "y_onset"],
        help=(
            "Spike label definition: 'y' (quantile), 'y_sd' (mu+2.5*sd), or 'y_onset'"
        ),
    )
    parser.add_argument(
        "--include-followers",
        action="store_true",
        help=(
            "If set, include cross-book arbitrage follower spikes (mask_leader=False)"
        ),
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results",
        help="Directory to save summary metrics and optional predictions",
    )
    parser.add_argument(
        "--save-preds",
        action="store_true",
        help="If set, save per-cluster test predictions to results/",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional label appended to the run directory name (e.g. 'sign_fix')",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    registry = discover_models()

    if "all" in args.models:
        selected_models = list(registry.keys())
    else:
        selected_models = args.models

    cfg = TargetConfig(
        horizon=args.horizon,
        label_prefix=args.label_type,
        leader_only=not args.include_followers,
        markout_col=f"mid_fwd_h{args.horizon}",
    )

    print("=" * 90)
    print(
        f"Kalshi Spike Pipeline | H={cfg.horizon} | Label={cfg.label_prefix} |"
        f" Leader-Only={cfg.leader_only}"
    )
    print(f"Registered Models     : {list(registry.keys())}")
    print(f"Selected Models       : {selected_models}")
    print("=" * 90)

    t0 = time.perf_counter()
    ds = MarketDataset().load(ALL_CONTRACTS, verbose=True)
    print(f"[*] Dataset loaded in {time.perf_counter() - t0:.2f}s\n")

    train_df, val_df = ds.get_primary_train_val()
    summary_rows = []

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.tag:
        run_id += f"_{args.tag}"
    out_dir = Path(args.save_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=False)  # never reuse an existing run dir
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(
            {**vars(args), "selected_models": selected_models, "run_id": run_id},
            f,
            indent=2,
        )
    print(f"[*] Results will be saved to {out_dir}")

    for model_name in selected_models:
        print(f"--- [Model: {model_name}] Training on {PRIMARY_CONTRACT} ---")
        t_fit = time.perf_counter()
        model = get_model(model_name)
        model.fit(train_df, val_df, cfg)
        print(f"    Fit completed in {time.perf_counter() - t_fit:.2f}s")

        # Validate on CONTROLS-2026 val split
        val_preds = model.predict_proba(val_df, cfg)
        val_metrics = evaluate_predictions(val_df, val_preds, cfg, split="val")
        if val_metrics:
            print(
                f"    [Val  | {PRIMARY_CONTRACT}] PR-AUC={val_metrics['PR-AUC']:.4f} "
                f"(Lift={val_metrics['PR-Lift (x)']}x) | Brier={val_metrics['Brier']:.5f}"
            )

        # Locked Out-of-Sample Evaluation across all 3 contracts on split == 'test'
        for contract in ALL_CONTRACTS:
            if contract not in ds.by_contract:
                continue
            full_contract_df = ds.get_contract_full(contract)
            preds = model.predict_proba(full_contract_df, cfg)
            test_metrics = evaluate_predictions(
                full_contract_df, preds, cfg, split="test"
            )
            if not test_metrics:
                continue

            row = {
                "Model": model_name,
                "Contract": contract,
                "Regime": REGIME_LABELS.get(contract, "Test"),
                **test_metrics,
            }
            summary_rows.append(row)

            if args.save_preds:
                pred_file = (
                    out_dir
                    / f"preds_{model_name}_{contract}_h{cfg.horizon}_{cfg.label_prefix}.parquet"
                )
                preds.write_parquet(pred_file, compression="zstd")

        print()

    print("=" * 90)
    print(
        f"LOCKED TEST SET COMPARISON (H={cfg.horizon},"
        f" Label={cfg.label_prefix}, Leader-Only={cfg.leader_only})"
    )
    print("=" * 90)
    print(format_comparison_table(summary_rows, cfg))

    if summary_rows:
        summary_path = (
            out_dir
            / f"summary_h{cfg.horizon}_{cfg.label_prefix}_leader{int(cfg.leader_only)}.csv"
        )
        pl.DataFrame(summary_rows).write_csv(summary_path)
        print(f"\n[*] Saved summary table to {summary_path}")
