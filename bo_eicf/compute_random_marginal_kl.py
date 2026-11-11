#!/usr/bin/env python3
"""Reconstruct a random baseline and compute its marginal h-space KL."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from compute_table_metrics import (
    DEFAULT_H_COLUMNS,
    _h_kl_to_uniform,
    _labels_from,
    _mean_sd,
    _prepare_metric_h,
    _training_reference,
)
from config.settings import Config
from utils.detection import DetectionCalculator
from utils.visualization import _simulate_random_baseline


def _numeric_detection_limits(config):
    limits = {}
    for filt in config.detection_filters:
        raw = config.ZTF_LIMITS.get(filt)
        if isinstance(raw, dict) and "limit" in raw:
            limits[filt] = float(raw["limit"])
        elif raw is not None:
            limits[filt] = float(raw)
    return limits


def _random_path(run_root, plots_root, seed):
    candidates = [
        run_root / f"seed_{seed}" / "random_baseline_samples.csv",
        plots_root / f"seed_{seed}" / "random_baseline_samples.csv",
        run_root / f"seed_{seed}plots" / "random_baseline_samples.csv",
    ]
    return next((path for path in candidates if path.exists()), None)


def _reconstruct_random(seed, budget, config, gp_models, scaler_x, calculator, flat_limits):
    return _simulate_random_baseline(
        n_new=budget,
        bounds=config.bounds,
        gp_models=gp_models,
        scaler_X=scaler_x,
        flat_limits=flat_limits,
        detection_calc=calculator,
        feature_columns=config.feature_columns,
        seed=seed + 100000,
    )


def _random_metrics(random_df, reference, edges, budget):
    random_df = _prepare_metric_h(random_df.iloc[:budget].copy())
    missing = [column for column in DEFAULT_H_COLUMNS if column not in random_df]
    if missing:
        raise ValueError(f"Random baseline is missing h columns: {', '.join(missing)}")

    labels = _labels_from(random_df)
    measurable = random_df.iloc[np.flatnonzero(labels == 1)][DEFAULT_H_COLUMNS]
    measurable = measurable.apply(pd.to_numeric, errors="coerce").dropna()
    final = pd.concat([reference, measurable], ignore_index=True)
    kl_sum, kl_mean, components = _h_kl_to_uniform(
        final, DEFAULT_H_COLUMNS, edges
    )
    return {
        "measurable_count": int(labels.sum()),
        "measurable_fraction": float(labels.mean()),
        "marginal_kl_sum": kl_sum,
        "marginal_kl_mean": kl_mean,
        **{f"kl_{name}": value for name, value in components.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", help="Directory containing seed_1, seed_2, ...")
    parser.add_argument("--plots-root", default="plots")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--gp-cache", default="gp_cache/initial_gp_models_h11.joblib")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--budget", type=int, default=80)
    parser.add_argument("--bins", type=int, default=12)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save-reconstructed", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    plots_root = Path(args.plots_root).resolve()
    reference, edges = _training_reference(args.data_dir, args.threshold, args.bins)

    config = Config()
    flat_limits = _numeric_detection_limits(config)
    calculator = DetectionCalculator(
        detection_filters=config.detection_filters,
        flat_limits=flat_limits,
    )
    for filt in config.detection_filters:
        raw = config.ZTF_LIMITS.get(filt)
        if isinstance(raw, dict) and "error" in raw:
            calculator.filter_errors[filt] = float(raw["error"])

    cache = joblib.load(args.gp_cache)
    gp_models = cache["gp_models"]
    scaler_x = cache["scaler_X"]

    rows = []
    for seed in args.seeds:
        saved_path = _random_path(run_root, plots_root, seed)
        if saved_path is not None:
            random_df = pd.read_csv(saved_path)
            source = str(saved_path)
        else:
            random_df = _reconstruct_random(
                seed, args.budget, config, gp_models, scaler_x,
                calculator, flat_limits,
            )
            source = "reconstructed"
            if args.save_reconstructed:
                output_dir = run_root / "reconstructed_random" / f"seed_{seed}"
                output_dir.mkdir(parents=True, exist_ok=True)
                saved_path = output_dir / "random_baseline_samples.csv"
                random_df.to_csv(saved_path, index=False)
                source = str(saved_path)

        if len(random_df) < args.budget:
            raise ValueError(
                f"Seed {seed} has {len(random_df)} random candidates; "
                f"{args.budget} are required"
            )
        rows.append({
            "seed": seed,
            "source": source,
            **_random_metrics(random_df, reference, edges, args.budget),
        })

    per_seed = pd.DataFrame(rows)
    fraction_mean, fraction_sd = _mean_sd(per_seed["measurable_fraction"])
    kl_mean, kl_sd = _mean_sd(per_seed["marginal_kl_sum"])

    output = Path(
        args.output or run_root / f"random_kl_first_{args.budget}.csv"
    )
    per_seed.to_csv(output, index=False)

    print(per_seed.to_string(index=False))
    print("\nTable values across seeds (sample standard deviation):")
    print(f"Random measurable fraction: {fraction_mean:.3f} +/- {fraction_sd:.3f}")
    print(f"Random marginal KL sum:     {kl_mean:.3f} +/- {kl_sd:.3f}")
    print(f"\nSaved {output}")


if __name__ == "__main__":
    main()
