#!/usr/bin/env python3
"""Compute measurable yield, marginal h-space KL, and BO overhead by seed."""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import Config
from data.loader import load_and_prepare_data
from utils.detection import DetectionCalculator
from utils.h_columns import prepare_h_columns
from utils.visualization import _kl_to_uniform


DEFAULT_H_COLUMNS = [
    "snr", "area90", "log10_mej_dyn", "log10_mej_wind",
    "g", "r", "i", "distance",
]


def _prepare_metric_h(df):
    df = prepare_h_columns(df)
    for raw, logged in (
        ("mej_dyn", "log10_mej_dyn"),
        ("mej_wind", "log10_mej_wind"),
    ):
        if logged not in df.columns and raw in df.columns:
            values = pd.to_numeric(df[raw], errors="coerce")
            df[logged] = np.log10(values.where(values > 0))
    return df


def _compute_training_detection_probs(magnitudes, calculator, n_rows):
    probabilities = np.zeros(n_rows, dtype=float)
    arrays = {
        filt: pd.to_numeric(values, errors="coerce").to_numpy(float)
        for filt, values in magnitudes.items()
    }
    for index in range(n_rows):
        row = {
            filt: float(values[index])
            for filt, values in arrays.items()
            if index < len(values) and np.isfinite(values[index])
        }
        probabilities[index], _ = calculator.compute_detection_probability(row)
    return probabilities


def _numeric_detection_limits(config):
    limits = {}
    for filt in config.detection_filters:
        raw = config.ZTF_LIMITS.get(filt)
        if isinstance(raw, dict) and "limit" in raw:
            limits[filt] = float(raw["limit"])
        elif raw is not None:
            limits[filt] = float(raw)
    return limits


def _labels_from(df, threshold=0.5):
    for col in ("active_search_detectable", "true_detectable", "detected", "detectable"):
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).to_numpy()
    for col in ("P_detect", "detection_probability", "true_detection_probability"):
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").fillna(0)
            return (values >= threshold).astype(int).to_numpy()
    raise ValueError("No detection-label or detection-probability column was found")


def _finite_h_frame(df, columns):
    out = pd.DataFrame(index=df.index)
    for col in columns:
        out[col] = pd.to_numeric(df[col], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")


def _make_bin_edges(reference, columns, n_bins):
    edges = {}
    for col in columns:
        values = pd.to_numeric(reference[col], errors="coerce").to_numpy(float)
        values = values[np.isfinite(values)]
        lo, hi = float(values.min()), float(values.max())
        if np.isclose(lo, hi):
            pad = max(abs(lo) * 1e-9, 1e-12)
        else:
            pad = 1e-9 * (hi - lo)
        edges[col] = np.linspace(lo - pad, hi + pad, n_bins + 1)
    return edges


def _h_kl_to_uniform(df, columns, edges):
    epsilon = 1e-12
    values = []
    for col in columns:
        data = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        data = data[np.isfinite(data)]
        edge = edges[col]
        counts = np.histogram(data, bins=edge)[0].astype(float)
        probabilities = (counts + epsilon) / (counts.sum() + epsilon * len(counts))
        uniform = 1.0 / len(counts)
        kl = float(np.sum(probabilities * np.log(probabilities / uniform)))
        values.append(kl)
    return float(np.sum(values)), float(np.mean(values)), dict(zip(columns, values))


def _training_reference(data_dir: str, threshold: float, n_bins: int):
    config = Config()
    calculator = DetectionCalculator(
        detection_filters=config.detection_filters,
        flat_limits=_numeric_detection_limits(config),
    )
    for filt in config.detection_filters:
        raw = config.ZTF_LIMITS.get(filt)
        if isinstance(raw, dict) and "error" in raw:
            calculator.filter_errors[filt] = float(raw["error"])

    training = _prepare_metric_h(load_and_prepare_data(base_path=data_dir))
    mags = {
        col: training[col]
        for col in config.detection_filters
        if col in training.columns
    }
    probabilities = _compute_training_detection_probs(mags, calculator, len(training))
    training = training.assign(P_detect=probabilities)
    reference = _finite_h_frame(
        training[training["P_detect"] >= threshold], DEFAULT_H_COLUMNS
    )
    edges = _make_bin_edges(reference, DEFAULT_H_COLUMNS, n_bins=n_bins)
    return reference, edges


def _random_csv(root: Path, seed_dir: Path, plots_root=None) -> Path:
    candidates = [
        root / f"{seed_dir.name}plots" / "random_baseline_samples.csv",
        seed_dir / "random_baseline_samples.csv",
    ]
    if plots_root is not None:
        candidates.insert(0, Path(plots_root).resolve() / seed_dir.name / "random_baseline_samples.csv")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No random_baseline_samples.csv found for {seed_dir.name}")


def _require_h(df: pd.DataFrame, label: str):
    missing = [col for col in DEFAULT_H_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"{label} is missing h columns: {', '.join(missing)}. "
            "These values were not saved by older runs; rerun with the updated code."
        )


def _method_metrics(df, reference, edges, budget, overhead=False):
    df = _prepare_metric_h(df.iloc[:budget].copy())
    if len(df) < budget:
        raise ValueError(f"Requested {budget} candidates but only {len(df)} are available")
    _require_h(df, "results file")
    labels = _labels_from(df)
    selected = _finite_h_frame(df.iloc[np.flatnonzero(labels == 1)], DEFAULT_H_COLUMNS)
    final = pd.concat([reference, selected], ignore_index=True)
    kl_sum, _, _ = _h_kl_to_uniform(final, DEFAULT_H_COLUMNS, edges)
    result = {
        "measurable_fraction": float(labels.mean()),
        "marginal_kl_sum": float(kl_sum),
    }
    if overhead:
        if "algorithm_overhead_seconds" not in df.columns:
            raise ValueError(
                "algorithm_overhead_seconds is missing. Older runs did not record "
                "timing and must be rerun to report overhead."
            )
        result["algorithm_overhead_seconds"] = float(
            pd.to_numeric(df["algorithm_overhead_seconds"], errors="coerce").mean()
        )
    return result


def _mean_sd(values):
    values = np.asarray(values, dtype=float)
    return float(np.mean(values)), float(np.std(values, ddof=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", help="Directory containing seed_1, seed_2, ...")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--plots-root", default="plots")
    parser.add_argument("--budget", type=int, default=80)
    parser.add_argument("--bins", type=int, default=12)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(args.run_root).resolve()
    reference, edges = _training_reference(args.data_dir, args.threshold, args.bins)
    rows = []
    for seed_dir in sorted(root.glob("seed_*")):
        if not seed_dir.is_dir() or not (seed_dir / "closed_loop_results.csv").exists():
            continue
        seed = seed_dir.name.removeprefix("seed_")
        bo = pd.read_csv(seed_dir / "closed_loop_results.csv")
        random = pd.read_csv(_random_csv(root, seed_dir, args.plots_root))
        bo_metrics = _method_metrics(bo, reference, edges, args.budget, overhead=True)
        random_metrics = _method_metrics(random, reference, edges, args.budget)
        rows.extend([
            {"seed": seed, "method": "EI-CF", **bo_metrics},
            {"seed": seed, "method": "Random", **random_metrics},
        ])

    if not rows:
        raise RuntimeError(f"No seed results found under {root}")

    per_seed = pd.DataFrame(rows)
    output = Path(args.output or root / f"table_metrics_first_{args.budget}.csv")
    per_seed.to_csv(output, index=False)

    summary_rows = []
    for method, group in per_seed.groupby("method", sort=False):
        row = {"method": method, "n_seeds": len(group)}
        for column in (
            "measurable_fraction",
            "marginal_kl_sum",
            "algorithm_overhead_seconds",
        ):
            if column in group and group[column].notna().any():
                mean, sd = _mean_sd(group[column].dropna())
                row[f"{column}_mean"] = mean
                row[f"{column}_sd"] = sd
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary_path = output.with_name(output.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(per_seed.to_string(index=False))
    print("\nSummary:\n" + summary.to_string(index=False))
    print(f"\nSaved {output}\nSaved {summary_path}")


if __name__ == "__main__":
    main()
