#!/usr/bin/env python3
"""Run the paper's EI-CF-without-ROBOT baseline for multiple seeds."""

import argparse
import os

import pandas as pd

from run_bo import run_single


def parse_seeds(text):
    seeds = []
    for item in text.split(","):
        item = item.strip()
        if "-" in item:
            start, stop = (int(value) for value in item.split("-", 1))
            seeds.extend(range(start, stop + 1))
        elif item:
            seeds.append(int(item))
    return seeds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="1-3")
    parser.add_argument("--n-iterations", type=int, default=80)
    parser.add_argument("--output-root", default="out/eicf_no_robot")
    parser.add_argument("--candidate-samples", type=int, default=1500)
    parser.add_argument("--candidate-uniform-samples", type=int, default=300)
    parser.add_argument("--eicf-mc-samples", type=int, default=16)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    rows = []
    for seed in parse_seeds(args.seeds):
        result = run_single(
            seed=seed,
            n_iterations=args.n_iterations,
            acquisition="EI",
            output_folder=os.path.join(args.output_root, f"seed_{seed}"),
            make_plots=not args.no_plots,
            candidate_samples=args.candidate_samples,
            candidate_uniform_samples=args.candidate_uniform_samples,
            eicf_mc_samples=args.eicf_mc_samples,
            active_search=True,
        )
        events = result["new_events_df"]
        labels = pd.to_numeric(
            events.get("active_search_detectable", pd.Series(dtype=float)),
            errors="coerce",
        )
        overhead = pd.to_numeric(
            events.get("algorithm_overhead_seconds", pd.Series(dtype=float)),
            errors="coerce",
        )
        rows.append(
            {
                "seed": seed,
                "events": len(events),
                "measurable_fraction": labels.mean(),
                "mean_algorithm_overhead_seconds": overhead.mean(),
                "output_folder": result["output_folder"],
            }
        )

    path = os.path.join(args.output_root, "multi_seed_summary.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved multi-seed summary to {path}")


if __name__ == "__main__":
    main()
