"""
experiments/compare_pilot.py
----------------------------
Aggregate the local-vs-API pilot: per (model, task), summarise solve rate,
final error, Koopman rho, and the behavioral metrics across replicates.

Usage: python experiments/compare_pilot.py
"""

import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stability_analysis import load_trajectory, build_observables, edmd_spectral_radius  # noqa: E402
from behavioral_metrics import compute_metrics  # noqa: E402

MODELS = ["haiku", "qwen"]
TASKS = ["C", "S", "S_infeasible"]
LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
MIN_FIT = 14  # need ~depth*(m+1)+ for a meaningful EDMD fit


def cell_rows(model, task):
    rows = []
    for f in sorted(glob.glob(os.path.join(LOGS, f"{model}_{task}_r*.jsonl"))):
        recs = load_trajectory(f)
        if not recs:
            continue
        m = compute_metrics(recs)
        rho = float("nan")
        if len(recs) >= MIN_FIT:
            O, _ = build_observables(recs)
            rho, _ = edmd_spectral_radius(O, depth=2)
        rows.append((m, rho, len(recs)))
    return rows


def _mean(rows, key):
    vals = [r[0].get(key) for r in rows
            if isinstance(r[0].get(key), (int, float)) and not isinstance(r[0].get(key), bool)]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    hdr = (f"{'model':6} {'n':>2} {'solve%':>6} {'finErr':>6} {'rho':>6} "
           f"{'badstp':>6} {'osc':>4} {'persist':>7} {'recov':>5} {'falseC':>6} {'len':>4}")
    for task in TASKS:
        print(f"\n=== task: {task} ===")
        print(hdr)
        for model in MODELS:
            rows = cell_rows(model, task)
            if not rows:
                print(f"{model:6}  (no runs yet)")
                continue
            n = len(rows)
            solve = 100 * np.mean([1.0 if r[0].get("solved") else 0.0 for r in rows])
            finerr = np.mean([r[0]["final_error_percent"] for r in rows])
            rhos = [r[1] for r in rows if not np.isnan(r[1])]
            rho = float(np.mean(rhos)) if rhos else float("nan")
            meanlen = np.mean([r[2] for r in rows])
            print(f"{model:6} {n:>2} {solve:>5.0f}% {finerr:>6.1f} {rho:>6.3f} "
                  f"{_mean(rows,'bad_step_rate'):>6.2f} {_mean(rows,'oscillation_count'):>4.1f} "
                  f"{_mean(rows,'persistence'):>7.2f} {_mean(rows,'recovery'):>5.2f} "
                  f"{_mean(rows,'false_convergence_rate'):>6.2f} {meanlen:>4.0f}")


if __name__ == "__main__":
    main()
