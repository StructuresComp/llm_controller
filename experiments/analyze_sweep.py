# ---------------------------------------------------------------------------
# experiments/analyze_sweep.py
# ---------------------------------------------------------------------------
# Post-run analysis for a run_sweep.py sweep. Fully standalone — no imports
# from `src/`, no dependency on the controller's runtime (LangGraph, LLM
# providers, physics solver). Only reads plain JSONL trajectory files that
# cli.py already wrote to logs/.
#
# Two deliberate simplifications, so nothing here needs benchmark internals:
#   - Baseline error e_0 is the *first logged iteration's* error_percent, not
#     a reconstructed pre-control error (that would need src.models/initial_q0
#     and src.benchmarks loss functions).
#   - "Success" uses one global SUCCESS_THRESHOLD_PERCENT constant below, not
#     each benchmark's own success_threshold (that would need src.benchmarks).
#
# Runs are discovered by scanning logs/*.jsonl directly -- not via run_sweep.py's
# commands.txt, which only reflects whatever MODELS/TARGETS its *last* invocation
# happened to use and gets overwritten every run. logs/ is the actual data.
#
# Model and target are inferred per run, not from a hardcoded list: target comes
# from the jsonl header's own target_shape field (ground truth), and that's used
# to split the stem "<model>_<target>_run_<NN>" back into its model part -- no
# duplication of run_sweep.py's MODELS/TARGETS, so nothing here can drift out of
# sync with it.
#
# Usage (run from anywhere):
#   python3 experiments/analyze_sweep.py
#   python3 experiments/analyze_sweep.py --logs-dir logs/
#   python3 experiments/analyze_sweep.py --success-threshold 10.0
# ---------------------------------------------------------------------------

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

EXPERIMENTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT    = EXPERIMENTS_DIR.parent

DEFAULT_LOGS_DIR      = PROJECT_ROOT / "logs"
DEFAULT_FIGURES_DIR   = EXPERIMENTS_DIR / "figures"
DEFAULT_METRICS_CSV   = EXPERIMENTS_DIR / "metrics_per_run.csv"

EPS = 1e-6
SUCCESS_THRESHOLD_PERCENT = 5.0  # error_percent below this counts as "solved"

_STEM_SUFFIX_RE = re.compile(r"^(?P<rest>.+)_run_(?P<rep>\d+)$")


# ---------------------------------------------------------------------------
# 1. DISCOVERING RUNS
# ---------------------------------------------------------------------------

def discover_stems(logs_dir: Path) -> list[str]:
    """Every run present in logs_dir, found by directly scanning for <stem>.jsonl files."""
    return sorted(p.stem for p in logs_dir.glob("*.jsonl"))


def parse_stem(stem: str, target: str) -> tuple[str, int] | None:
    """
    Recover (model, repetition) from a stem "<model>_<target>_run_<NN>", given
    the target already read from the run's own jsonl header (ground truth).
    No hardcoded model/target list needed: the known target is the anchor that
    lets us split off the model part unambiguously. `model` here is whatever
    sanitized form run_sweep.py's _sanitize() produced (":" -> "-"), since
    that's what's actually in the filename.
    """
    m = _STEM_SUFFIX_RE.match(stem)
    if not m:
        return None
    rest, rep = m.group("rest"), int(m.group("rep"))
    suffix = f"_{target}"
    if not rest.endswith(suffix):
        return None
    model = rest[: -len(suffix)]
    return model, rep


# ---------------------------------------------------------------------------
# 2. LOADING TRAJECTORIES
# ---------------------------------------------------------------------------

def load_run(jsonl_path: Path) -> dict:
    """
    Load one run's JSONL file. Returns:
      {"rod_length": float, "target_shape": str, "target_tip": [x,y] | None,
       "iterations": [{"iteration", "bc", "error_percent"}, ...]}
    Multiple snapshots can share one iteration (SAVE_STRIDE > 1) — only the
    last snapshot per iteration is kept, matching what the run actually scored.
    """
    rod_length = None
    target_shape = None
    target_tip = None
    by_iteration: dict[int, dict] = {}

    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("type") == "header":
                rod_length = record["physical_system"]["rod_length"]
                target_shape = record.get("target_shape")
                target_tip = record.get("target_tip")
                continue
            by_iteration[record["iteration"]] = record  # last write wins

    iterations = [by_iteration[i] for i in sorted(by_iteration)]
    return {"rod_length": rod_length, "target_shape": target_shape,
            "target_tip": target_tip, "iterations": iterations}


# ---------------------------------------------------------------------------
# 3. PER-RUN METRICS
# ---------------------------------------------------------------------------

def compute_metrics(run: dict) -> dict:
    """Sections 3-6: normalized controls, basic performance, error stability, control gain."""
    iterations = run["iterations"]
    L = run["rod_length"]
    n = len(iterations)

    metrics = {"n_iterations": n}
    if n == 0:
        return metrics

    e = np.array([it["error_percent"] for it in iterations], dtype=float)
    u = np.array(
        [[it["bc"]["x0"] / L, it["bc"]["y0"] / L, it["bc"]["theta"] / np.pi] for it in iterations],
        dtype=float,
    )

    # --- Section 4: basic performance ---
    e0 = e[0] + EPS
    metrics["E_T"]              = float(e[-1])
    metrics["E_final_norm"]     = float(e[-1] / e0)
    metrics["E_best_norm"]      = float(e.min() / e0)
    metrics["AUC_E"]            = float(np.mean(e) / e0)
    metrics["D_regress"]        = float((e[-1] - e.min()) / e0)
    metrics["success"]          = bool(e[-1] < SUCCESS_THRESHOLD_PERCENT)

    if n < 2:
        return metrics  # nothing below needs at least one transition

    diffs = np.diff(e)  # length T = n - 1
    metrics["R_bad"]     = float(np.mean(diffs > 0))
    metrics["R_improve"] = float(np.mean(diffs < 0))
    signs = np.sign(diffs)
    nonzero_signs = signs[signs != 0]
    metrics["N_osc"] = int(np.sum(nonzero_signs[1:] != nonzero_signs[:-1])) if len(nonzero_signs) > 1 else 0
    metrics["S_max"] = float(diffs.max())

    # --- Section 6: control-gain metrics ---
    deltas = np.diff(u, axis=0)          # length T, direction vectors
    a = np.linalg.norm(deltas, axis=1)   # length T, control step sizes
    e_head = e[:-1]                      # e_t for t = 0..T-1

    metrics["a_bar"]   = float(np.mean(a))
    per_step_gain      = a / (e_head + EPS)
    metrics["K_eff"]   = float(np.mean(per_step_gain))
    metrics["sigma_K"] = float(np.std(per_step_gain))

    improvement = np.maximum(e_head - e[1:], 0.0)
    harm        = np.maximum(e[1:] - e_head, 0.0)
    a_sum       = np.sum(a) + EPS
    metrics["G_useful"]   = float(np.sum(improvement) / a_sum)
    metrics["G_harmful"]  = float(np.sum(harm) / a_sum)

    if len(deltas) >= 2:
        dots  = np.sum(deltas[1:] * deltas[:-1], axis=1)
        metrics["R_flip"] = float(np.mean(dots < 0))
        jerk  = np.linalg.norm(np.diff(deltas, axis=0), axis=1)
        metrics["H_u"] = float(np.mean(jerk))

    return metrics


# ---------------------------------------------------------------------------
# 4. AGGREGATE TABLES (section 7)
# ---------------------------------------------------------------------------

def _group_by(rows: list[dict], keys: list[str]) -> dict[tuple, list[dict]]:
    """Group rows by the given key(s), preserving first-seen group order."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        gkey = tuple(row[k] for k in keys)
        groups.setdefault(gkey, []).append(row)
    return groups


def _mean(rows: list[dict], field: str) -> float:
    values = [r[field] for r in rows if field in r]
    return float(np.mean(values)) if values else float("nan")


def build_summary_tables(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Model x Task table, and overall Model phenotype table, matching TODO_experiments.md section 7."""
    by_model_task = []
    for (model, target), group in _group_by(rows, ["model", "target"]).items():
        by_model_task.append({
            "Model":        model,
            "Task":         target,
            "Success %":    100 * _mean(group, "success"),
            "Final error":  _mean(group, "E_T"),
            "Best error":   _mean(group, "E_best_norm"),
            "AUC":          _mean(group, "AUC_E"),
            "Bad-step %":   100 * _mean(group, "R_bad"),
            "Osc. count":   _mean(group, "N_osc"),
            "Gain":         _mean(group, "K_eff"),
            "Gain SD":      _mean(group, "sigma_K"),
            "Useful gain":  _mean(group, "G_useful"),
            "Harmful gain": _mean(group, "G_harmful"),
            "Flip %":       100 * _mean(group, "R_flip"),
        })

    by_model = []
    for (model,), group in _group_by(rows, ["model"]).items():
        by_model.append({
            "Model":       model,
            "Success %":   100 * _mean(group, "success"),
            "Final error": _mean(group, "E_T"),
            "AUC":         _mean(group, "AUC_E"),
            "Bad-step %":  100 * _mean(group, "R_bad"),
            "Gain":        _mean(group, "K_eff"),
            "Gain SD":     _mean(group, "sigma_K"),
            "Useful gain": _mean(group, "G_useful"),
            "Flip %":      100 * _mean(group, "R_flip"),
        })

    return by_model_task, by_model


def write_csv(rows: list[dict], path: Path) -> None:
    """Write a list of flat dicts to CSV, taking the field union across all rows as columns."""
    if not rows:
        return
    fieldnames = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# 5. FIGURES (section 8)
# ---------------------------------------------------------------------------

def make_figures(rows: list[dict], runs: dict[str, dict], figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    models  = sorted({r["model"] for r in rows})
    targets = sorted({r["target"] for r in rows})

    # Error vs iteration, representative rollout (repetition 1) per target.
    for target in targets:
        fig, ax = plt.subplots(figsize=(8, 5))
        for model in models:
            match = next((r for r in rows
                          if r["model"] == model and r["target"] == target and r["repetition"] == 1), None)
            if match is None:
                continue
            e = [it["error_percent"] for it in runs[match["stem"]]["iterations"]]
            ax.plot(range(1, len(e) + 1), e, "o-", markersize=3, label=model)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Error (%)")
        ax.set_title(f"Error vs iteration — target {target} (repetition 1)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.savefig(figures_dir / f"error_curves_{target}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    # Model x task heatmap of mean final error.
    grid = np.full((len(models), len(targets)), np.nan)
    groups = _group_by(rows, ["model", "target"])
    for i, model in enumerate(models):
        for j, target in enumerate(targets):
            group = groups.get((model, target))
            if group:
                grid[i, j] = _mean(group, "E_T")
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_title("Mean final error (%) — model x task")
    fig.colorbar(im, ax=ax)
    fig.savefig(figures_dir / "heatmap_final_error.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    # Controller gain vs final error scatter.
    fig, ax = plt.subplots(figsize=(7, 6))
    for model in models:
        sub = [r for r in rows if r["model"] == model and "K_eff" in r]
        ax.scatter([r["K_eff"] for r in sub], [r["E_T"] for r in sub], label=model, alpha=0.7)
    ax.set_xlabel("Effective gain (K_eff)")
    ax.set_ylabel("Final error (%)")
    ax.set_title("Controller gain vs final error")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(figures_dir / "gain_vs_final_error.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    # Bad-step rate vs useful gain scatter.
    fig, ax = plt.subplots(figsize=(7, 6))
    for model in models:
        sub = [r for r in rows if r["model"] == model and "R_bad" in r]
        ax.scatter([r["R_bad"] for r in sub], [r["G_useful"] for r in sub], label=model, alpha=0.7)
    ax.set_xlabel("Bad-step rate")
    ax.set_ylabel("Useful gain")
    ax.set_title("Bad-step rate vs useful gain")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(figures_dir / "badstep_vs_usefulgain.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    # Failure-mode distribution per model (success vs fail counts).
    fig, ax = plt.subplots(figsize=(8, 5))
    success_counts = [sum(1 for r in rows if r["model"] == m and r.get("success")) for m in models]
    fail_counts    = [sum(1 for r in rows if r["model"] == m and not r.get("success")) for m in models]
    x = np.arange(len(models))
    ax.bar(x, success_counts, label="success", color="seagreen")
    ax.bar(x, fail_counts, bottom=success_counts, label="fail", color="indianred")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("Run count")
    ax.set_title("Success / failure distribution per model")
    ax.legend()
    fig.savefig(figures_dir / "success_distribution.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    # Shape snapshots: best and worst run overall, final configuration.
    scored = [r for r in rows if "E_T" in r]
    best   = min(scored, key=lambda r: r["E_T"])
    worst  = max(scored, key=lambda r: r["E_T"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, row, label in [(axes[0], best, "Best"), (axes[1], worst, "Worst")]:
        q = np.array(runs[row["stem"]]["iterations"][-1]["q"])
        nodes = q.reshape(-1, 2)
        ax.plot(nodes[:, 0], nodes[:, 1], "o-")
        target_tip = runs[row["stem"]].get("target_tip")
        if target_tip is not None:
            ax.plot(target_tip[0], target_tip[1], "r*", markersize=16, label="target")
            ax.legend(fontsize=8)
        ax.set_aspect("equal")
        ax.set_title(f"{label}: {row['stem']}")
        ax.grid(alpha=0.3)
    fig.savefig(figures_dir / "shape_best_worst.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------

def main():
    global SUCCESS_THRESHOLD_PERCENT
    parser = argparse.ArgumentParser(description="Analyze a run_sweep.py sweep from its JSONL logs.")
    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS_DIR,
                         help="Directory to scan for <stem>.jsonl files (default: logs/).")
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR,
                         help="Where to save figures (default: experiments/figures/).")
    parser.add_argument("--out", type=Path, default=DEFAULT_METRICS_CSV,
                         help="Where to save the per-run metrics CSV.")
    parser.add_argument("--success-threshold", type=float, default=SUCCESS_THRESHOLD_PERCENT,
                         help="error_percent below this counts as solved (default: 5.0).")
    args = parser.parse_args()
    SUCCESS_THRESHOLD_PERCENT = args.success_threshold

    stems = discover_stems(args.logs_dir)
    print(f"Found {len(stems)} run(s) in {args.logs_dir}")

    rows = []
    runs = {}
    skipped = []
    for stem in stems:
        jsonl_path = args.logs_dir / f"{stem}.jsonl"
        if not jsonl_path.exists():
            skipped.append(stem)
            continue

        run = load_run(jsonl_path)
        target = run.get("target_shape")
        parsed = parse_stem(stem, target) if target else None
        if parsed is None:
            skipped.append(stem)
            continue

        model, repetition = parsed
        metrics = compute_metrics(run)
        runs[stem] = run
        rows.append({"stem": stem, "model": model, "target": target, "repetition": repetition, **metrics})

    if skipped:
        print(f"Skipped {len(skipped)} stem(s) (missing jsonl or unrecognized name):")
        for s in skipped:
            print(f"  {s}")

    if not rows:
        print("No runs found — nothing to analyze.")
        return

    write_csv(rows, args.out)
    print(f"Saved per-run metrics: {args.out}  ({len(rows)} rows)")

    by_model_task, by_model = build_summary_tables(rows)
    write_csv(by_model_task, args.out.parent / "summary_by_model_task.csv")
    write_csv(by_model, args.out.parent / "summary_by_model.csv")
    print(f"Saved summary_by_model_task.csv and summary_by_model.csv in {args.out.parent}")

    make_figures(rows, runs, args.figures_dir)
    print(f"Saved figures to {args.figures_dir}")


if __name__ == "__main__":
    main()
