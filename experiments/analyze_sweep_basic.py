# ---------------------------------------------------------------------------
# experiments/analyze_sweep_basic.py
# ---------------------------------------------------------------------------
# Minimal companion to analyze_sweep.py -- three plots only, no metrics CSV,
# no summary tables:
#   1. Error vs. iteration, one line per model (per target).
#   2. Control signal (x0, y0, theta) vs. iteration, one line per model (per target).
#   3. Iterations-to-threshold, one bar per model (mean across repetitions).
#      A run that never reaches the threshold uses a sentinel value (1.2x the
#      max iteration count actually seen in the data, so it's always visibly
#      off the chart) and the model's bar is hatched + labeled "DNF".
#
# Reuses discover_stems/parse_stem/load_run from analyze_sweep.py -- same file
# discovery and header/iteration parsing, nothing duplicated. Still fully
# standalone: no src/ imports, only stdlib + numpy + matplotlib.
#
# Usage:
#   python3 experiments/analyze_sweep_basic.py                    # plot every model
#   python3 experiments/analyze_sweep_basic.py --provider ollama  # ollama models only
#   python3 experiments/analyze_sweep_basic.py --logs-dir logs/ --success-threshold 5.0
# ---------------------------------------------------------------------------

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

EXPERIMENTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENTS_DIR))
from analyze_sweep import discover_stems, parse_stem, load_run  # noqa: E402

DEFAULT_LOGS_DIR    = EXPERIMENTS_DIR.parent / "logs"
DEFAULT_FIGURES_DIR = EXPERIMENTS_DIR / "figures_basic"

SUCCESS_THRESHOLD_PERCENT = 5.0
_KNOWN_PROVIDERS = ("anthropic", "openai", "gemini", "ollama")

# ---------------------------------------------------------------------------
# PLOT STYLE — journal single-column figure convention (matches the group's
# house style: LaTeX/serif text, vector PDF+SVG output, boxed axes).
# ---------------------------------------------------------------------------
plt.rcParams["text.usetex"]  = True
plt.rcParams["font.family"]  = "serif"
plt.rcParams["font.size"]    = 10

PWIDTH, PHEIGHT = 4, 3  # inches — base single-panel size


def _save_vector(fig, path: Path, extra_artists=()) -> None:
    """Save a figure as editable vector PDF + SVG (no PNG/dpi -- matches house style)."""
    kwargs = {"bbox_inches": "tight"}
    if extra_artists:
        kwargs["bbox_extra_artists"] = extra_artists
    fig.savefig(path.with_suffix(".pdf"), **kwargs)
    fig.savefig(path.with_suffix(".svg"), **kwargs)


def _texsafe(s: str) -> str:
    """Escape LaTeX-special characters so arbitrary model names never break
    usetex rendering (model names come from MODELS in run_sweep.py, user-edited,
    not guaranteed LaTeX-safe -- e.g. a future model name containing '_' or '%')."""
    for ch in ("\\", "_", "%", "#", "&", "{", "}"):
        s = s.replace(ch, "\\" + ch)
    return s


def _infer_provider(model: str) -> str:
    """
    Infer provider from a model name, same convention documented in config.py:
    claude-* -> anthropic, gpt-/o1-/o3-/o4-* -> openai, gemini-* -> gemini,
    else -> ollama. Provider isn't stored anywhere in the jsonl data (by design --
    see MEMORY/session notes), only the model name is, so this is inferred, not looked up.
    """
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith(("gpt-", "o1-", "o3", "o4-")):
        return "openai"
    if model.startswith("gemini-"):
        return "gemini"
    return "ollama"


# ---------------------------------------------------------------------------
# 1. DISCOVERY
# ---------------------------------------------------------------------------

def load_all_runs(logs_dir: Path) -> tuple[list[dict], dict[str, dict]]:
    """Same discovery/parsing as analyze_sweep.py, without the metrics computation."""
    rows, runs, skipped = [], {}, []
    for stem in discover_stems(logs_dir):
        run = load_run(logs_dir / f"{stem}.jsonl")
        target = run.get("target_shape")
        parsed = parse_stem(stem, target) if target else None
        if parsed is None:
            skipped.append(stem)
            continue
        model, repetition = parsed
        runs[stem] = run
        rows.append({"stem": stem, "model": model, "target": target, "repetition": repetition})

    if skipped:
        print(f"Skipped {len(skipped)} stem(s) (missing jsonl or unrecognized name): {skipped}")
    return rows, runs


# ---------------------------------------------------------------------------
# 2. ERROR VS ITERATION
# ---------------------------------------------------------------------------

def plot_error_vs_iteration(rows: list[dict], runs: dict[str, dict], figures_dir: Path,
                             suffix: str = "") -> None:
    """One figure per target: error_percent vs iteration, one line per model (repetition 1)."""
    models  = sorted({r["model"] for r in rows})
    targets = sorted({r["target"] for r in rows})

    for target in targets:
        fig, ax = plt.subplots(figsize=(PWIDTH, PHEIGHT))
        for model in models:
            match = next((r for r in rows if r["model"] == model
                          and r["target"] == target and r["repetition"] == 1), None)
            if match is None:
                continue
            e = [it["error_percent"] for it in runs[match["stem"]]["iterations"]]
            ax.plot(range(1, len(e) + 1), e, "o-", markersize=3, label=_texsafe(model))
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Error (\\%)")
        ax.set_title(f"Error vs iteration --- {target}")
        ax.set_box_aspect(PHEIGHT / PWIDTH)
        # Legend outside the axes (below), multi-column -- keeps it readable with many models.
        leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25),
                         ncol=3, fontsize=6, frameon=True)
        leg.get_frame().set_edgecolor("black")
        ax.grid(alpha=0.3)
        _save_vector(fig, figures_dir / f"error_vs_iteration_{target}{suffix}.pdf", extra_artists=(leg,))
        plt.close(fig)


# ---------------------------------------------------------------------------
# 3. CONTROL SIGNAL VS ITERATION
# ---------------------------------------------------------------------------

def plot_control_signal(rows: list[dict], runs: dict[str, dict], figures_dir: Path,
                         suffix: str = "") -> None:
    """One figure per target: x0, y0, theta vs iteration (3 stacked subplots), one line per model."""
    models  = sorted({r["model"] for r in rows})
    targets = sorted({r["target"] for r in rows})

    for target in targets:
        # 3 stacked panels -- width matches the single-panel convention, height
        # scaled per panel (PHEIGHT each) rather than forced into one PHEIGHT total.
        # Panel height increased vs. the single-panel case (PHEIGHT*3.2 total, not
        # *2.2): the y-axis label strings are long, and rotated 90deg text needs
        # roughly its own character-length in vertical panel space to avoid
        # overflowing into the panel above -- a layout-engine setting can't fix
        # that on its own, since the text is physically longer than a short panel.
        fig, axes = plt.subplots(3, 1, figsize=(PWIDTH, PHEIGHT * 3.2), sharex=True,
                                  constrained_layout=True)
        for model in models:
            match = next((r for r in rows if r["model"] == model
                          and r["target"] == target and r["repetition"] == 1), None)
            if match is None:
                continue
            iterations = runs[match["stem"]]["iterations"]
            iter_nums = range(1, len(iterations) + 1)
            for ax, key in zip(axes, ("x0", "y0", "theta")):
                vals = [it["bc"][key] for it in iterations]
                ax.plot(iter_nums, vals, "o-", markersize=3, label=_texsafe(model))
        ylabels = (
            "Horizontal Position Control, $x_0$ [m]",
            "Vertical Position Control, $y_0$ [m]",
            "Angle Control, $\\theta$ [radian]",
        )
        for ax, ylabel in zip(axes, ylabels):
            ax.set_ylabel(ylabel, fontsize=8)
            ax.grid(alpha=0.3)
        axes[0].set_title(f"Control signal vs iteration --- {target}")
        # Legend above the top panel -- bottom is already the shared x-axis label.
        leg = axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.35),
                              ncol=3, fontsize=6, frameon=True)
        leg.get_frame().set_edgecolor("black")
        axes[-1].set_xlabel("Iteration")
        _save_vector(fig, figures_dir / f"control_signal_{target}{suffix}.pdf", extra_artists=(leg,))
        plt.close(fig)


# ---------------------------------------------------------------------------
# 4. ITERATIONS TO THRESHOLD
# ---------------------------------------------------------------------------

def _iterations_to_threshold(iterations: list[dict], threshold: float) -> int | None:
    """First iteration index (1-based) where error_percent < threshold; None if never."""
    for i, it in enumerate(iterations, start=1):
        if it["error_percent"] < threshold:
            return i
    return None


def plot_iterations_to_threshold(rows: list[dict], runs: dict[str, dict],
                                  figures_dir: Path, threshold: float, suffix: str = "") -> None:
    """One bar per model: mean iterations-to-threshold across the repetitions that
    actually converged. A model with zero converged repetitions gets a zero-height
    bar labeled "NO CONVERGENCE" instead of an inflated sentinel value; a model
    with some (but not all) converged repetitions shows the mean of just those."""
    models = sorted({r["model"] for r in rows})

    bar_values, no_convergence = [], []
    for model in models:
        converged = [
            n for r in rows if r["model"] == model
            for n in [_iterations_to_threshold(runs[r["stem"]]["iterations"], threshold)]
            if n is not None
        ]
        bar_values.append(float(np.mean(converged)) if converged else 0.0)
        no_convergence.append(not converged)

    # Unlike the legend-based plots, model names sit directly on the x-axis here and
    # can't be moved "outside" -- widen the figure instead so labels stay readable
    # (deliberate deviation from the fixed PWIDTH for this one plot).
    width = max(PWIDTH, 0.35 * len(models))
    fig, ax = plt.subplots(figsize=(width, PHEIGHT))
    x = np.arange(len(models))
    colors = ["indianred" if nc else "steelblue" for nc in no_convergence]
    ax.bar(x, bar_values, color=colors)
    label_y = 0.02 * (max(bar_values, default=1) or 1)  # small offset above the baseline
    for xi, nc in zip(x, no_convergence):
        if nc:
            ax.text(xi, label_y, "NO CONVERGENCE", ha="center", va="bottom",
                     fontsize=6, fontweight="bold", rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([_texsafe(m) for m in models], rotation=30, ha="right")
    ax.set_ylabel("Iterations to reach threshold")
    ax.set_title(f"Iterations to reach {threshold:g}\\% error (mean across repetitions)")
    ax.grid(alpha=0.3, axis="y")
    _save_vector(fig, figures_dir / f"iterations_to_threshold{suffix}.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Minimal 3-plot analysis of a run_sweep.py sweep.")
    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS_DIR,
                         help="Directory to scan for <stem>.jsonl files (default: logs/).")
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR,
                         help="Where to save figures (default: experiments/figures_basic/).")
    parser.add_argument("--success-threshold", type=float, default=SUCCESS_THRESHOLD_PERCENT,
                         help="error_percent below this counts as reaching the target (default: 5.0).")
    parser.add_argument("--provider", choices=_KNOWN_PROVIDERS, default=None,
                         help="Only plot models inferred to belong to this provider "
                              "(default: no filter, plot every model).")
    args = parser.parse_args()

    rows, runs = load_all_runs(args.logs_dir)
    print(f"Found {len(rows)} run(s) in {args.logs_dir}")
    if not rows:
        print("No runs found — nothing to analyze.")
        return

    if args.provider is not None:
        rows = [r for r in rows if _infer_provider(r["model"]) == args.provider]
        print(f"Filtered to provider={args.provider!r}: {len(rows)} run(s)")
        if not rows:
            print("No runs match that provider — nothing to plot.")
            return

    # Filenames get a _<provider> suffix only when filtered -- otherwise different
    # --provider runs (e.g. openai, gemini) would silently overwrite each other's
    # figures, since target alone doesn't distinguish them.
    suffix = f"_{args.provider}" if args.provider else ""

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    plot_error_vs_iteration(rows, runs, args.figures_dir, suffix=suffix)
    plot_control_signal(rows, runs, args.figures_dir, suffix=suffix)
    plot_iterations_to_threshold(rows, runs, args.figures_dir, args.success_threshold, suffix=suffix)
    print(f"Saved figures to {args.figures_dir}")


if __name__ == "__main__":
    main()
