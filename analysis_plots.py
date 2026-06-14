# ---------------------------------------------------------------------------
# analysis_plots.py
# ---------------------------------------------------------------------------
# Post-run trajectory analysis from JSONL streaming files.
#
# Reads run_*.jsonl (trajectory snapshots) and generates analysis plots:
#   - Error curve (static plot)
#   - Trajectory animation (all snapshots over time)
#   - Iteration shapes animation (final shape per iteration)
#   - Detailed iteration view (all snapshots within one iteration)
#
# Usage:
#   python3 analysis_plots.py                    # uses latest run
#   python3 analysis_plots.py --run run_1234     # specific run
#   python3 analysis_plots.py --iter 5           # detail view of iteration 5
#   python3 analysis_plots.py --all --output-dir viz/  # save all plots
# ---------------------------------------------------------------------------

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.core.paths import LOGS_DIR


# ---------------------------------------------------------------------------
# 1. LOADING & PARSING
# ---------------------------------------------------------------------------

def load_trajectory(run_name: str) -> dict:
    """
    Load JSONL file and return structured trajectory.

    Args:
        run_name: stem of JSONL file (e.g., "run_20260613_12345")

    Returns:
        dict with keys:
          - "iterations": list of iteration dicts, each with:
              - "iteration": int
              - "bc": {x0, y0, theta}
              - "snapshots": list of {q, error_percent, timestamp}
          - "metadata": {start_time, end_time, snapshot_count}
    """
    jsonl_path = LOGS_DIR / f"{run_name}.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {jsonl_path}")

    snapshots = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                snapshots.append(json.loads(line))

    if not snapshots:
        raise ValueError(f"Trajectory file is empty: {jsonl_path}")

    # Group by iteration
    iterations_dict = {}
    for snap in snapshots:
        it = snap["iteration"]
        if it not in iterations_dict:
            iterations_dict[it] = {
                "iteration": it,
                "bc": snap["bc"],
                "snapshots": [],
            }
        snap_data = {
            "iteration": snap["iteration"],
            "bc": snap["bc"],
            "q": snap["q"],
            "error_percent": snap["error_percent"],
            "timestamp": snap["timestamp"],
        }
        # Preserve orchestrator_prompt if present (only on first snapshot of iteration)
        if "orchestrator_prompt" in snap:
            snap_data["orchestrator_prompt"] = snap["orchestrator_prompt"]
        iterations_dict[it]["snapshots"].append(snap_data)

    iterations = [iterations_dict[it] for it in sorted(iterations_dict.keys())]

    return {
        "iterations": iterations,
        "metadata": {
            "snapshot_count": len(snapshots),
            "iteration_count": len(iterations),
            "jsonl_path": str(jsonl_path),
        },
    }


def extract_nodes(q: list, nv: int) -> np.ndarray:
    """Extract (x, y) positions for all nodes from DOF vector q."""
    nodes = []
    for i in range(nv):
        nodes.append([q[2 * i], q[2 * i + 1]])
    return np.array(nodes)


def infer_nv(q_length: int) -> int:
    """Infer number of vertices from DOF vector length."""
    return q_length // 2


# ---------------------------------------------------------------------------
# 2. PLOTTING FUNCTIONS
# ---------------------------------------------------------------------------

def plot_error_curve(trajectory: dict, save_path: Path = None):
    """
    Plot error_percent vs iteration.

    Args:
        trajectory: loaded trajectory dict
        save_path: optional file path to save figure (if None, figure stays open)
    """
    iterations = trajectory["iterations"]
    iter_nums = [it["iteration"] for it in iterations]
    errors = [it["snapshots"][-1]["error_percent"] for it in iterations]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(iter_nums, errors, "o-", linewidth=2, markersize=6, color="steelblue")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Error (%)", fontsize=12)
    ax.set_title("Shape Error Over Time", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
        print(f"Saved error curve: {save_path}")
        plt.close(fig)


def plot_trajectory_animation(trajectory: dict, output_dir: Path = None, fps: int = 10):
    """
    Animate all snapshots over time. Either displays live or saves frames to disk.

    Args:
        trajectory: loaded trajectory dict
        output_dir: if provided, save frames as PNG files; otherwise display live
        fps: frames per second (controls pause duration for display)
    """
    iterations = trajectory["iterations"]
    all_snapshots = []
    for it in iterations:
        all_snapshots.extend(it["snapshots"])

    if not all_snapshots:
        print("No snapshots to animate")
        return

    nv = infer_nv(len(all_snapshots[0]["q"]))

    # Compute global axis limits once
    all_nodes = []
    for snap in all_snapshots:
        all_nodes.append(extract_nodes(snap["q"], nv))
    all_nodes = np.vstack(all_nodes)

    x_min, x_max = all_nodes[:, 0].min() - 0.02, all_nodes[:, 0].max() + 0.02
    y_min, y_max = all_nodes[:, 1].min() - 0.02, all_nodes[:, 1].max() + 0.02

    if output_dir:
        frame_dir = output_dir / "trajectory_frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving {len(all_snapshots)} trajectory frames to {frame_dir}/")
    else:
        print(f"Animation: {len(all_snapshots)} frames, {nv} nodes per frame")
        print(f"  x range: [{x_min:.4f}, {x_max:.4f}]")
        print(f"  y range: [{y_min:.4f}, {y_max:.4f}]")
        print(f"  Playing at {fps} fps... (close window to stop)")

    pause_time = 1.0 / fps if not output_dir else 0

    # Plot each frame
    plt.figure(figsize=(8, 10))
    for frame_idx, snap in enumerate(all_snapshots):
        plt.clf()
        ax = plt.gca()

        nodes = extract_nodes(snap["q"], nv)
        ax.plot(nodes[:, 0], nodes[:, 1], "o-", linewidth=2, markersize=5, color="steelblue")

        # Build title with BC and error
        it_num = snap.get("iteration", "?")
        bc = snap.get("bc", {})
        err = snap.get("error_percent", 0)
        x0 = bc.get("x0", 0)
        y0 = bc.get("y0", 0)
        theta = bc.get("theta", 0)
        # Show orchestrator prompt on first frame of iteration
        prompt_indicator = ""
        if snap.get("orchestrator_prompt"):
            prompt_indicator = "  [📋 PROMPT AVAILABLE]"

        title = (
            f"Frame {frame_idx:05d}/{len(all_snapshots)-1:05d}  |  "
            f"Iter {it_num}  |  BC: x₀={x0:+.4f} y₀={y0:+.4f} θ={theta:+.4f}  |  error={err:.2f}%{prompt_indicator}"
        )
        ax.set_title(title, fontsize=10)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(True, alpha=0.2)

        if output_dir:
            frame_file = frame_dir / f"{frame_idx:05d}.png"
            plt.savefig(frame_file, dpi=100, bbox_inches="tight")
        else:
            plt.pause(pause_time)


def plot_iteration_shapes(trajectory: dict, output_dir: Path = None, fps: int = 2):
    """
    Animate final shape from each iteration (one per frame).

    Args:
        trajectory: loaded trajectory dict
        output_dir: if provided, save frames as PNG files; otherwise display live
        fps: frames per second (default 2 = 0.5s per iteration)
    """
    iterations = trajectory["iterations"]

    if not iterations:
        print("No iterations to animate")
        return

    # Compute global axis limits
    nv = None
    all_nodes_list = []
    for it in iterations:
        final_snap = it["snapshots"][-1]
        q = final_snap["q"]
        if nv is None:
            nv = infer_nv(len(q))
        all_nodes_list.append(extract_nodes(q, nv))

    all_nodes = np.vstack(all_nodes_list)
    x_min, x_max = all_nodes[:, 0].min() - 0.02, all_nodes[:, 0].max() + 0.02
    y_min, y_max = all_nodes[:, 1].min() - 0.02, all_nodes[:, 1].max() + 0.02

    pause_time = 1.0 / fps if not output_dir else 0

    if output_dir:
        frame_dir = output_dir / "orchestrator_iterations"
        frame_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving {len(iterations)} orchestrator iteration shapes to {frame_dir}/")
    else:
        print(f"Iteration shapes: {len(iterations)} iterations")
        print(f"  Playing at {fps} fps... (close window to stop)")

    plt.figure(figsize=(10, 8))
    for it in iterations:
        plt.clf()
        ax = plt.gca()

        final_snap = it["snapshots"][-1]
        nodes = extract_nodes(final_snap["q"], nv)

        ax.plot(nodes[:, 0], nodes[:, 1], "o-", linewidth=2, markersize=6, color="steelblue")

        # Title with iteration, BC, and error
        it_num = it["iteration"]
        bc = it["bc"]
        error = final_snap["error_percent"]
        title = (
            f"Iteration {it_num} — Final Shape\n"
            f"BC: x₀={bc['x0']:+.4f} y₀={bc['y0']:+.4f} θ={bc['theta']:+.4f}  |  error={error:.2f}%"
        )
        ax.set_title(title, fontsize=11)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(True, alpha=0.2)

        if output_dir:
            frame_file = frame_dir / f"{it_num:05d}.png"
            plt.savefig(frame_file, dpi=100, bbox_inches="tight")
        else:
            plt.pause(pause_time)


def plot_single_iteration(trajectory: dict, iteration_num: int, save_path: Path = None):
    """
    Plot all snapshots for one iteration in a single figure.

    Args:
        trajectory: loaded trajectory dict
        iteration_num: which iteration to plot (1-indexed)
        save_path: optional file path to save figure
    """
    iterations = trajectory["iterations"]
    it = next((x for x in iterations if x["iteration"] == iteration_num), None)

    if it is None:
        print(f"Iteration {iteration_num} not found")
        return

    snapshots = it["snapshots"]
    nv = infer_nv(len(snapshots[0]["q"]))

    # Determine limits
    all_nodes = [extract_nodes(s["q"], nv) for s in snapshots]
    all_x = np.concatenate([n[:, 0] for n in all_nodes])
    all_y = np.concatenate([n[:, 1] for n in all_nodes])
    x_min, x_max = all_x.min() - 0.02, all_x.max() + 0.02
    y_min, y_max = all_y.min() - 0.02, all_y.max() + 0.02

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot all snapshots with fading color
    for idx, nodes in enumerate(all_nodes):
        alpha = 0.2 + 0.8 * (idx / len(all_nodes))  # fade in from light to dark
        ax.plot(
            nodes[:, 0],
            nodes[:, 1],
            "o-",
            linewidth=1.5,
            markersize=3,
            alpha=alpha,
            color="steelblue",
        )

    # Highlight final shape
    final_nodes = all_nodes[-1]
    ax.plot(final_nodes[:, 0], final_nodes[:, 1], "o-", linewidth=3, markersize=6, color="darkred", label="Final")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=12)
    ax.set_ylabel("y (m)", fontsize=12)

    bc = it["bc"]
    error = snapshots[-1]["error_percent"]
    ax.set_title(
        f"Iteration {iteration_num} Trajectory\n"
        f"BC: x₀={bc['x0']:+.4f}, y₀={bc['y0']:+.4f}, θ={bc['theta']:+.4f}\n"
        f"Final error={error:.2f}%",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.2)
    ax.legend()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
        print(f"Saved iteration {iteration_num}: {save_path}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# 3. MAIN & CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize trajectory from streaming JSONL file."
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        metavar="RUN_NAME",
        help="Stem of JSONL file (e.g., run_20260613_12345). If omitted, uses latest run.",
    )
    parser.add_argument(
        "--error-curve",
        action="store_true",
        help="Plot error vs iteration",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Animate all snapshots",
    )
    parser.add_argument(
        "--shapes",
        action="store_true",
        help="Grid of final shapes per iteration",
    )
    parser.add_argument(
        "--iter",
        type=int,
        metavar="N",
        help="Plot all snapshots in iteration N",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all plots",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Save plots to directory instead of showing",
    )

    args = parser.parse_args()

    # If no --run specified, find the latest
    if not args.run:
        import sys
        jsonl_files = sorted(LOGS_DIR.glob("run_*.jsonl"))
        if not jsonl_files:
            print("Error: no trajectory files found in logs/", file=sys.stderr)
            sys.exit(1)
        args.run = jsonl_files[-1].stem
        print(f"Using latest run: {args.run}\n")

    # Load trajectory
    print(f"Loading trajectory from {args.run}...")
    trajectory = load_trajectory(args.run)
    meta = trajectory["metadata"]
    print(f"  Snapshots: {meta['snapshot_count']}")
    print(f"  Iterations: {meta['iteration_count']}")
    print(f"  File: {meta['jsonl_path']}")

    # If no specific plot requested, show all by default
    if not (args.error_curve or args.animate or args.shapes or args.iter or args.all):
        args.all = True

    output_dir = Path(args.output_dir) if args.output_dir else None

    # Generate plots
    if args.error_curve or args.all:
        save_path = output_dir / f"{args.run}_error.png" if output_dir else None
        plot_error_curve(trajectory, save_path)

    # Animations display live or save as frame sequences
    if args.animate or args.all:
        plot_trajectory_animation(trajectory, output_dir)

    if args.shapes or args.all:
        plot_iteration_shapes(trajectory, output_dir)

    if args.iter:
        save_path = output_dir / f"{args.run}_iter{args.iter}.png" if output_dir else None
        plot_single_iteration(trajectory, args.iter, save_path)

    # Show all figures at once if not saving to disk
    if not output_dir:
        plt.show()

    print("Done.")


# ---------------------------------------------------------------------------
# 4. ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
