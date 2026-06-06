# ---------------------------------------------------------------------------
# src/benchmarks/visualization.py
# ---------------------------------------------------------------------------
# Shared visualization for benchmark shapes.
#
# plot_shape   — single XY panel for one shape (used in per-shape main())
# plot_grid    — grid of all registered benchmarks (used in test_all_shapes.py)
# ---------------------------------------------------------------------------

import math
import matplotlib.pyplot as plt


def plot_shape(
    nodes: list,
    name: str,
    error_percent: float = 0.0,
    wall_H: float = None,
    ax=None,
    show: bool = True,
) -> None:
    """
    Plot one beam shape: nodes connected as a curve.

    Args:
        nodes:         List of (x, y) positions.
        name:          Shape name shown in the title.
        error_percent: Loss value shown in the title (0.0 = perfect).
        wall_H:        If given, draw a horizontal wall line at this y value.
        ax:            Existing matplotlib Axes; creates a new figure if None.
        show:          Call plt.show() when done (set False when embedding in a grid).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 5))

    xs = [p[0] for p in nodes]
    ys = [p[1] for p in nodes]

    ax.plot(xs, ys, "bo-", linewidth=2, markersize=6)
    ax.plot(xs[0],  ys[0],  "gs", markersize=10, label="held end (node 0)")
    ax.plot(xs[-1], ys[-1], "r^", markersize=10, label="tip")

    if wall_H is not None:
        x_min, x_max = min(xs) - 0.01, max(xs) + 0.01
        ax.axhline(wall_H, color="gray", linewidth=1.5, linestyle="--", label=f"wall y={wall_H:.3f}")

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"{name}   {error_percent:.1f}% error")
    ax.legend(fontsize=7)

    if show:
        plt.tight_layout()
        plt.show()


def plot_grid(benchmarks: list, n_cols: int = 2, show: bool = True) -> None:
    """
    Plot all benchmarks in a grid. Each benchmark is represented by its
    reference shape (ideal at 0% error).

    Args:
        benchmarks: List of Benchmark objects (already have reference_fn called).
        n_cols:     Number of columns in the grid.
        show:       Call plt.show() at the end.
    """
    n = len(benchmarks)
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))

    # Flatten axes even for a single row
    axes_flat = axes.flatten() if n > 1 else [axes]

    for i, bm in enumerate(benchmarks):
        nodes = bm.reference_fn()
        plot_shape(nodes, name=bm.name, error_percent=0.0, ax=axes_flat[i], show=False)

    # Hide unused axes
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Benchmark shapes — ideal reference (0% error)", fontsize=13)
    plt.tight_layout()

    if show:
        plt.show()
