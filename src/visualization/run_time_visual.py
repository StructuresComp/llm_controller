"""
Runtime visualization for beam/rod configurations and benchmark shapes.

    plot_shape       — single XY panel for one shape (reference or actual)
    plot_comparison  — reference vs actual overlaid, with loss annotation
    plot_grid        — grid of all registered benchmarks at ideal (0% error)
"""

import math
import matplotlib.pyplot as plt


def _draw_walls(ax, walls: list) -> None:
    """Draw all walls on axes ax."""
    for w in walls:
        P0, P1 = w["P0"], w["P1"]
        ax.plot([P0[0], P1[0]], [P0[1], P1[1]],
                color="gray", linewidth=1.5, linestyle="--")


def plot_shape(
    nodes: list,
    name: str = "",
    error_percent: float = 0.0,
    walls: list = None,
    ax=None,
    show: bool = True,
) -> None:
    """Plot one beam shape: nodes connected as a curve."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 5))

    xs = [p[0] for p in nodes]
    ys = [p[1] for p in nodes]

    ax.plot(xs, ys, "bo-", linewidth=2, markersize=6)
    ax.plot(xs[0],  ys[0],  "gs", markersize=10, label="node 0 (held)")
    ax.plot(xs[-1], ys[-1], "r^", markersize=10, label="tip")

    if walls:
        _draw_walls(ax, walls)

    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"{name}   {error_percent:.1f}% error" if name else f"{error_percent:.1f}% error")
    ax.legend(fontsize=7)

    if show:
        plt.tight_layout()
        plt.show()


def plot_comparison(
    ref_nodes: list,
    actual_nodes: list,
    result: dict,
    name: str = "",
    walls: list = None,
    axes=None,
    show: bool = True,
) -> None:
    """
    Overlay reference shape (dashed red) and actual configuration (solid blue).

    result is the dict returned by loss_fn — needs at least error_percent.
    """
    if axes is None:
        _, axes = plt.subplots(figsize=(4, 5))

    rx = [p[0] for p in ref_nodes]
    ry = [p[1] for p in ref_nodes]
    bx = [p[0] for p in actual_nodes]
    by = [p[1] for p in actual_nodes]

    axes.plot(rx, ry, "r--o", linewidth=2, markersize=5, label="reference (ideal)")
    axes.plot(bx, by, "bo-",  linewidth=2, markersize=6, label="actual")
    axes.plot(bx[0],  by[0],  "gs", markersize=10)
    axes.plot(bx[-1], by[-1], "r^", markersize=10)

    if walls:
        _draw_walls(axes, walls)

    err   = result.get("error_percent", float("nan"))
    title = f"{name}   {err:.1f}% error" if name else f"{err:.1f}% error"
    axes.set_aspect("equal")
    axes.set_xlabel("x (m)")
    axes.set_ylabel("y (m)")
    axes.set_title(title)
    axes.legend(fontsize=7)

    if show:
        plt.tight_layout()
        plt.show()


def plot_grid(benchmarks: list, n_cols: int = 2, show: bool = True) -> None:
    """Plot all benchmarks in a grid. Each panel shows the ideal reference shape."""
    n = len(benchmarks)
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes_flat = axes.flatten() if n > 1 else [axes]

    for i, bm in enumerate(benchmarks):
        nodes = bm.reference_fn()
        if nodes is None:
            axes_flat[i].text(0.5, 0.5, f"{bm.name}\n(no reference shape)",
                              ha="center", va="center", transform=axes_flat[i].transAxes)
            axes_flat[i].set_title(bm.name)
            continue
        plot_shape(nodes, name=bm.name, error_percent=0.0, ax=axes_flat[i], show=False)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Benchmark shapes — ideal reference (0% error)", fontsize=13)
    plt.tight_layout()
    if show:
        plt.show()
