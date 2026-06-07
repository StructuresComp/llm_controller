"""
Runtime visualization for beam/rod configurations and benchmark shapes.

    plot_shape       — single XY panel for one shape (reference or actual)
    plot_comparison  — reference vs actual overlaid, with loss annotation
    plot_grid        — grid of all registered benchmarks at ideal (0% error)
"""

import math
import matplotlib.pyplot as plt


def plot_shape(
    nodes: list,
    name: str = "",
    error_percent: float = 0.0,
    wall_H: float = None,
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

    if wall_H is not None:
        ax.axhline(wall_H, color="gray", linewidth=1.5, linestyle="--",
                   label=f"wall y={wall_H:.3f}")

    ax.set_aspect("equal")
    ax.invert_yaxis()
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
    wall_H: float = None,
    axes=None,
    show: bool = True,
) -> None:
    """
    Overlay reference shape (dashed red) and actual configuration (solid blue).

    result is the dict returned by loss_fn — needs at least error_percent
    and diagnostics.
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

    if wall_H is not None:
        axes.axhline(wall_H, color="gray", linewidth=1.5, linestyle="--",
                     label=f"wall y={wall_H:.3f}")

    err   = result.get("error_percent", float("nan"))
    diag  = result.get("diagnostics", "")
    title = f"{name}   {err:.1f}% error" if name else f"{err:.1f}% error"
    axes.set_aspect("equal")
    axes.invert_yaxis()
    axes.set_xlabel("x (m)")
    axes.set_ylabel("y (m)")
    axes.set_title(title)
    axes.legend(fontsize=7)

    if diag:
        axes.text(0.02, 0.02, diag, transform=axes.transAxes,
                  fontsize=7, verticalalignment="bottom", color="gray")

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
        plot_shape(nodes, name=bm.name, error_percent=0.0, ax=axes_flat[i], show=False)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Benchmark shapes — ideal reference (0% error)", fontsize=13)
    plt.tight_layout()
    if show:
        plt.show()
