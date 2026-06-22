"""
Difficulty-scaled C-shape benchmarks: C_30, C_60, C_90, C_120, C_150.

Each case targets a constant-curvature arc (C-shape) with a specific total
turning angle. Loss is a fixed-target θ-profile RMS: the ideal arc is pinned
to the target angle, so a beam bent to the wrong amplitude scores high error
even if its shape is perfectly arc-like.

This provides a scalar-parameterised difficulty axis for cross-model comparison.
Each model's capability threshold — the turning angle at which improvement_stability
drops and bad_step_rate rises — is a quantitative behavioral fingerprint.

No walls: difficulty comes purely from the required turning angle, not wall
contact geometry. The tip is clamped (controlled_nodes = [0,1,nv-2,nv-1]).

Run standalone:
    python -m src.benchmarks.cases.beam_C_scaled
"""

import numpy as np

from src.benchmarks import BenchmarkCase, register
from src.benchmarks.profile_fitting import build_ideal_from_edge_profile, profile_loss
from src.benchmarks.loss_beam.profile_C import profile_c, make_fit_c
from src.visualization.run_time_visual import plot_shape


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _make_c_scaled(total_turn_deg: int) -> BenchmarkCase:
    """Build and return a BenchmarkCase for a C-arc of `total_turn_deg` degrees."""
    total_turn_rad = np.radians(total_turn_deg)
    _fit = make_fit_c(total_turn_rad)

    class _CScaled(BenchmarkCase):
        name               = f"C_{total_turn_deg}"
        physics_type       = "beam"
        walls              = []    # free-space: difficulty is purely kinematic
        target_description = (
            f"A smooth C-arc with a total turning angle of exactly {total_turn_deg}°. "
            f"Constant curvature throughout, all nodes bending in the same direction. "
            f"No walls — bend the beam in free space to the required angle."
        )
        success_threshold  = 0.05

        def reference_fn(self, x0: float = 0.0, y0: float = 0.0, theta: float = 0.0) -> list:
            return build_ideal_from_edge_profile(
                profile_c, n=9, arc_length=0.10,
                x0=x0, y0=y0, theta=theta, total_turn=total_turn_rad,
            )

        def loss_fn(self, nodes: list, ps=None, q_ref=None) -> dict:
            return profile_loss(nodes, profile_c, _fit)

    _CScaled.__name__ = f"CShape{total_turn_deg}"
    return _CScaled()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_DEGREES = [30, 60, 90, 120, 150]

for _deg in _DEGREES:
    register(_make_c_scaled(_deg))


# ---------------------------------------------------------------------------
# STANDALONE VISUALISER
# Plots all five reference shapes side-by-side for a visual sanity check.
# ---------------------------------------------------------------------------

def main():
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(_DEGREES), figsize=(3 * len(_DEGREES), 4))
    for ax, deg in zip(axes, _DEGREES):
        from src.benchmarks import get_benchmark
        case = get_benchmark(f"C_{deg}")
        ref  = case.reference_fn()
        xs   = [p[0] for p in ref]
        ys   = [p[1] for p in ref]
        ax.plot(xs, ys, "o-", markersize=4)
        ax.set_title(f"C_{deg}°")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Scaled C-arc reference shapes (free-space)", fontweight="bold")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
