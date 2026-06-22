"""
C-shape recovery benchmark: identical target to beam_C, but the beam starts
in a deliberately bad configuration (bent leftward, opposite the target).

The initial_bc attribute instructs cli.py to pre-run the physics to
(-0.03, -0.01, -0.8 rad) before the LLM loop starts.  The beam is already
bent in the wrong direction — tip at approximately (-0.08, -0.10) — so the
LLM must first undo the bad state and then drive toward the C-target.

Purpose: measure the recovery behavioral metric under controlled conditions.
Compare against beam_C runs (same target, straight-down start) to isolate
the cost of recovery vs. normal convergence.

Run standalone:
    python -m src.benchmarks.cases.beam_C_recover
"""

import numpy as np

from src.benchmarks import BenchmarkCase, register
from src.benchmarks.profile_fitting import build_ideal_from_edge_profile, profile_loss
from src.benchmarks.loss_beam.profile_C import profile_c, fit_c
from src.visualization.run_time_visual import plot_shape


class CShapeRecover(BenchmarkCase):
    name               = "C_recover"
    physics_type       = "beam"
    # Same wall as beam_C — wall-assisted shaping, same difficulty once recovered.
    walls              = [{"P0": [-100., -0.12], "P1": [100., -0.12]}]
    # Pre-run BC: beam bent hard left (θ = -0.8 rad), opposite the C-target direction.
    # Verified reachable: tip lands at ≈ (-0.080, -0.097) — well away from the target.
    initial_bc         = [-0.03, -0.01, -0.8]
    success_threshold  = 0.05

    target_description = (
        "A smooth C-arc: constant curvature throughout, all nodes bending in the "
        "same direction, total turning angle ~90°, tip pressed against the wall. "
        "NOTE: the beam starts in a bad configuration — bent in the wrong direction. "
        "You must first undo the initial misbend, then drive toward the C-shape target."
    )

    def reference_fn(self, x0: float = 0.0, y0: float = 0.0, theta: float = 0.0) -> list:
        """Quarter-circle arc anchored at (x0, y0) with initial slope theta."""
        return build_ideal_from_edge_profile(
            profile_c, n=9, arc_length=0.10,
            x0=x0, y0=y0, theta=theta, total_turn=np.pi / 2,
        )

    def loss_fn(self, nodes: list, ps=None, q_ref=None) -> dict:
        return profile_loss(nodes, profile_c, fit_c)


register(CShapeRecover())


# ---------------------------------------------------------------------------
# STANDALONE TEST
# Shows the C-shape reference (the goal), not the bad initial state.
# Set TEST_CONFIG to node positions to evaluate a specific configuration.
# ---------------------------------------------------------------------------

TEST_CONFIG = None


def main():
    case = CShapeRecover()
    ref  = case.reference_fn()

    x0_bad, y0_bad, theta_bad = case.initial_bc
    print(f"C_recover  |  initial_bc: x0={x0_bad:+.3f}  y0={y0_bad:+.3f}  θ={theta_bad:+.3f} rad")
    print(f"           |  target: C-arc (quarter-circle, ~90° turn, wall at y=-0.12)")

    if TEST_CONFIG is None:
        plot_shape(ref, name="C_recover (target reference)")
    else:
        result = case.loss_fn(TEST_CONFIG)
        print(f"           |  error: {result['error_percent']:.2f}%  |  {result['diagnostics']}")
        from src.visualization.run_time_visual import plot_comparison
        plot_comparison(ref, TEST_CONFIG, result, name="C_recover")


if __name__ == "__main__":
    main()
