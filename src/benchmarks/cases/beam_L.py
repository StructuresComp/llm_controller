"""
L-shape benchmark: straight section followed by a sharp ~90° bend near the tip.

Run standalone:
    python -m src.benchmarks.cases.beam_L
"""

import numpy as np

from src.benchmarks import BenchmarkCase, register
from src.benchmarks.profile_fitting import build_ideal_from_edge_profile, profile_loss
from src.benchmarks.shapes.profile_L import profile_l, fit_l
from src.visualization.run_time_visual import plot_shape, plot_comparison


class LShape(BenchmarkCase):
    name               = "L"
    physics_type       = "beam"
    target_description = (
        "An L-shape: nearly straight along most of the beam, then a sharp "
        "near-90° bend concentrated near the tip, tip deflected perpendicular "
        "to the initial hanging direction."
    )
    success_threshold  = 0.05

    def reference_fn(self, x0: float = 0.0, y0: float = 0.0, theta: float = 0.0) -> list:
        """Straight-then-step (turn at 65% arc length) anchored at (x0, y0)."""
        return build_ideal_from_edge_profile(
            profile_l, n=9, arc_length=0.10,
            x0=x0, y0=y0, theta=theta, total_turn=np.pi / 2, turn_at=0.65,
        )

    def loss_fn(self, nodes: list, ps=None, q_ref=None) -> dict:
        return profile_loss(nodes, profile_l, fit_l)


register(LShape())


# ---------------------------------------------------------------------------
# STANDALONE TEST
# ---------------------------------------------------------------------------

TEST_CONFIG = None


def main():
    case = LShape()
    ref  = case.reference_fn()

    if TEST_CONFIG is None:
        plot_shape(ref, name="L (reference)")
    else:
        result = case.loss_fn(TEST_CONFIG)
        print(f"L  |  error: {result['error_percent']:.2f}%  |  {result['diagnostics']}")
        plot_comparison(ref, TEST_CONFIG, result, name="L")


if __name__ == "__main__":
    main()
