"""
S-shape benchmark: full-sine curvature profile, one inflection point,
net turning angle near zero.

Run standalone:
    python -m src.benchmarks.cases.beam_S
"""

import numpy as np

from src.benchmarks import BenchmarkCase, register
from src.benchmarks.profile_fitting import build_ideal_from_edge_profile, profile_loss
from src.benchmarks.shapes.profile_S import profile_s, fit_s
from src.visualization.run_time_visual import plot_shape, plot_comparison


class SShape(BenchmarkCase):
    name               = "S"
    physics_type       = "beam"
    target_description = (
        "An S-curve: curvature changes sign at the midpoint, one inflection "
        "point, equal bending in opposite directions, net total turning angle "
        "near zero."
    )
    success_threshold  = 0.05

    def reference_fn(self, x0: float = 0.0, y0: float = 0.0, theta: float = 0.0) -> list:
        """Full-sine θ profile (amplitude 1.2 rad) anchored at (x0, y0)."""
        return build_ideal_from_edge_profile(
            profile_s, n=9, arc_length=0.10,
            x0=x0, y0=y0, theta=theta, amplitude=1.2,
        )

    def loss_fn(self, nodes: list, ps=None, q_ref=None) -> dict:
        return profile_loss(nodes, profile_s, fit_s)


register(SShape())


# ---------------------------------------------------------------------------
# STANDALONE TEST
# ---------------------------------------------------------------------------

TEST_CONFIG = None


def main():
    case = SShape()
    ref  = case.reference_fn()

    if TEST_CONFIG is None:
        plot_shape(ref, name="S (reference)")
    else:
        result = case.loss_fn(TEST_CONFIG)
        print(f"S  |  error: {result['error_percent']:.2f}%  |  {result['diagnostics']}")
        plot_comparison(ref, TEST_CONFIG, result, name="S")


if __name__ == "__main__":
    main()
