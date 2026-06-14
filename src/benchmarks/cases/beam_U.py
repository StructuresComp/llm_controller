"""
U-shape benchmark: deep semicircular bend, total turning angle ≈ π.

Run standalone:
    python -m src.benchmarks.cases.beam_U
"""

import numpy as np

from src.benchmarks import BenchmarkCase, register
from src.benchmarks.profile_fitting import build_ideal_from_edge_profile, profile_loss
from src.benchmarks.loss_beam.profile_U import profile_u, fit_u
from src.visualization.run_time_visual import plot_shape, plot_comparison


class UShape(BenchmarkCase):
    name               = "U"
    physics_type       = "beam"
    target_description = (
        "A deep U-bend: total turning angle near π (semicircle), consistent "
        "curvature sign throughout, endpoints much closer together than arc "
        "length, tip in contact with the wall."
    )
    success_threshold  = 0.05

    def reference_fn(self, x0: float = 0.0, y0: float = 0.0, theta: float = 0.0) -> list:
        """Semicircle anchored at (x0, y0) with initial slope theta."""
        return build_ideal_from_edge_profile(
            profile_u, n=9, arc_length=0.10,
            x0=x0, y0=y0, theta=theta, total_turn=np.pi,
        )

    def loss_fn(self, nodes: list, ps=None, q_ref=None) -> dict:
        return profile_loss(nodes, profile_u, fit_u)


register(UShape())


# ---------------------------------------------------------------------------
# STANDALONE TEST
# ---------------------------------------------------------------------------

TEST_CONFIG = None


def main():
    case = UShape()
    ref  = case.reference_fn()

    if TEST_CONFIG is None:
        plot_shape(ref, name="U (reference)")
    else:
        result = case.loss_fn(TEST_CONFIG)
        print(f"U  |  error: {result['error_percent']:.2f}%  |  {result['diagnostics']}")
        plot_comparison(ref, TEST_CONFIG, result, name="U")


if __name__ == "__main__":
    main()
