"""
C-shape benchmark: smooth single-sided arc, constant curvature.

Run standalone:
    python -m src.benchmarks.shapes.beam_C
"""

import numpy as np

from src.benchmarks import BenchmarkCase, register
from src.benchmarks.losses import build_ideal_from_edge_profile, profile_loss
from src.visualization.run_time_visual import plot_shape, plot_comparison


# ---------------------------------------------------------------------------
# THETA PROFILE + FIT  (linear ramp = constant curvature = circular arc)
# ---------------------------------------------------------------------------

def _profile_c(s_norm: np.ndarray, total_turn: float = 1.0) -> np.ndarray:
    return total_turn * s_norm


def _fit_c(s_norm: np.ndarray, dtheta: np.ndarray) -> dict:
    denom = float(np.dot(s_norm, s_norm)) + 1e-12
    return {"total_turn": float(np.dot(s_norm, dtheta) / denom)}


# ---------------------------------------------------------------------------
# CASE
# ---------------------------------------------------------------------------

class CShape(BenchmarkCase):
    name               = "C"
    physics_type       = "beam"
    target_description = (
        "A smooth C-arc: constant curvature throughout, all nodes bending in the "
        "same direction, total turning angle ~90°, tip pressed against the wall."
    )
    success_threshold  = 0.05

    def reference_fn(self, x0: float = 0.0, y0: float = 0.0, theta: float = 0.0) -> list:
        """Quarter-circle arc anchored at (x0, y0) with initial slope theta."""
        return build_ideal_from_edge_profile(
            _profile_c, n=9, arc_length=0.10,
            x0=x0, y0=y0, theta=theta, total_turn=np.pi / 2,
        )

    def loss_fn(self, nodes: list, ps=None, q_ref=None) -> dict:
        return profile_loss(nodes, _profile_c, _fit_c)


register(CShape())


# ---------------------------------------------------------------------------
# STANDALONE TEST
# Set TEST_CONFIG to a list of (x, y) tuples to also compute and display loss.
# ---------------------------------------------------------------------------

TEST_CONFIG = None  # e.g. [(0.0, 0.0), (0.01, -0.02), ...]


def main():
    case = CShape()
    ref  = case.reference_fn()

    if TEST_CONFIG is None:
        plot_shape(ref, name="C (reference)")
    else:
        result = case.loss_fn(TEST_CONFIG)
        print(f"C  |  error: {result['error_percent']:.2f}%  |  {result['diagnostics']}")
        plot_comparison(ref, TEST_CONFIG, result, name="C")


if __name__ == "__main__":
    main()
