"""
Sine-wave benchmark: extract profile from ideal nodes using make_profile_from_nodes.

Target: bend the beam into a smooth half-sine wave.

Run standalone:
    python -m src.benchmarks.cases.beam_sine
"""

import numpy as np

from src.benchmarks import BenchmarkCase, register
from src.benchmarks.profile_fitting import (
    build_ideal_from_edge_profile,
    profile_loss,
    make_profile_from_nodes,
)
from src.visualization.run_time_visual import plot_shape, plot_comparison


class SineCase(BenchmarkCase):
    name = "sine"
    physics_type = "beam"
    target_description = (
        "A smooth half-sine wave: starting straight, gradually curving upward then "
        "downward, returning to straight orientation. Tests shape extraction from nodes."
    )
    success_threshold = 0.05

    def __init__(self):
        # Define ideal nodes as a half sine wave (arc-length ≈ 0.10 m, amplitude 0.03 m)
        t = np.linspace(0, np.pi, 12)
        self.ideal_nodes = [(x * 0.10 / np.pi, 0.03 * np.sin(x)) for x in t]

        # Compute actual arc-length from ideal nodes
        ideal_array = np.array(self.ideal_nodes, dtype=float)
        edge_vecs = np.diff(ideal_array, axis=0)
        edge_lengths = np.linalg.norm(edge_vecs, axis=1)
        self.arc_length = float(np.sum(edge_lengths))

        # Extract profile from ideal nodes
        self.profile_fn, self.fit_fn = make_profile_from_nodes(self.ideal_nodes)

    def reference_fn(self, x0: float = 0.0, y0: float = 0.0, theta: float = 0.0) -> list:
        """Reconstruct from extracted profile, anchored at (x0, y0) with initial slope theta."""
        return build_ideal_from_edge_profile(
            self.profile_fn,
            n=len(self.ideal_nodes),
            arc_length=self.arc_length,
            x0=x0,
            y0=y0,
            theta=theta,
        )

    def loss_fn(self, nodes: list, ps=None, q_ref=None) -> dict:
        return profile_loss(nodes, self.profile_fn, self.fit_fn)


register(SineCase())


# ---------------------------------------------------------------------------
# STANDALONE TEST
# Set TEST_CONFIG to a list of (x, y) tuples to also compute and display loss.
# ---------------------------------------------------------------------------

TEST_CONFIG = None  # e.g. [(0.0, 0.0), (0.01, 0.015), ...]


def main():
    case = SineCase()
    ref = case.reference_fn()

    if TEST_CONFIG is None:
        plot_shape(ref, name="Sine (reference)")
    else:
        result = case.loss_fn(TEST_CONFIG)
        print(
            f"Sine  |  error: {result['error_percent']:.2f}%  |  {result['diagnostics']}"
        )
        plot_comparison(ref, TEST_CONFIG, result, name="Sine")


if __name__ == "__main__":
    main()
