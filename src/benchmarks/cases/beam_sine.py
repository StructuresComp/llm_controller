"""
Sine-wave benchmark: beam follows a half-sine-shaped path; curvature is derived
numerically from the ideal shape's node positions (not a hand-designed function).

Run standalone:
    python -m src.benchmarks.cases.beam_sine
"""

import numpy as np

from src.benchmarks import BenchmarkCase, register
from src.tools.Discrete_Elastic_Beams import compute_kappa_dimensional
from src.visualization.run_time_visual import plot_shape


class SineCase(BenchmarkCase):
    name = "sine"
    physics_type = "beam"
    loss_type = "curvature"
    target_description = (
        "Bend the DLO into a smooth arc resembling half of a sine wave. Target curvature "
        "values along the arc-length are provided below. "
        "Starts and ends straight (zero curvature), with a single bump peaking at the "
        "midpoint (negative curvature throughout — bends like a hill, not a valley). "
        "Horizontal span ~0.10 m, arc length ~0.12 m, amplitude ~0.03 m. "
        "Invariant to translation and rotation — a rotated or shifted curve still scores 0%."
    )
    success_threshold = 0.05

    def __init__(self):
        t     = np.linspace(0, np.pi, 12)
        ideal = np.array([(x * 0.10 / np.pi, 0.03 * np.sin(x)) for x in t])
        self.ideal_nodes        = ideal
        arc_length              = float(np.sum(np.linalg.norm(np.diff(ideal, axis=0), axis=1)))
        n                       = len(ideal)
        dL                      = arc_length / (n - 1)
        self.reference_kappa_fn = compute_kappa_dimensional(ideal.flatten(), n, dL)

    def reference_fn(self, x0: float = 0.0, y0: float = 0.0, theta: float = -np.pi / 2, ps=None):
        centered     = self.ideal_nodes - self.ideal_nodes[0]
        native_theta = np.arctan2(centered[1, 1], centered[1, 0])
        delta        = theta - native_theta
        rotation     = np.array([[np.cos(delta), -np.sin(delta)], [np.sin(delta), np.cos(delta)]])
        rotated      = centered @ rotation.T
        return rotated + np.array([x0, y0])


register(SineCase())


# ---------------------------------------------------------------------------
# STANDALONE TEST
# ---------------------------------------------------------------------------

def main():
    case = SineCase()
    ref  = case.reference_fn()
    plot_shape(ref, name="Sine (reference)")


if __name__ == "__main__":
    main()
