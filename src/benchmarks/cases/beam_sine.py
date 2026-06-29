"""
Sine-wave benchmark: half-sine curvature profile extracted from ideal nodes.

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
        "A smooth half-sine wave: starting straight, gradually curving then "
        "returning to straight. Arc length ≈ 0.10 m, amplitude 0.03 m."
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
        return self.ideal_nodes - self.ideal_nodes[0] + np.array([x0, y0])


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
