"""
Circular-arc benchmark: uniform curvature, radius = rod_length / 2.

Total turning angle = rod_length / radius = 2 radians (~114 degrees).
Run standalone:
    python -m src.benchmarks.cases.beam_arc
"""

import numpy as np

from src.benchmarks import BenchmarkCase, register
from src.visualization.run_time_visual import plot_shape


def kappa_arc(s_norm, rod_length=1.0):
    """Constant curvature: radius = rod_length / 2, so kappa = 2 / rod_length."""
    return np.full_like(np.asarray(s_norm, dtype=float), 2.0 / rod_length)


class ArcShape(BenchmarkCase):
    name               = "arc"
    physics_type       = "beam"
    loss_type          = "curvature"
    reference_kappa_fn = staticmethod(kappa_arc)
    target_description = (
        "A uniform circular arc with constant positive curvature along the entire beam. "
        "Positive curvature means the bend is CCW as we go from s=0 (control node) to s=1. "
        "Negative curvature means CW turn. "
        "The radius of curvature equals half the beam length, giving a total turning angle of "
        "2 radians (about 114 degrees)."
    )
    success_threshold  = 0.05


register(ArcShape())


# ---------------------------------------------------------------------------
# STANDALONE TEST
# ---------------------------------------------------------------------------

def main():
    case = ArcShape()
    ref  = case.reference_fn()
    plot_shape(ref, name="arc (reference)")


if __name__ == "__main__":
    main()
