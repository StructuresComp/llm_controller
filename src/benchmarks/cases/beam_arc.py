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
        "Bend the DLO into a uniform circular arc. Target curvature values along the "
        "arc-length are provided below. "
        "Curvature is constant (the same everywhere) and positive, meaning the bend is "
        "CCW from s=0 (control node) to s=1 — negative curvature would mean a CW turn. "
        "The radius of curvature equals half the beam length. "
        "Invariant to translation and rotation — a rotated or shifted arc still scores 0%."
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
