"""
L-shape benchmark: straight first arm, a concentrated ~90° bend centered at 65%
arc-length, then a shorter straight second arm to the tip.

Run standalone:
    python -m src.benchmarks.cases.beam_L
"""

from src.benchmarks import BenchmarkCase, register
from src.benchmarks.loss_beam.profile_L import kappa_l
from src.visualization.run_time_visual import plot_shape


class LShape(BenchmarkCase):
    name               = "L"
    physics_type       = "beam"
    loss_type          = "curvature"
    reference_kappa_fn = staticmethod(kappa_l)
    target_description = (
        "Bend the DLO into an L-shape. Target curvature values along the arc-length "
        "are provided below. "
        "Straight for the first ~52.5% of the beam, then a concentrated 90-degree "
        "cosine-bump turn centered at 65% arc-length (spanning roughly 52.5%-77.5%), "
        "followed by a straight ~22.5% second arm to the tip. "
        "Invariant to translation and rotation — a rotated or shifted L still scores 0%."
    )
    success_threshold  = 0.05


register(LShape())


# ---------------------------------------------------------------------------
# STANDALONE TEST
# ---------------------------------------------------------------------------

def main():
    case = LShape()
    ref  = case.reference_fn()
    plot_shape(ref, name="L (reference)")


if __name__ == "__main__":
    main()
