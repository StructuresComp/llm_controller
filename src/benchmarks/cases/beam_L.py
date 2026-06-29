"""
L-shape benchmark: straight section followed by a sharp ~90° bend near the tip.

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
        "Straight along the first 65% of the beam, then a concentrated 90° cosine-bump turn centred at 65% arc-length. "
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
