"""
U-shape benchmark: sharp mid-arc bend with mostly straight stems.

Run standalone:
    python -m src.benchmarks.cases.beam_U
"""

from src.benchmarks import BenchmarkCase, register
from src.benchmarks.loss_beam.profile_U import kappa_u
from src.visualization.run_time_visual import plot_shape


class UShape(BenchmarkCase):
    name               = "U"
    physics_type       = "beam"
    loss_type          = "curvature"
    reference_kappa_fn = staticmethod(kappa_u)
    target_description = (
        "Straight stems (zero curvature) with a concentrated 180° cosine-bump turn centred at 50% arc-length. "
        "Invariant to translation and rotation — a rotated or shifted U still scores 0%."
    )
    success_threshold  = 0.05


register(UShape())


# ---------------------------------------------------------------------------
# STANDALONE TEST
# ---------------------------------------------------------------------------

def main():
    case = UShape()
    ref  = case.reference_fn()
    plot_shape(ref, name="U (reference)")


if __name__ == "__main__":
    main()
