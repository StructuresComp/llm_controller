"""
U-shape benchmark: broad mid-arc cosine bend (~65% of arc-length) with short straight stems.

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
        "Bend the DLO into a U-shape. Target curvature values along the arc-length "
        "are provided below. "
        "Straight stems at both ends (zero curvature, roughly 35% of the total length "
        "combined) with a broad cosine-shaped turn spanning the middle ~65% of the "
        "arc-length, centered at 50%, totaling a 180-degree turn. "
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
