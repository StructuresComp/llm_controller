"""
S-shape benchmark: two opposite bowls via full-sine curvature profile.

Run standalone:
    python -m src.benchmarks.cases.beam_S
"""

from src.benchmarks import BenchmarkCase, register
from src.benchmarks.loss_beam.profile_S import kappa_s
from src.visualization.run_time_visual import plot_shape


class SShape(BenchmarkCase):
    name               = "S"
    physics_type       = "beam"
    loss_type          = "curvature"
    reference_kappa_fn = staticmethod(kappa_s)
    target_description = (
        "Bend the DLO into an S-shape. Target curvature values along the arc-length "
        "are provided below. "
        "Overall, it is a full-sine curvature profile: positive first half, negative "
        "second half, exactly antisymmetric about the midpoint with one inflection at "
        "50% arc-length. End corrections add nonzero curvature at both tips while "
        "preserving that antisymmetry. "
        "Invariant to translation and rotation — a rotated or shifted S still scores 0%."
    )
    success_threshold  = 0.05


register(SShape())


# ---------------------------------------------------------------------------
# STANDALONE TEST
# ---------------------------------------------------------------------------

def main():
    case = SShape()
    ref  = case.reference_fn()
    plot_shape(ref, name="S (reference)")


if __name__ == "__main__":
    main()
