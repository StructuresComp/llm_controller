"""
C-shape benchmark: realistic C with terminal curls.

Run standalone:
    python -m src.benchmarks.cases.beam_C
"""

from src.benchmarks import BenchmarkCase, register
from src.benchmarks.loss_beam.profile_C import kappa_c
from src.visualization.run_time_visual import plot_shape


class CShape(BenchmarkCase):
    name               = "C"
    physics_type       = "beam"
    loss_type          = "curvature"
    reference_kappa_fn = staticmethod(kappa_c)
    target_description = (
        "Uniform curvature throughout with stronger curls at both endpoints. "
        "All bending in one direction. "
        "Invariant to translation and rotation — a rotated or shifted C still scores 0%."
    )
    success_threshold  = 0.05


register(CShape())


# ---------------------------------------------------------------------------
# STANDALONE TEST
# ---------------------------------------------------------------------------

def main():
    case = CShape()
    ref  = case.reference_fn()
    plot_shape(ref, name="C (reference)")


if __name__ == "__main__":
    main()
