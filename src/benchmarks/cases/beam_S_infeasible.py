"""
S-infeasible benchmark: a tight, high-curvature S whose opposite-signed
curvature on each half is beyond what single-end (clamped) actuation can
impose. Intended to run in free space (WALLS = []); the loop is expected to
plateau without solving. This is the stress / infeasibility-recognition cell
for the stability study.

Run standalone:
    python -m src.benchmarks.cases.beam_S_infeasible
"""

import numpy as np

from src.benchmarks import BenchmarkCase, register
from src.benchmarks.loss_beam.profile_S import kappa_s
from src.visualization.run_time_visual import plot_shape


def kappa_s_infeasible(s_norm: np.ndarray, rod_length: float = 1.0) -> np.ndarray:
    """Tighter S target than the standard feasible S benchmark."""
    return kappa_s(s_norm, rod_length=rod_length, As=20.0, Aend=6.0, sig=0.22)


class SInfeasibleShape(BenchmarkCase):
    name               = "S_infeasible"
    physics_type       = "beam"
    loss_type          = "curvature"
    reference_kappa_fn = staticmethod(kappa_s_infeasible)
    target_description = (
        "A tight S-curve with large opposite-signed curvature on each half. "
        "Deliberately beyond what single-end actuation can realize — used to "
        "probe whether the controller recognizes infeasibility rather than "
        "optimizing indefinitely. Invariant to translation and rotation — a "
        "rotated or shifted tight S still scores 0%."
    )
    success_threshold  = 0.05


register(SInfeasibleShape())


# ---------------------------------------------------------------------------
# STANDALONE TEST
# ---------------------------------------------------------------------------

TEST_CONFIG = None


def main():
    case = SInfeasibleShape()
    ref  = case.reference_fn()
    plot_shape(ref, name="S_infeasible (reference)")


if __name__ == "__main__":
    main()
