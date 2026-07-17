"""
Circular-arc benchmark: uniform curvature, normalized by rod_length like every
other profile in this package (kappa_c, kappa_l, kappa_u, kappa_s all divide by
rod_length too) -- dimensional curvature = CURVATURE / rod_length.

Run standalone:
    python -m src.benchmarks.cases.beam_arc
"""

import numpy as np

from src.benchmarks import BenchmarkCase, register
from src.visualization.run_time_visual import plot_shape

# Single source of truth for this case's curvature. Dimensional curvature =
# CURVATURE / rod_length; radius = rod_length / CURVATURE. If ROD_LENGTH = X
# (config.py) and CURVATURE = Y, dimensional curvature = Y/X (unit: 1/meter),
# radius of curvature = X/Y.
CURVATURE = - 1.0


def kappa_arc(s_norm, rod_length=1.0, curvature=CURVATURE):
    """Constant curvature = curvature / rod_length (radius = rod_length / curvature)."""
    return np.full_like(np.asarray(s_norm, dtype=float), curvature / rod_length)


class ArcShape(BenchmarkCase):
    name               = "arc"
    physics_type       = "beam"
    loss_type          = "curvature"
    reference_kappa_fn = staticmethod(kappa_arc)
    target_description = (
        "Bend the DLO into a uniform circular arc. Curvature is constant (the same "
        "everywhere) along the whole arc — sign and magnitude are given below: "
        "positive curvature means the bend is CCW from s=0 (control node) to s=1, "
        "negative means CW. Invariant to translation and rotation — a rotated or "
        "shifted arc still scores 0%."
    )
    success_threshold  = 0.05

    def describe_fn(self, q, ps) -> str:
        """Append concrete radius/curvature numbers computed from this run's actual
        ps.rod_length (target_description itself must stay a static string, so it
        can't carry a per-run value) after target_description in the "Target:" line."""
        radius     = ps.rod_length / CURVATURE
        curvature  = 1.0 / radius  # same sign as CURVATURE
        concrete = (
            f"Target radius of curvature is {radius:+g} m, i.e., curvature is "
            f"{curvature:+g} /m. Recall, DLO length is "
            f"{ps.rod_length:g} m."
        )
        target_line = f"Target: {self.target_description}"
        return super().describe_fn(q, ps).replace(target_line, f"{target_line} {concrete}", 1)


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
