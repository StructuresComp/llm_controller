"""
S-infeasible benchmark: a tight, high-amplitude S whose opposite-signed
curvature on each half is beyond what single-end (clamped) actuation can
impose. Intended to run in free space (WALLS = []); the loop is expected to
plateau without solving. This is the stress / infeasibility-recognition cell
for the stability study — the place we expect non-converging closed-loop
dynamics (rho near or above 1).

It reuses the S turning-angle profile but pins a much larger target amplitude
(2.5 rad vs the feasible S's 1.2 rad) via make_fit_s.

Run standalone:
    python -m src.benchmarks.cases.beam_S_infeasible
"""

from src.benchmarks import BenchmarkCase, register
from src.benchmarks.profile_fitting import build_ideal_from_edge_profile, profile_loss
from src.benchmarks.loss_beam.profile_S import profile_s, make_fit_s
from src.visualization.run_time_visual import plot_shape, plot_comparison


# Target amplitude well beyond the feasible-S cell (1.2 rad).
_TARGET_AMPLITUDE = 2.5
_fit_infeasible = make_fit_s(_TARGET_AMPLITUDE)


class SInfeasibleShape(BenchmarkCase):
    name               = "S_infeasible"
    physics_type       = "beam"
    target_description = (
        "A tight S-curve with large opposite-signed curvature on each half "
        "(turning amplitude ~2.5 rad). Deliberately beyond what single-end "
        "actuation can realize — used to probe whether the controller "
        "recognizes infeasibility rather than optimizing indefinitely."
    )
    success_threshold  = 0.05

    def reference_fn(self, x0: float = 0.0, y0: float = 0.0, theta: float = 0.0) -> list:
        """Full-sine theta profile (amplitude 2.5 rad) anchored at (x0, y0)."""
        return build_ideal_from_edge_profile(
            profile_s, n=9, arc_length=0.10,
            x0=x0, y0=y0, theta=theta, amplitude=_TARGET_AMPLITUDE,
        )

    def loss_fn(self, nodes: list, ps=None, q_ref=None) -> dict:
        return profile_loss(nodes, profile_s, _fit_infeasible)


register(SInfeasibleShape())


# ---------------------------------------------------------------------------
# STANDALONE TEST
# ---------------------------------------------------------------------------

TEST_CONFIG = None


def main():
    case = SInfeasibleShape()
    ref  = case.reference_fn()

    if TEST_CONFIG is None:
        plot_shape(ref, name="S_infeasible (reference)")
    else:
        result = case.loss_fn(TEST_CONFIG)
        print(f"S_infeasible  |  error: {result['error_percent']:.2f}%  |  {result['diagnostics']}")
        plot_comparison(ref, TEST_CONFIG, result, name="S_infeasible")


if __name__ == "__main__":
    main()
