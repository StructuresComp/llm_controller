"""
Curvature round-trip tests.

Build ideal nodes from build_ideal_from_edge_profile, flatten to q, then verify
compute_curvature_loss against the same kappa_fn is near zero (small discretization
error only). Anything above PASS_THRESHOLD is a bug.

Run from project root:
    python -m src.tests.test_shape_checker
"""

import numpy as np
from types import SimpleNamespace

from src.benchmarks.beam_losses import build_ideal_from_edge_profile, compute_curvature_loss
from src.benchmarks.loss_beam.profile_C import kappa_c
from src.benchmarks.loss_beam.profile_U import kappa_u
from src.benchmarks.loss_beam.profile_S import kappa_s
from src.benchmarks.loss_beam.profile_L import kappa_l

ROD_LENGTH     = 0.10
N              = 21
PASS_THRESHOLD = 5.0   # % — expect small discretization error, not zero


def _round_trip_error(kappa_fn) -> float:
    ps  = SimpleNamespace(nv=N, deltaL=ROD_LENGTH / (N - 1), rod_length=ROD_LENGTH)
    ref = build_ideal_from_edge_profile(kappa_fn, N, ROD_LENGTH)
    return compute_curvature_loss(ref.flatten(), kappa_fn, ps)


def run() -> bool:
    cases = [("C", kappa_c), ("U", kappa_u), ("S", kappa_s), ("L", kappa_l)]
    print("=" * 60)
    print("CURVATURE ROUND-TRIP  (expect < %.1f%% error)" % PASS_THRESHOLD)
    print("=" * 60)
    all_pass = True
    for name, kappa_fn in cases:
        err    = _round_trip_error(kappa_fn)
        status = "PASS" if err < PASS_THRESHOLD else "FAIL"
        if err >= PASS_THRESHOLD:
            all_pass = False
        print(f"  [{status}]  {name}  {err:.4f}%")
    print("=" * 60)
    print("All passed." if all_pass else "SOME TESTS FAILED.")
    return all_pass


if __name__ == "__main__":
    run()
