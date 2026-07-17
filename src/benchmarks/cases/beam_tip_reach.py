"""
Tip-reach benchmark: drive the free tip to a fixed target position.

A single floor wall spans x in [-0.05, 0.05] at y = -0.2 m, below the target.
Loss is Euclidean distance from tip to target, normalised by rod length.

Run standalone:
    python -m src.benchmarks.cases.beam_tip_reach
"""

from src.benchmarks import BenchmarkCase, register

TARGET_X = 0.05
TARGET_Y = -0.08


class TipReachCase(BenchmarkCase):
    name             = "tip_reach"
    physics_type     = "beam"
    loss_type        = "tip"
    reference_tip_fn = (TARGET_X, TARGET_Y)
    success_threshold = 0.05

    walls = [
        {"P0": [-0.05, -0.2], "P1": [0.05, -0.2]},   # floor, below the target
    ]

    target_description = (
        f"Drive the free tip (arc-length, s=1) to ({TARGET_X:.2f}, {TARGET_Y:.2f}) m. "
        f"The beam hangs from the origin; bend it so the last node reaches the target. "
        f"A floor wall sits below the target at y = -0.2 m (x in [-0.05, 0.05] m) — "
        f"it should not block reaching this target, but stops the beam from passing "
        f"through the floor."
    )


register(TipReachCase())


# ---------------------------------------------------------------------------
# STANDALONE TEST
# ---------------------------------------------------------------------------

def main():
    from types import SimpleNamespace
    import numpy as np
    from src.core import config
    case    = TipReachCase()
    tx, ty  = case.reference_tip_fn
    nv      = 5
    rod_len = config.ROD_LENGTH
    ps      = SimpleNamespace(nv=nv, deltaL=rod_len / (nv - 1), rod_length=rod_len)
    q       = np.zeros(2 * nv)
    q[-2], q[-1] = tx, ty
    result  = case.loss_fn(q, ps)
    print(f"tip_reach  tip-at-target  error: {result['error_percent']:.2f}%")


if __name__ == "__main__":
    main()
