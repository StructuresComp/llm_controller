"""
Snake-L benchmark: navigate the beam tip through an L-shaped tube and reach
the target point inside the horizontal exit channel.

Run standalone:
    python -m src.benchmarks.cases.beam_snake_L
"""

from src.benchmarks import BenchmarkCase, register

_TARGET = (0.05, -0.11)


class SnakeLCase(BenchmarkCase):
    name             = "snake_L"
    physics_type     = "beam"
    loss_type        = "tip"
    reference_tip_fn = _TARGET
    success_threshold = 0.05

    # L-shaped tube: vertical entry channel x ∈ [-0.01, +0.01] (top open),
    # turning into a horizontal exit channel y ∈ [-0.12, -0.10] going right to x = +0.12.
    walls = [
        {"P0": [-0.02,  -0.02], "P1": [-0.02, -0.12]},   # left wall of vertical channel
        {"P0": [-0.02,  -0.12], "P1": [ 0.12, -0.12]},   # floor
        {"P0": [ 0.12,  -0.08], "P1": [ 0.02, -0.08]},   # ceiling of horizontal channel
        {"P0": [ 0.02,  -0.08], "P1": [ 0.02, -0.02]},   # right wall of vertical channel
    ]

    target_description = (
        f"Navigate the free tip through an L-shaped tube and reach "
        f"({_TARGET[0]:.2f}, {_TARGET[1]:.2f}) m inside the horizontal exit channel.\n"
        f"Strategy: drive node 0 straight down (theta ~ -pi/2) to push the tip through "
        f"the vertical slot, then rotate theta toward 0 once the tip clears the corner "
        f"at y ~ -0.10 m to bend it rightward along the horizontal channel."
    )


register(SnakeLCase())


# ---------------------------------------------------------------------------
# STANDALONE TEST
# ---------------------------------------------------------------------------

def main():
    from types import SimpleNamespace
    import numpy as np
    from src.core import config
    case    = SnakeLCase()
    tx, ty  = case.reference_tip_fn
    nv      = 5
    rod_len = config.ROD_LENGTH
    ps      = SimpleNamespace(nv=nv, deltaL=rod_len / (nv - 1), rod_length=rod_len)
    q       = np.zeros(2 * nv)
    q[-2], q[-1] = tx, ty
    result  = case.loss_fn(q, ps)
    print(f"snake_L  tip-at-target  error: {result['error_percent']:.2f}%")


if __name__ == "__main__":
    main()
