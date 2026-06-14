"""
Snake-L benchmark: navigate the beam tip through an L-shaped tube and reach
the target point (0.05, -0.11) m inside the horizontal exit channel.

Run standalone:
    python -m src.benchmarks.cases.beam_snake_L
"""

from src.benchmarks import BenchmarkCase, register
from src.benchmarks.loss_beam.profile_tip_reach import tip_reach_loss


TARGET_X = 0.05
TARGET_Y = -0.11


class SnakeLCase(BenchmarkCase):
    name          = "snake_L"
    physics_type  = "beam"
    success_threshold = 0.05

    # L-shaped tube: vertical entry channel x ∈ [-0.01, +0.01] (top open),
    # turning into a horizontal exit channel y ∈ [-0.12, -0.10] going right to x = +0.12.
    walls = [
        {"P0": [-0.01,  0.02], "P1": [-0.01, -0.12]},   # left wall of vertical channel
        {"P0": [-0.01, -0.12], "P1": [ 0.12, -0.12]},   # floor
        {"P0": [ 0.12, -0.12], "P1": [ 0.12, -0.10]},   # right end cap
        {"P0": [ 0.12, -0.10], "P1": [ 0.01, -0.10]},   # ceiling of horizontal channel
        {"P0": [ 0.01, -0.10], "P1": [ 0.01,  0.02]},   # right wall of vertical channel
    ]

    target_description = (
        f"Navigate the free tip through an L-shaped tube and reach "
        f"({TARGET_X:.2f}, {TARGET_Y:.2f}) m inside the horizontal exit channel.\n"
        f"Strategy: drive node 0 straight down (θ ≈ 0) to push the tip through "
        f"the vertical slot, then rotate θ positive once the tip clears the corner "
        f"at y ≈ -0.10 m to bend it rightward along the horizontal channel."
    )

    def reference_fn(self, x0=0.0, y0=0.0, theta=0.0):
        return None   # tip-reach case — no reference shape

    def loss_fn(self, nodes: list, ps=None, q_ref=None) -> dict:
        rod_length = ps.rod_length if ps is not None else 0.1
        return tip_reach_loss(nodes, TARGET_X, TARGET_Y, rod_length)


register(SnakeLCase())


# ---------------------------------------------------------------------------
# STANDALONE TEST
# Set TEST_CONFIG to (x, y) node positions to evaluate a real configuration.
# ---------------------------------------------------------------------------

TEST_CONFIG = None


def main():
    case   = SnakeLCase()
    nodes  = [(0.0, 0.0)] * 4 + [(TARGET_X, TARGET_Y)] if TEST_CONFIG is None else TEST_CONFIG
    result = case.loss_fn(nodes)
    label  = "sanity check (tip at target)" if TEST_CONFIG is None else "test config"
    print(f"snake_L  [{label}]")
    print(f"  error: {result['error_percent']:.2f}%  |  {result['diagnostics']}")


if __name__ == "__main__":
    main()
