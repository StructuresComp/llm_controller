"""
Tip-reach benchmark: drive the free tip to a fixed target position.

No walls. Loss is Euclidean distance from tip to target, normalised by rod length.
TARGET_X and TARGET_Y default to the values below but can be overridden in a
custom config file (config.TARGET_X = ..., config.TARGET_Y = ...).

Run standalone:
    python -m src.benchmarks.cases.beam_tip_reach
"""

from src.benchmarks import BenchmarkCase, register
from src.benchmarks.loss_beam.profile_tip_reach import tip_reach_loss


TARGET_X = 0.05
TARGET_Y = -0.08


class TipReachCase(BenchmarkCase):
    name          = "tip_reach"
    physics_type  = "beam"
    success_threshold = 0.05

    target_description = (
        f"Drive the free tip to ({TARGET_X:.2f}, {TARGET_Y:.2f}) m. "
        f"The beam hangs from the origin; bend it so the last node reaches the target. "
        f"No walls — the tip position is the only goal."
    )

    def reference_fn(self, x0=0.0, y0=0.0, theta=0.0):
        return None   # tip-reach case — no reference shape

    def loss_fn(self, nodes: list, ps=None, q_ref=None) -> dict:
        rod_length = ps.rod_length if ps is not None else 0.1
        return tip_reach_loss(nodes, TARGET_X, TARGET_Y, rod_length)


register(TipReachCase())


# ---------------------------------------------------------------------------
# STANDALONE TEST
# Set TEST_CONFIG to (x, y) node positions to evaluate a real configuration.
# ---------------------------------------------------------------------------

TEST_CONFIG = None


def main():
    case   = TipReachCase()
    nodes  = [(0.0, 0.0)] * 4 + [(TARGET_X, TARGET_Y)] if TEST_CONFIG is None else TEST_CONFIG
    result = case.loss_fn(nodes)
    label  = "sanity check (tip at target)" if TEST_CONFIG is None else "test config"
    print(f"tip_reach  [{label}]")
    print(f"  error: {result['error_percent']:.2f}%  |  {result['diagnostics']}")


if __name__ == "__main__":
    main()
