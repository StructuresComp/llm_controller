# ---------------------------------------------------------------------------
# src/benchmarks/shapes/beam_L.py
# ---------------------------------------------------------------------------
# L-shape benchmark: straight section followed by a sharp ~90° bend near
# the tip.
#
# Run standalone to visualize the ideal shape and verify 0% error:
#   python -m src.benchmarks.shapes.beam_L
# ---------------------------------------------------------------------------

import numpy as np

from src.benchmarks import Benchmark, register
from src.benchmarks.losses import profile_loss
from src.benchmarks.visualization import plot_shape
from src.tools.shape_checker import THETA_PROFILES, build_ideal_from_edge_profile


# ---------------------------------------------------------------------------
# REFERENCE GENERATOR
# ---------------------------------------------------------------------------

def reference_fn(n: int = 9, arc_length: float = 0.10) -> list:
    """Ideal L-shape: straight then abrupt 90° turn at 65% of arc length."""
    return build_ideal_from_edge_profile(
        THETA_PROFILES["L"], n, arc_length, total_turn=np.pi / 2, turn_at=0.65
    )


# ---------------------------------------------------------------------------
# LOSS
# ---------------------------------------------------------------------------

def loss_fn(nodes: list, ps=None) -> dict:
    return profile_loss(nodes, "L")


# ---------------------------------------------------------------------------
# REGISTRATION
# ---------------------------------------------------------------------------

benchmark = Benchmark(
    name               = "L",
    physics_type       = "beam",
    target_description = (
        "An L-shape: nearly straight along most of the beam, then a sharp "
        "near-90° bend concentrated near the tip, tip deflected perpendicular "
        "to the initial hanging direction."
    ),
    loss_fn            = loss_fn,
    reference_fn       = reference_fn,
    success_threshold  = 0.05,
)

register(benchmark)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    nodes  = reference_fn()
    result = loss_fn(nodes)
    print(f"L-shape  |  error: {result['error_percent']:.2f}%  |  {result['diagnostics']}")
    plot_shape(nodes, name="L (ideal)", error_percent=result["error_percent"])


if __name__ == "__main__":
    main()
