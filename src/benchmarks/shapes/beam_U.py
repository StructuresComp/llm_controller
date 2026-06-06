# ---------------------------------------------------------------------------
# src/benchmarks/shapes/beam_U.py
# ---------------------------------------------------------------------------
# U-shape benchmark: deep semicircular bend, turning angle ≈ π.
#
# Run standalone to visualize the ideal shape and verify 0% error:
#   python -m src.benchmarks.shapes.beam_U
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
    """Ideal U-shape: semicircle (π total turn)."""
    return build_ideal_from_edge_profile(
        THETA_PROFILES["U"], n, arc_length, total_turn=np.pi
    )


# ---------------------------------------------------------------------------
# LOSS
# ---------------------------------------------------------------------------

def loss_fn(nodes: list, ps=None) -> dict:
    return profile_loss(nodes, "U")


# ---------------------------------------------------------------------------
# REGISTRATION
# ---------------------------------------------------------------------------

benchmark = Benchmark(
    name               = "U",
    physics_type       = "beam",
    target_description = (
        "A deep U-bend: total turning angle near π (semicircle), consistent "
        "curvature sign throughout, endpoints much closer together than at start, "
        "tip in contact with the wall."
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
    print(f"U-shape  |  error: {result['error_percent']:.2f}%  |  {result['diagnostics']}")
    plot_shape(nodes, name="U (ideal)", error_percent=result["error_percent"])


if __name__ == "__main__":
    main()
