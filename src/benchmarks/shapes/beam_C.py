# ---------------------------------------------------------------------------
# src/benchmarks/shapes/beam_C.py
# ---------------------------------------------------------------------------
# C-shape benchmark: smooth single-sided arc, constant curvature.
#
# Run standalone to visualize the ideal shape and verify 0% error:
#   python -m src.benchmarks.shapes.beam_C
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
    """Ideal C-shape: quarter-circle arc (90° total turn)."""
    return build_ideal_from_edge_profile(
        THETA_PROFILES["C"], n, arc_length, total_turn=np.pi / 2
    )


# ---------------------------------------------------------------------------
# LOSS
# ---------------------------------------------------------------------------

def loss_fn(nodes: list, ps=None) -> dict:
    return profile_loss(nodes, "C")


# ---------------------------------------------------------------------------
# REGISTRATION
# ---------------------------------------------------------------------------

benchmark = Benchmark(
    name               = "C",
    physics_type       = "beam",
    target_description = (
        "A smooth C-arc: constant curvature throughout, all nodes bending in the "
        "same direction, large total turning angle, tip pressed against the wall."
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
    print(f"C-shape  |  error: {result['error_percent']:.2f}%  |  {result['diagnostics']}")
    plot_shape(nodes, name="C (ideal)", error_percent=result["error_percent"])


if __name__ == "__main__":
    main()
