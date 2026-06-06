# ---------------------------------------------------------------------------
# src/benchmarks/shapes/beam_S.py
# ---------------------------------------------------------------------------
# S-shape benchmark: full-sine curvature profile, one inflection point,
# net turning angle near zero.
#
# Run standalone to visualize the ideal shape and verify 0% error:
#   python -m src.benchmarks.shapes.beam_S
# ---------------------------------------------------------------------------

from src.benchmarks import Benchmark, register
from src.benchmarks.losses import profile_loss
from src.benchmarks.visualization import plot_shape
from src.tools.shape_checker import THETA_PROFILES, build_ideal_from_edge_profile


# ---------------------------------------------------------------------------
# REFERENCE GENERATOR
# ---------------------------------------------------------------------------

def reference_fn(n: int = 9, arc_length: float = 0.10) -> list:
    """Ideal S-shape: full-sine θ profile with amplitude 1.2 rad."""
    return build_ideal_from_edge_profile(
        THETA_PROFILES["S"], n, arc_length, amplitude=1.2
    )


# ---------------------------------------------------------------------------
# LOSS
# ---------------------------------------------------------------------------

def loss_fn(nodes: list, ps=None) -> dict:
    return profile_loss(nodes, "S")


# ---------------------------------------------------------------------------
# REGISTRATION
# ---------------------------------------------------------------------------

benchmark = Benchmark(
    name               = "S",
    physics_type       = "beam",
    target_description = (
        "An S-curve: curvature changes sign at the midpoint, one inflection point, "
        "equal bending in opposite directions, net total turning angle near zero."
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
    print(f"S-shape  |  error: {result['error_percent']:.2f}%  |  {result['diagnostics']}")
    plot_shape(nodes, name="S (ideal)", error_percent=result["error_percent"])


if __name__ == "__main__":
    main()
