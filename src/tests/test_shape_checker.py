# ---------------------------------------------------------------------------
# src/tests/test_shape_checker.py
# ---------------------------------------------------------------------------
# Tests for the θ-profile-based shape checker.
# Run from project root:
#   python -m src.tests.test_shape_checker
#
# Each ideal shape is generated with build_ideal_from_edge_profile, which uses
# exactly the same integration scheme as the evaluator.  This guarantees 0 %
# RMS error for all ideal shapes regardless of node count.
#
# Plot layout (one row per shape, two panels):
#   Left:  θ(s') profile — provided (relative tangent angles) vs. fitted ideal
#   Right: XY overlay   — provided nodes vs. ideal shape anchored to node 0
# ---------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from src.tools.shape_checker import (
    THETA_PROFILES,
    build_ideal_from_edge_profile,
    compute_shape_error,
)

ARC_LENGTH = 0.10   # beam length (m) — scale only, does not affect shape


# ---------------------------------------------------------------------------
# IDEAL SHAPE GENERATORS
# Each function calls build_ideal_from_edge_profile with the appropriate
# profile function and parameters.  Because the test uses the same θ-profile
# and integration as the evaluator, the round-trip error is exactly 0 %.
# ---------------------------------------------------------------------------

def create_c_shape(n: int = 9, total_turn: float = 0.8) -> list:
    """Constant-curvature arc (circular).  total_turn controls the bend angle."""
    return build_ideal_from_edge_profile(
        THETA_PROFILES["C"], n, ARC_LENGTH, total_turn=total_turn
    )


def create_u_shape(n: int = 9, total_turn: float = np.pi) -> list:
    """Semicircle.  total_turn = π by default (can be negative for CCW)."""
    return build_ideal_from_edge_profile(
        THETA_PROFILES["U"], n, ARC_LENGTH, total_turn=total_turn
    )


def create_s_shape(n: int = 9, amplitude: float = 0.8) -> list:
    """Half-sine θ profile: curves one way then back, net turn = 0."""
    return build_ideal_from_edge_profile(
        THETA_PROFILES["S"], n, ARC_LENGTH, amplitude=amplitude
    )


def create_l_shape(n: int = 9, total_turn: float = np.pi / 2, turn_at: float = 0.65) -> list:
    """Straight section (θ=0) followed by an abrupt turn at turn_at ∈ [0,1]."""
    return build_ideal_from_edge_profile(
        THETA_PROFILES["L"], n, ARC_LENGTH, total_turn=total_turn, turn_at=turn_at
    )


# ---------------------------------------------------------------------------
# PLOT  (two panels per shape)
# ---------------------------------------------------------------------------

def _extract_provided_theta(nodes: list):
    """Return (s_norm, dtheta) at edge midpoints, same convention as evaluator."""
    obs = np.array(nodes)
    N = len(obs)
    edge_vecs = np.diff(obs, axis=0)
    edge_lengths = np.linalg.norm(edge_vecs, axis=1)
    arc_length = float(np.sum(edge_lengths))

    s_nodes = np.zeros(N)
    s_nodes[1:] = np.cumsum(edge_lengths) / arc_length
    s_edges = (s_nodes[:-1] + s_nodes[1:]) / 2.0

    s0 = s_edges[0]
    span = s_edges[-1] - s0 + 1e-12
    s_norm = (s_edges - s0) / span

    theta_edges = np.arctan2(edge_vecs[:, 0], -edge_vecs[:, 1])
    theta_start = float(theta_edges[0])
    dtheta = (theta_edges - theta_start + np.pi) % (2 * np.pi) - np.pi
    return s_norm, dtheta


def plot_shape(
    nodes: list,
    target_shape: str,
    result: dict,
    ax_theta,
    ax_xy,
) -> None:
    """
    Two-panel plot for one shape.
      Left  (ax_theta): θ(s') — provided (blue circles) vs. fitted ideal (red dashed)
      Right (ax_xy):    XY   — provided (blue) vs. ideal anchored to node 0 (red dashed)
    """
    obs = np.array(nodes)
    ideal = np.array(result["ideal_nodes"])
    params = result["fitted_params"]
    err = result["error_percent"]

    # — θ profile —
    s_norm, dtheta_prov = _extract_provided_theta(nodes)
    dtheta_ideal = THETA_PROFILES[target_shape](s_norm, **params)

    ax_theta.plot(s_norm, dtheta_prov,  "bo",  label="provided", markersize=6)
    ax_theta.plot(s_norm, dtheta_ideal, "r--", label="ideal (fitted)", linewidth=2)
    ax_theta.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax_theta.set_xlabel("s'  (normalised arc-length)")
    ax_theta.set_ylabel("Δθ  (rad)")
    ax_theta.set_title(f"θ profile — {target_shape}")
    ax_theta.legend(fontsize=8)

    # — XY overlay —
    ax_xy.plot(obs[:, 0],   obs[:, 1],   "bo-",  linewidth=2, markersize=5, label="provided")
    ax_xy.plot(ideal[:, 0], ideal[:, 1], "r--o", linewidth=2, markersize=5, label="ideal")
    ax_xy.set_aspect("equal")
    ax_xy.set_xlabel("x (m)")
    ax_xy.set_ylabel("y (m)")
    ax_xy.invert_yaxis()
    ax_xy.set_title(
        f"{target_shape}:  {err:.1f} % error  "
        f"({result['confidence_percent']:.1f} % confidence)"
    )
    ax_xy.legend(fontsize=8)


# ---------------------------------------------------------------------------
# TESTS
# ---------------------------------------------------------------------------

def test_all_shapes() -> None:
    test_cases = [
        ("C", create_c_shape),
        ("U", create_u_shape),
        ("S", create_s_shape),
        ("L", create_l_shape),
    ]

    _, axes = plt.subplots(len(test_cases), 2, figsize=(13, 4 * len(test_cases)))

    print("=" * 60)
    print("SHAPE CHECKER — ideal shapes, expected error ≈ 0 %")
    print("=" * 60)

    all_pass = True
    for i, (target, fn) in enumerate(test_cases):
        nodes  = fn()
        result = compute_shape_error(nodes, target)
        err    = result["error_percent"]
        status = "✓ PASS" if err < 1.0 else "✗ FAIL"
        if err >= 1.0:
            all_pass = False
        print(f"  {target}: {err:.4f} %  {status}  — {result['diagnostics']}")
        plot_shape(nodes, target, result, axes[i, 0], axes[i, 1])

    print("=" * 60)
    print("All passed." if all_pass else "Some tests FAILED.")

    plt.suptitle("Shape checker — ideal shapes (expected ≈ 0 % error)", fontsize=13)
    plt.tight_layout()
    plt.show()


def test_perturbations() -> None:
    """Verify that degraded or wrong shapes register nonzero error."""
    print("\n" + "=" * 60)
    print("PERTURBATION TESTS")
    print("=" * 60)

    c_nodes = create_c_shape()

    # Add random noise to node positions
    rng = np.random.default_rng(42)
    noisy = np.array(c_nodes) + rng.normal(0, 0.003, np.array(c_nodes).shape)
    r = compute_shape_error(noisy.tolist(), "C")
    print(f"  C + position noise:   {r['error_percent']:.1f} %  (> 0 expected)")

    r2 = compute_shape_error(create_u_shape(), "C")
    print(f"  U tested as C:        {r2['error_percent']:.1f} %  (U has wrong total_turn for C)")

    r3 = compute_shape_error(create_s_shape(), "C")
    print(f"  S tested as C:        {r3['error_percent']:.1f} %  (S profile differs from linear ramp)")

    r4 = compute_shape_error(create_c_shape(), "L")
    print(f"  C tested as L:        {r4['error_percent']:.1f} %  (uniform κ ≠ concentrated κ)")

    r5 = compute_shape_error(create_l_shape(), "S")
    print(f"  L tested as S:        {r5['error_percent']:.1f} %  (step ≠ half-sine)")

    print("=" * 60)


if __name__ == "__main__":
    test_all_shapes()
    test_perturbations()
