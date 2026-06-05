# ---------------------------------------------------------------------------
# src/tests/test_shape_checker.py
# ---------------------------------------------------------------------------
# Comprehensive test suite for shape_checker.py with visual plots.
# Run from project root:
#   python -m src.tests.test_shape_checker
# ---------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from src.tools.shape_checker import compute_shape_error, compute_shape_signature, SHAPE_SPECS


# Synthetic reference shapes (observable nodes, simulated from ideal beam deformations)

WALL_H = -0.12  # must match SHAPE_SPECS / config


def create_c_shape() -> list:
    """
    Ideal C-shape: arm at top, beam curves RIGHT to wall.

    Uses x = R*(1-cos φ), y = -R*sin φ so the beam starts pointing
    straight down (φ=0) and rotates clockwise to pointing right (φ=π/2).
    Tangent angle = φ (increases) → curvature signs all +1.
    Last node lands exactly at y = WALL_H.
    """
    R = 0.15
    phi_end = np.arcsin(abs(WALL_H) / R)          # node N4 lands at y = WALL_H
    phi = np.linspace(0, phi_end, 5)
    x = R * (1 - np.cos(phi))
    y = -R * np.sin(phi)
    return list(zip(x, y))


def create_u_shape() -> list:
    """
    Ideal U-shape: symmetric semicircle, one open end to the other through the wall.

    Uses x = R*cos(φ), y = -R*sin(φ), φ: 0 → π.
    - N0: right arm top at (R, 0)
    - N3: bottom centre at (0, -R) — exactly at wall when R = |WALL_H|
    - N6: left arm top at (-R, 0)
    Tangent angle = -φ (strictly decreasing) → curvature signs all -1.
    Perfectly symmetric around x = 0.
    """
    R = abs(WALL_H)                                 # = 0.12: bottom node lands at wall
    phi = np.linspace(0, np.pi, 7)
    x = R * np.cos(phi)
    y = -R * np.sin(phi)
    return list(zip(x, y))


def create_s_shape() -> list:
    """
    Ideal S-shape: beam curves right then immediately curves left.

    First arc: +1 curvature (right).
    Second arc: -1 curvature (left), joined continuously.
    Net turning angle ≈ 0 (shape doubles back).
    """
    R = 0.06
    phi_arc = np.pi / 3      # 60-degree arc each half

    # Right-curving arc (positive curvature)
    phi1 = np.linspace(0, phi_arc, 4)
    x1 = R * (1 - np.cos(phi1))
    y1 = -R * np.sin(phi1)

    # Left-curving arc (negative curvature) — continues from end of first arc
    ex, ey = x1[-1], y1[-1]
    phi2 = np.linspace(phi_arc, 0, 4)[1:]          # reverse: unwinds the turn
    x2 = ex + R * (np.cos(phi2) - np.cos(phi_arc))
    y2 = ey - R * (np.sin(phi_arc) - np.sin(phi2))

    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])
    return list(zip(x, y))


def create_l_shape() -> list:
    """
    Ideal L-shape: mostly straight down, then tight right-angle turn at the bottom.

    Straight section: arm at (0,0) going down to (0, -0.04).
    Arc section: R=0.08 circle, curving right from (0, -0.04) to (0.08, -0.12).
    Curvature signs: [0, +1, +1, +1] — zeros in straight part, +1 at the bend.
    Last node lands exactly at wall.
    """
    R = 0.08
    y_join = -0.04

    x_str = np.array([0.0, 0.0, 0.0])
    y_str = np.linspace(0.0, y_join, 3)

    phi_arc = np.linspace(0, np.pi / 2, 4)
    x_arc = R * (1 - np.cos(phi_arc))
    y_arc = y_join - R * np.sin(phi_arc)

    x = np.concatenate([x_str, x_arc[1:]])
    y = np.concatenate([y_str, y_arc[1:]])
    return list(zip(x, y))


def plot_shape(observable_positions: list, target_shape: str, error_result: dict, wall_h: float = -0.12, title_suffix: str = ""):
    """
    Plot a shape with visual annotations.

    Shows:
      - Beam configuration (nodes and connections)
      - Wall position
      - Tangent angles at each node
      - Curvature signs between nodes
      - Error metrics and diagnostics
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    obs = np.array(observable_positions)

    # Draw wall
    wall_width = 0.2
    wall_rect = Rectangle((-wall_width / 2, wall_h - 0.01), wall_width, 0.01, color="brown", alpha=0.7, label="Wall")
    ax.add_patch(wall_rect)
    ax.axhline(y=wall_h, color="brown", linestyle="--", alpha=0.5)

    # Draw beam nodes and connections
    ax.plot(obs[:, 0], obs[:, 1], "b-o", linewidth=2, markersize=8, label="Beam nodes")

    # Annotate nodes
    for i, (x, y) in enumerate(obs):
        ax.text(x + 0.005, y + 0.005, f"N{i}", fontsize=9, ha="left")

    # Draw tangent vectors at each node
    sig = error_result.get("signature", {})
    if sig and "tangent_angles" in sig:
        tangents = sig["tangent_angles"]
        arrow_len = 0.02
        for i, (x, y) in enumerate(obs[:-1]):
            angle = tangents[i]
            dx = arrow_len * np.sin(angle)
            dy = -arrow_len * np.cos(angle)
            ax.arrow(x, y, dx, dy, head_width=0.005, head_length=0.003, fc="green", ec="green", alpha=0.6)

    # Annotate curvature signs between nodes
    if sig and "curvature_signs" in sig:
        curvature = sig["curvature_signs"]
        for i, curve_sign in enumerate(curvature):
            x_mid = (obs[i, 0] + obs[i + 1, 0]) / 2
            y_mid = (obs[i, 1] + obs[i + 1, 1]) / 2
            label = "↻" if curve_sign > 0 else ("↺" if curve_sign < 0 else "→")
            ax.text(x_mid + 0.01, y_mid, label, fontsize=12, color="red", fontweight="bold")

    # Mark nodes touching wall
    wall_tol = 0.005
    touching_wall = np.abs(obs[:, 1] - wall_h) < wall_tol
    if np.any(touching_wall):
        ax.scatter(obs[touching_wall, 0], obs[touching_wall, 1], color="orange", s=100, marker="*", zorder=5, label="Wall contact")

    # Build title and info text
    overall_error = error_result.get("overall_error_percent", 100)
    confidence = error_result.get("match_confidence_percent", 0)
    breakdown = error_result.get("breakdown", {})
    diagnostics = error_result.get("diagnostics", "")

    title = f"Target: {target_shape} — Error: {overall_error:.1f}% | Confidence: {confidence:.1f}% {title_suffix}"
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Info text box
    info_text = (
        f"Error breakdown:\n"
        f"  Curvature: {breakdown.get('curvature', 0):.1f}%\n"
        f"  Angle: {breakdown.get('turning_angle', 0):.1f}%\n"
        f"  Contact: {breakdown.get('wall_contact', 0):.1f}%\n"
        f"  Symmetry: {breakdown.get('symmetry', 0):.1f}%\n\n"
        f"Diagnostics:\n{diagnostics}"
    )
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8), family="monospace")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # Y-axis pointing down

    return fig


def test_all_shapes():
    """Test all shape types and generate plots."""
    test_cases = [
        ("C", create_c_shape),
        ("U", create_u_shape),
        ("S", create_s_shape),
        ("L", create_l_shape),
    ]

    print("=" * 70)
    print("SHAPE CHECKER TEST SUITE")
    print("=" * 70)
    print("\nIdeal synthetic shapes — all should score 0.00% error.\n")

    figs = []

    for target_shape, shape_fn in test_cases:
        print(f"\nTesting {target_shape}-shape:")
        print("-" * 70)

        obs_positions = shape_fn()
        print(f"  Observable nodes: {len(obs_positions)}")
        for i, (x, y) in enumerate(obs_positions):
            print(f"    N{i}: ({x:+.6f}, {y:+.6f})")

        # Compute error
        error_result = compute_shape_error(obs_positions, target_shape)

        print(f"  Overall error: {error_result['overall_error_percent']:.2f}%")
        print(f"  Match confidence: {error_result['match_confidence_percent']:.2f}%")
        print(f"  Breakdown:")
        for key, val in error_result["breakdown"].items():
            print(f"    {key}: {val:.2f}%")
        print(f"  Diagnostics: {error_result['diagnostics']}")

        # Plot
        fig = plot_shape(obs_positions, target_shape, error_result, title_suffix=f"(Synthetic {target_shape})")
        figs.append(fig)

        expected_max = 1.0
        status = "✓ PASS" if error_result["overall_error_percent"] <= expected_max else "✗ FAIL"
        print(f"  {status}")

    print("\nDisplaying plots...")
    plt.tight_layout()
    plt.show()


def test_perturbed_shapes():
    """Test shapes with perturbations to verify error sensitivity."""
    print("\n" + "=" * 70)
    print("PERTURBATION TEST: How does error change with deviations?")
    print("=" * 70)

    base_shape = create_c_shape()
    print(f"\nBase C-shape: {len(base_shape)} nodes")

    # Test 1: Slight wall contact reduction
    perturbed1 = list(base_shape)
    perturbed1[-1] = (perturbed1[-1][0], -0.11)  # Lift free end away from wall
    error1 = compute_shape_error(perturbed1, "C")
    print(f"\nPerturbation 1 (reduced wall contact):")
    print(f"  Error: {error1['overall_error_percent']:.2f}%")
    print(f"  Diagnostics: {error1['diagnostics']}")

    # Test 2: Reduce curvature
    perturbed2 = [(x * 0.5, y) for x, y in base_shape]
    error2 = compute_shape_error(perturbed2, "C")
    print(f"\nPerturbation 2 (reduced curvature):")
    print(f"  Error: {error2['overall_error_percent']:.2f}%")
    print(f"  Diagnostics: {error2['diagnostics']}")

    # Test 3: Wrong shape (U instead of C)
    u_shape = create_u_shape()
    error3 = compute_shape_error(u_shape, "C")
    print(f"\nPerturbation 3 (U-shape tested as C-shape):")
    print(f"  Error: {error3['overall_error_percent']:.2f}%")
    print(f"  Diagnostics: {error3['diagnostics']}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_all_shapes()
    test_perturbed_shapes()
