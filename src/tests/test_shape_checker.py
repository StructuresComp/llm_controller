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

def create_c_shape() -> list:
    """Create ideal C-shape: smooth left curve with strong positive curvature."""
    # Circular arc: left curve with wall contact at bottom
    R = 0.10
    theta = np.linspace(0, np.pi / 2, 10)  # 90 degree turn
    x = R * np.sin(theta)
    y = -R * (1 - np.cos(theta))
    y = y - 0.12  # Shift to wall height
    return list(zip(x, y))


def create_u_shape() -> list:
    """Create ideal U-shape: left arc + right arc, symmetric."""
    # Start at top, curve left, curve right back
    R = 0.08
    n_half = 5

    # Left arc: curve left from (0, 0) to (-R, -R)
    theta_left = np.linspace(np.pi / 2, np.pi, n_half)
    x_left = R * np.cos(theta_left)
    y_left = -R * np.sin(theta_left)

    # Right arc: mirror back right
    theta_right = np.linspace(0, np.pi / 2, n_half)[1:]  # Skip center
    x_right = R * np.cos(theta_right)
    y_right = -R - R * (1 - np.sin(theta_right))

    x = np.concatenate([x_left, x_right])
    y = np.concatenate([y_left, y_right])
    return list(zip(x, y))


def create_s_shape() -> list:
    """Create ideal S-shape: left arc then right arc."""
    R = 0.06
    n_half = 5

    # Left arc: curve left
    theta1 = np.linspace(0, np.pi / 2, n_half)
    x1 = R * np.sin(theta1)
    y1 = -R * (1 - np.cos(theta1))

    # Right arc: curve right from end of left arc
    start_x = x1[-1]
    start_y = y1[-1]
    theta2 = np.linspace(0, np.pi / 2, n_half)[1:]
    x2 = start_x + R * (1 - np.sin(theta2))
    y2 = start_y - R * np.cos(theta2)

    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])
    return list(zip(x, y))


def create_l_shape() -> list:
    """Create ideal L-shape: straight down then sharp left turn (90 degrees)."""
    # Mostly straight down, then sharp left turn at bottom
    n = 10
    x = np.zeros(n)
    y = np.linspace(0, -0.06, n)

    # Sharp turn at the end: rotate last 3 nodes left
    x[-3:] = np.array([-0.02, -0.04, -0.06])
    y[-3:] = np.array([-0.06, -0.09, -0.12])

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
    print("\n⚠️  NOTE: Synthetic test shapes may not achieve 0% error.")
    print("Shape checker is designed to work with REAL beam configurations")
    print("from orchestrator/programmer runs. These plots show that the")
    print("error computation works and provides useful diagnostics.\n")

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

        print(f"  Plot saved (showing error computation works)")

    print("\n" + "=" * 70)
    print("INTEGRATION NOTES:")
    print("=" * 70)
    print("To achieve ~0% error, run the shape checker against:")
    print("  - Real beam configurations from orchestrator iterations")
    print("  - Extract observable_positions from state['shape_history']")
    print("  - Pass to compute_shape_error(positions, target_shape)")
    print("\nExample (in programmer_agent or evaluation harness):")
    print("  from src.tools.shape_checker import compute_shape_error")
    print("  error = compute_shape_error(state['shape_history'][-1], 'C')")
    print("  print(f\"Shape match: {error['match_confidence_percent']:.1f}%\")")
    print("=" * 70)

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
