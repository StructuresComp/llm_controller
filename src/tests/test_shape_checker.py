# ---------------------------------------------------------------------------
# src/tests/test_shape_checker.py
# ---------------------------------------------------------------------------
# Tests for the DDG-based shape checker.
# Run from project root:
#   python -m src.tests.test_shape_checker
#
# Each ideal shape is built by integrating a κ(s) profile from scratch,
# then verified against its own target — expected error: 0 %.
#
# Plot layout: one row per shape, two panels each —
#   Left:  turning-angle profile κ_i (provided vs. ideal)
#   Right: XY overlay (provided vs. ideal anchored to node 0)
# ---------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from src.tools.shape_checker import (
    extract_turning_angles,
    integrate_kappa,
    compute_shape_error,
)

ARC_LENGTH = 0.10   # beam length (m) — only affects scale, not shape


# ---------------------------------------------------------------------------
# SHAPE GENERATORS
# Each function builds nodes by integrating a specific κ(s) profile.
# ---------------------------------------------------------------------------

def create_c_shape(n: int = 7, total_turn: float = 0.8) -> list:
    """
    C-shape: constant curvature (circular arc).
    κ_i = total_turn / (n−2)  for all interior nodes.
    """
    N_int = n - 2
    kappa = np.full(N_int, total_turn / N_int)
    return integrate_kappa(kappa, ARC_LENGTH)


def create_u_shape(n: int = 7) -> list:
    """
    U-shape: constant curvature, total turn = π (semicircle).
    Identical profile to C but larger total turning angle.
    """
    return create_c_shape(n, total_turn=np.pi)


def create_s_shape(n: int = 7, amp: float = 0.5) -> list:
    """
    S-shape: antisymmetric κ profile, net total turn = 0.
    First half of interior nodes: +amp. Second half: −amp.
    Middle node (odd N_int): κ = 0 (inflection point).
    Constructed explicitly to avoid floating-point sign(sin(π)) ≠ 0 issues.
    """
    N_int = n - 2
    mid = N_int // 2
    kappa = np.zeros(N_int)
    kappa[:mid] = amp
    kappa[N_int - mid:] = -amp
    return integrate_kappa(kappa, ARC_LENGTH)


def create_l_shape(n: int = 7, total_turn: float = 1.3, turn_at: float = 0.7) -> list:
    """
    L-shape: curvature concentrated at one interior node.
    turn_at ∈ [0,1] controls where along the beam the bend occurs.
    """
    N_int = n - 2
    i_turn = int(round(turn_at * (N_int - 1)))
    kappa = np.zeros(N_int)
    kappa[i_turn] = total_turn
    return integrate_kappa(kappa, ARC_LENGTH)


# ---------------------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------------------

def plot_shape(nodes: list, target_shape: str, result: dict,
               ax_k, ax_xy) -> None:
    """
    Two-panel plot for one shape.
      Left  (ax_k):  κ_i values — provided (blue) vs. ideal (red dashed)
      Right (ax_xy): XY positions — provided (blue) vs. ideal overlay (red dashed)
    """
    kp = np.array(result["kappa_provided"])
    ki = np.array(result["kappa_ideal"])
    ideal = np.array(result["ideal_nodes"])
    obs   = np.array(nodes)

    idx = np.arange(len(kp))

    # — Left: κ profile —
    ax_k.plot(idx, kp, "bo-", label="provided", markersize=5)
    ax_k.plot(idx, ki, "r--s", label="ideal", markersize=5)
    ax_k.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax_k.set_xlabel("interior node index")
    ax_k.set_ylabel("κ (rad)")
    ax_k.set_title(f"κ profile — target: {target_shape}")
    ax_k.legend(fontsize=8)

    # — Right: XY overlay —
    ax_xy.plot(obs[:, 0],   obs[:, 1],   "bo-",  linewidth=2, markersize=5, label="provided")
    ax_xy.plot(ideal[:, 0], ideal[:, 1], "r--o", linewidth=2, markersize=5, label="ideal")
    ax_xy.set_aspect("equal")
    ax_xy.set_xlabel("x (m)")
    ax_xy.set_ylabel("y (m)")
    ax_xy.invert_yaxis()
    ax_xy.set_title(
        f"{target_shape}: {result['error_percent']:.1f}% error  "
        f"({result['confidence_percent']:.1f}% confidence)"
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

    _, axes = plt.subplots(len(test_cases), 2, figsize=(12, 4 * len(test_cases)))

    print("=" * 60)
    print("SHAPE CHECKER — ideal shapes, expected error = 0 %")
    print("=" * 60)

    all_pass = True
    for i, (target, fn) in enumerate(test_cases):
        nodes  = fn()
        result = compute_shape_error(nodes, target)
        err    = result["error_percent"]
        status = "✓ PASS" if err < 1.0 else "✗ FAIL"
        if err >= 1.0:
            all_pass = False
        print(f"  {target}: {err:.2f}%  {status}  — {result['diagnostics']}")
        plot_shape(nodes, target, result, axes[i, 0], axes[i, 1])

    print("=" * 60)
    print("All passed." if all_pass else "Some tests FAILED.")

    plt.suptitle("Shape Checker — ideal shapes (0 % error expected)", fontsize=13)
    plt.tight_layout()
    plt.show()


def test_perturbations() -> None:
    """Verify that non-ideal shapes register nonzero error."""
    print("\n" + "=" * 60)
    print("PERTURBATION TESTS")
    print("=" * 60)

    c_nodes = create_c_shape()

    # Add noise to κ
    kappa_noisy = extract_turning_angles(c_nodes) + np.random.normal(0, 0.05, 5)
    obs = np.array(c_nodes)
    arc = float(np.sum(np.sqrt(np.sum(np.diff(obs, axis=0)**2, axis=1))))
    noisy_nodes = integrate_kappa(
        kappa_noisy, arc,
        start_pos=(obs[0, 0], obs[0, 1]),
        start_angle=float(np.arctan2(obs[1,0]-obs[0,0], -(obs[1,1]-obs[0,1]))),
    )
    r = compute_shape_error(noisy_nodes, "C")
    print(f"  C + noise:          {r['error_percent']:.1f}%  (> 0 expected)")

    r2 = compute_shape_error(create_u_shape(), "C")
    print(f"  U tested as C:      {r2['error_percent']:.1f}%  (> 0 for U-vs-C angle)")

    r3 = compute_shape_error(create_s_shape(), "C")
    print(f"  S tested as C:      {r3['error_percent']:.1f}%  (should be high)")

    r4 = compute_shape_error(create_c_shape(), "L")
    print(f"  C tested as L:      {r4['error_percent']:.1f}%  (should be high)")

    print("=" * 60)


if __name__ == "__main__":
    test_all_shapes()
    test_perturbations()
