"""
Test scale invariance via _fit_one_dir alignment.

Three shapes per test:
(1) SOURCE SHAPE — extract profile from this
(2) OBSERVED SHAPE — the shape we evaluate (different size, noisy)
(3) ALIGNED REFERENCE — reconstructed from profile using:
    - Profile's turning angles (from source)
    - Observed shape's edge lengths
    - Anchored at observed shape's first node/angle

Loss = distance between (2) and (3). Scale invariance: same source shape
at different observed sizes gives similar errors.

Run from project root:
    python -m src.tests.test_profile_from_nodes
"""

import numpy as np
import matplotlib.pyplot as plt

from src.benchmarks.profile_fitting import (
    make_profile_from_nodes,
    profile_loss,
)


def test_profile_from_nodes():
    """
    Test scale invariance: same source profile at different observed sizes.
    """
    print("\n" + "="*70)
    print("TEST: Scale Invariance via Profile Alignment")
    print("="*70)

    def generate_sine_nodes(arc_length, num_nodes, amplitude=0.3, noise_std=0.0):
        """Generate sine wave nodes with given arc-length."""
        t = np.linspace(0, np.pi, num_nodes)
        noise = np.random.normal(0, noise_std, num_nodes)
        return [(x * arc_length / np.pi, amplitude * np.sin(x) + noise[i]) for i, x in enumerate(t)]

    # ============================================================
    # (1) SOURCE SHAPE — Extract profile
    # ============================================================
    arc_src = 1.0
    source_shape = generate_sine_nodes(arc_src, num_nodes=12, amplitude=0.3)
    print(f"\n(1) SOURCE SHAPE — Profile Source")
    print(f"    Arc-length: {arc_src:.2f} m")
    print(f"    Nodes: {len(source_shape)}")

    profile_fn, fit_fn = make_profile_from_nodes(source_shape)
    print(f"    ✓ Profile extracted (turning angles)")

    # ============================================================
    # TEST 1: Small observed shape
    # ============================================================
    arc_obs1 = 0.6
    observed_1 = generate_sine_nodes(arc_obs1, num_nodes=10, amplitude=0.3, noise_std=0.015)
    print(f"\n[TEST 1: SMALL OBSERVED SHAPE]")
    print(f"  (2) OBSERVED SHAPE — Arc-length {arc_obs1:.2f} m")
    print(f"      Nodes: {len(observed_1)}")

    result_1 = profile_loss(observed_1, profile_fn, fit_fn)
    error_1 = result_1["error_percent"]
    aligned_1 = result_1["ideal_nodes"]
    print(f"  (3) ALIGNED REFERENCE — Reconstructed from profile")
    print(f"      Nodes: {len(aligned_1)}")
    print(f"  ✓ Loss: {error_1:.2f}%")

    # ============================================================
    # TEST 2: Large observed shape
    # ============================================================
    arc_obs2 = 2.8
    observed_2 = generate_sine_nodes(arc_obs2, num_nodes=20, amplitude=0.3, noise_std=0.015)
    print(f"\n[TEST 2: LARGE OBSERVED SHAPE]")
    print(f"  (2) OBSERVED SHAPE — Arc-length {arc_obs2:.2f} m ({arc_obs2/arc_obs1:.1f}x larger)")
    print(f"      Nodes: {len(observed_2)}")

    result_2 = profile_loss(observed_2, profile_fn, fit_fn)
    error_2 = result_2["error_percent"]
    aligned_2 = result_2["ideal_nodes"]
    print(f"  (3) ALIGNED REFERENCE — Reconstructed from profile")
    print(f"      Nodes: {len(aligned_2)}")
    print(f"  ✓ Loss: {error_2:.2f}%")

    # ============================================================
    # SCALE INVARIANCE CHECK
    # ============================================================
    print(f"\n[SCALE INVARIANCE CHECK]")
    print(f"  Test 1 (arc {arc_obs1:.2f}m): error {error_1:.2f}%")
    print(f"  Test 2 (arc {arc_obs2:.2f}m): error {error_2:.2f}%")
    print(f"  Size ratio: {arc_obs2/arc_obs1:.1f}x")
    print(f"  Error difference: {abs(error_1 - error_2):.2f}%")
    if abs(error_1 - error_2) < 5.0:
        print(f"  ✓ PASS: Errors within 5% (scale-invariant)")
    else:
        print(f"  ⚠ WARNING: Large error difference")

    # ============================================================
    # PLOT: Left = source & observed (raw). Right = observed & aligned.
    # ============================================================
    print(f"\nPlotting...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: Source and observed shapes (raw)
    src_x = [n[0] for n in source_shape]
    src_y = [n[1] for n in source_shape]
    obs2_x = [n[0] for n in observed_2]
    obs2_y = [n[1] for n in observed_2]

    ax1.plot(src_x, src_y, "g-o", linewidth=2, markersize=6, label=f"(1) Source ({arc_src:.2f}m)")
    ax1.plot(obs2_x, obs2_y, "b-^", linewidth=2, markersize=6, label=f"(2) Observed ({arc_obs2:.2f}m)")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_title("Source and Observed Shapes (Raw)")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal")

    # Right panel: Observed and aligned reference
    align2_x = [n[0] for n in aligned_2]
    align2_y = [n[1] for n in aligned_2]

    ax2.plot(obs2_x, obs2_y, "b-o", linewidth=2.5, markersize=7, label=f"(2) Observed (err={error_2:.1f}%)")
    ax2.plot(align2_x, align2_y, "r--^", linewidth=2, markersize=6, label=f"(3) Aligned Reference", alpha=0.8)
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.set_title("Profile Alignment")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal")

    plt.tight_layout()
    plt.savefig("/tmp/test_profile_from_nodes.png", dpi=100)
    print(f"✓ Plot saved: /tmp/test_profile_from_nodes.png")
    plt.show()

    print(f"\n{'='*70}")
    print(f"TEST COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    test_profile_from_nodes()
