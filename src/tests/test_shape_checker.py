"""
Profile-fitting round-trip tests.

Each ideal shape is generated with build_ideal_from_edge_profile, then fed
into profile_loss.  Because generation and evaluation use the same integration
scheme, round-trip error must be exactly 0%.

Run from project root:
    python -m src.tests.test_shape_checker
"""

import numpy as np
import matplotlib.pyplot as plt

from src.benchmarks.profile_fitting import build_ideal_from_edge_profile, profile_loss
from src.benchmarks.loss_beam.profile_C import profile_c as _profile_c, fit_c as _fit_c
from src.benchmarks.loss_beam.profile_U import profile_u as _profile_u, fit_u as _fit_u
from src.benchmarks.loss_beam.profile_S import profile_s as _profile_s, fit_s as _fit_s
from src.benchmarks.loss_beam.profile_L import profile_l as _profile_l, fit_l as _fit_l

ARC_LENGTH = 0.10


def create_c_shape(n: int = 9) -> list:
    return build_ideal_from_edge_profile(_profile_c, n, ARC_LENGTH, total_turn=np.pi / 2)


def create_u_shape(n: int = 9) -> list:
    return build_ideal_from_edge_profile(_profile_u, n, ARC_LENGTH, total_turn=np.pi)


def create_s_shape(n: int = 9) -> list:
    return build_ideal_from_edge_profile(_profile_s, n, ARC_LENGTH, amplitude=1.2)


def create_l_shape(n: int = 9) -> list:
    return build_ideal_from_edge_profile(_profile_l, n, ARC_LENGTH, total_turn=np.pi / 2, turn_at=0.65)


# ---------------------------------------------------------------------------
# PLOT HELPERS
# ---------------------------------------------------------------------------

def _extract_theta(nodes: list):
    obs = np.array(nodes)
    edge_vecs = np.diff(obs, axis=0)
    edge_lengths = np.linalg.norm(edge_vecs, axis=1)
    arc = float(np.sum(edge_lengths))
    s_nodes = np.zeros(len(obs))
    s_nodes[1:] = np.cumsum(edge_lengths) / arc
    s_edges = (s_nodes[:-1] + s_nodes[1:]) / 2.0
    s0, span = s_edges[0], s_edges[-1] - s_edges[0] + 1e-12
    s_norm = (s_edges - s0) / span
    theta_edges = np.arctan2(edge_vecs[:, 0], -edge_vecs[:, 1])
    dtheta = (theta_edges - float(theta_edges[0]) + np.pi) % (2 * np.pi) - np.pi
    return s_norm, dtheta


def _plot_case(nodes, profile_fn, result, label, ax_theta, ax_xy):
    obs   = np.array(nodes)
    ideal = np.array(result["ideal_nodes"])
    params = result["fitted_params"]
    err   = result["error_percent"]

    s_norm, dtheta_prov = _extract_theta(nodes)
    dtheta_ideal = profile_fn(s_norm, **params)

    ax_theta.plot(s_norm, dtheta_prov,  "bo",  markersize=6, label="provided")
    ax_theta.plot(s_norm, dtheta_ideal, "r--", linewidth=2,  label="ideal (fitted)")
    ax_theta.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax_theta.set_xlabel("s'")
    ax_theta.set_ylabel("Δθ (rad)")
    ax_theta.set_title(f"θ profile — {label}")
    ax_theta.legend(fontsize=8)

    ax_xy.plot(obs[:, 0],   obs[:, 1],   "bo-",  linewidth=2, markersize=5, label="provided")
    ax_xy.plot(ideal[:, 0], ideal[:, 1], "r--o", linewidth=2, markersize=5, label="ideal")
    ax_xy.set_aspect("equal")
    ax_xy.invert_yaxis()
    ax_xy.set_xlabel("x (m)")
    ax_xy.set_ylabel("y (m)")
    ax_xy.set_title(f"{label}: {err:.1f}% error")
    ax_xy.legend(fontsize=8)


# ---------------------------------------------------------------------------
# TESTS
# ---------------------------------------------------------------------------

def test_all_shapes() -> bool:
    cases = [
        ("C", create_c_shape, _profile_c, _fit_c),
        ("U", create_u_shape, _profile_u, _fit_u),
        ("S", create_s_shape, _profile_s, _fit_s),
        ("L", create_l_shape, _profile_l, _fit_l),
    ]

    _, axes = plt.subplots(len(cases), 2, figsize=(13, 4 * len(cases)))

    print("=" * 60)
    print("PROFILE-FITTING — ideal shapes, expected error ≈ 0 %")
    print("=" * 60)

    all_pass = True
    for i, (label, fn, pfn, fitfn) in enumerate(cases):
        nodes  = fn()
        result = profile_loss(nodes, pfn, fitfn)
        err    = result["error_percent"]
        status = "PASS" if err < 1.0 else "FAIL"
        if err >= 1.0:
            all_pass = False
        print(f"  {label}: {err:.4f} %  {status}  — {result['diagnostics']}")
        _plot_case(nodes, pfn, result, label, axes[i, 0], axes[i, 1])

    print("=" * 60)
    print("All passed." if all_pass else "Some tests FAILED.")

    plt.suptitle("Profile-fitting — ideal shapes (expected ≈ 0 % error)", fontsize=13)
    plt.tight_layout()
    plt.show()
    return all_pass


def test_perturbations() -> None:
    """Verify that degraded or wrong shapes register nonzero error."""
    print("\n" + "=" * 60)
    print("PERTURBATION TESTS")
    print("=" * 60)

    c_nodes = create_c_shape()
    rng = np.random.default_rng(42)
    noisy = (np.array(c_nodes) + rng.normal(0, 0.003, np.array(c_nodes).shape)).tolist()

    cases = [
        ("C + noise",    noisy,            _profile_c, _fit_c, "> 0 expected"),
        ("U tested as C", create_u_shape(), _profile_c, _fit_c, "wrong total_turn"),
        ("S tested as C", create_s_shape(), _profile_c, _fit_c, "sine ≠ ramp"),
        ("C tested as L", create_c_shape(), _profile_l, _fit_l, "uniform κ ≠ step"),
        ("L tested as S", create_l_shape(), _profile_s, _fit_s, "step ≠ sine"),
    ]

    for label, nodes, pfn, fitfn, note in cases:
        r = profile_loss(nodes, pfn, fitfn)
        print(f"  {label:<20}  {r['error_percent']:.1f}%   ({note})")

    print("=" * 60)


if __name__ == "__main__":
    test_all_shapes()
    test_perturbations()
