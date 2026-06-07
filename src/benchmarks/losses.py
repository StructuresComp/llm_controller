"""
Loss machinery for benchmark cases.

    extract_turning_angles        — discrete curvature κ_i at each interior node
    build_ideal_from_edge_profile — generate reference nodes from a θ*(s) profile
    profile_loss                  — fit profile to observed nodes, return RMS error dict

Never add case-specific logic here.
"""

import numpy as np
from typing import Callable, List, Tuple


# ---------------------------------------------------------------------------
# GEOMETRIC PRIMITIVE
# ---------------------------------------------------------------------------

def extract_turning_angles(nodes: list) -> np.ndarray:
    """Discrete curvature κ_i = θ_{i+1} − θ_i at each interior node. Length N-2."""
    obs = np.array(nodes)
    if len(obs) < 3:
        return np.array([])
    theta = np.arctan2(np.diff(obs[:, 0]), -np.diff(obs[:, 1]))
    kappa = np.diff(theta)
    return (kappa + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# PROFILE-FITTING  (beam θ-profile cases only)
#
# The ideal shape is described by a tangent-angle profile θ*(s) where
# s ∈ [0, 1] is normalised arc-length.  Both generation and evaluation use
# edge-midpoint integration so round-trip error is exactly 0%.
# ---------------------------------------------------------------------------

def build_ideal_from_edge_profile(
    profile_fn: Callable,
    n: int,
    arc_length: float,
    x0: float = 0.0,
    y0: float = 0.0,
    theta: float = 0.0,
    **params,
) -> List[Tuple[float, float]]:
    """
    Generate n nodes by integrating profile_fn at edge midpoints.

    profile_fn(s_norm, **params) -> relative tangent angles from start direction.
    s_norm in [0,1].  theta=0 means pointing down (-Y).
    x0, y0 are the position of node 0 (anchor = actual first node of q).
    """
    ds = arc_length / (n - 1)
    s_nodes = np.linspace(0.0, 1.0, n)
    s_edges = (s_nodes[:-1] + s_nodes[1:]) / 2.0
    s0, span = s_edges[0], s_edges[-1] - s_edges[0] + 1e-12
    s_norm = (s_edges - s0) / span

    theta_edges = theta + profile_fn(s_norm, **params)
    nodes = [(float(x0), float(y0))]
    x, y = float(x0), float(y0)
    for te in theta_edges:
        x += ds * float(np.sin(te))
        y -= ds * float(np.cos(te))
        nodes.append((x, y))
    return nodes


def _build_ideal_one_dir(
    obs: np.ndarray,
    profile_fn: Callable,
    fit_fn: Callable,
) -> Tuple[list, float, dict]:
    """Fit profile_fn to obs; reconstruct ideal shape. Returns (ideal, rms_normalised, params)."""
    N = len(obs)
    edge_vecs = np.diff(obs, axis=0)
    edge_lengths = np.linalg.norm(edge_vecs, axis=1)
    arc_length = float(np.sum(edge_lengths))
    if arc_length < 1e-10 or N < 3:
        return obs.tolist(), 0.0, {}

    s_nodes = np.zeros(N)
    s_nodes[1:] = np.cumsum(edge_lengths) / arc_length
    s_edges = (s_nodes[:-1] + s_nodes[1:]) / 2.0
    s0, span = s_edges[0], s_edges[-1] - s_edges[0] + 1e-12
    s_norm = (s_edges - s0) / span

    theta_edges = np.arctan2(edge_vecs[:, 0], -edge_vecs[:, 1])
    theta_start = float(theta_edges[0])
    dtheta = (theta_edges - theta_start + np.pi) % (2 * np.pi) - np.pi

    params = fit_fn(s_norm, dtheta)
    theta_ideal = theta_start + profile_fn(s_norm, **params)

    ideal = np.empty((N, 2))
    ideal[0] = obs[0]
    for i in range(N - 1):
        ds = float(edge_lengths[i])
        ideal[i + 1, 0] = ideal[i, 0] + ds * np.sin(float(theta_ideal[i]))
        ideal[i + 1, 1] = ideal[i, 1] - ds * np.cos(float(theta_ideal[i]))

    rms = float(np.sqrt(np.mean(np.sum((ideal - obs) ** 2, axis=1)))) / arc_length
    return ideal.tolist(), rms, params


def profile_loss(nodes: list, profile_fn: Callable, fit_fn: Callable) -> dict:
    """
    Fit profile_fn to observed nodes and return normalised RMS error.

    profile_fn and fit_fn are defined in the case file — this function has no
    knowledge of what shape is being evaluated.

    Returns dict with error_percent, confidence_percent, ideal_nodes,
    rms_normalized, fitted_params, diagnostics.
    """
    obs = np.array(nodes)
    if len(obs) < 3:
        return {
            "error_percent":      100.0,
            "confidence_percent": 0.0,
            "ideal_nodes":        [],
            "rms_normalized":     1.0,
            "fitted_params":      {},
            "diagnostics":        "Need at least 3 nodes.",
        }
    ideal, rms, params = _build_ideal_one_dir(obs, profile_fn, fit_fn)
    error_percent = min(rms * 100.0, 100.0)
    diag = "Shape matches profile well." if error_percent < 2.0 else (
        f"Fit params: { {k: f'{v:.3f}' for k, v in params.items()} }  →  {error_percent:.1f}% error."
    )
    return {
        "error_percent":      error_percent,
        "confidence_percent": 100.0 - error_percent,
        "ideal_nodes":        ideal,
        "rms_normalized":     rms,
        "fitted_params":      params,
        "diagnostics":        diag,
    }


