"""
src/benchmarks/profile_fitting.py
----------------------------------
Shared math for θ-profile benchmark cases (C, S, U, L, and any future shape
that is described by a tangent-angle curve θ*(s)).

CONTRACT
--------
A θ-profile case defines two things in its shapes/ file:

    profile_fn(s_norm, **params) -> ndarray
        Ideal relative turning angle at each normalised arc-length position.
        s_norm ∈ [0, 1].  θ*(0) = 0 by convention (relative to start tangent).

    fit_fn(s_norm, dtheta) -> dict
        Fit the free parameters of profile_fn to observed turning angles.
        Returns a dict whose keys match profile_fn's **params.

These two functions are then passed to build_ideal_from_edge_profile (for
reference_fn) and profile_loss (for loss_fn) — both defined here.

WHAT LIVES HERE
---------------
    build_ideal_from_edge_profile   generate reference nodes from a θ*(s) profile
    profile_loss                    fit profile to observed nodes, return RMS error dict

Cases that do not use a θ-profile (tip-reach, navigation) import nothing from
this file — they compute their loss directly in the case file.
"""

import numpy as np
from typing import Callable, List, Tuple
from scipy.interpolate import CubicSpline


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
    Generate n reference nodes by integrating profile_fn at edge midpoints.

    Uses edge midpoints (not node positions) so that profile_loss can do
    an exact round-trip: feed the output back in and get 0% error.

    theta = 0 means the beam points straight down (−Y).
    x0, y0 anchor the ideal shape at node 0's actual position.
    """
    ds = arc_length / (n - 1)
    s_nodes = np.linspace(0.0, 1.0, n)
    s_edges = (s_nodes[:-1] + s_nodes[1:]) / 2.0

    # Normalise to [0, 1] over the edge range (not node range) for profile_fn.
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


def _fit_one_dir(
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

    # Arc-length parameterisation at edge midpoints.
    s_nodes = np.zeros(N)
    s_nodes[1:] = np.cumsum(edge_lengths) / arc_length
    s_edges = (s_nodes[:-1] + s_nodes[1:]) / 2.0
    s0, span = s_edges[0], s_edges[-1] - s_edges[0] + 1e-12
    s_norm = (s_edges - s0) / span

    # Observed tangent angles at each edge; strip the absolute start angle
    # so dtheta is purely the relative turning (what profile_fn models).
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

    error_percent = 0   → nodes lie exactly on the ideal profile.
    error_percent = 100 → error equals the full arc length (maximally wrong).

    Returns: error_percent, confidence_percent, ideal_nodes,
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

    ideal, rms, params = _fit_one_dir(obs, profile_fn, fit_fn)
    error_percent = min(rms * 100.0, 100.0)
    diag = (
        "Shape matches profile well." if error_percent < 2.0 else
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


def make_profile_from_nodes(nodes: list) -> Tuple[Callable, Callable]:
    """
    Create a turning-angle profile from a sequence of (x, y) nodes.

    Extracts relative turning angles (change in direction at each arc-length)
    and interpolates with cubic spline. Follows the same convention as C, S, U, L:
    - First two nodes define initial position and slope
    - Profile encodes relative turning angles (dtheta = angle - first_angle)
    - Can be used directly with profile_loss and build_ideal_from_edge_profile

    Args:
        nodes: list of (x, y) tuples defining the target shape

    Returns:
        (profile_fn, fit_fn) tuple matching the standard profile API:
        - profile_fn(s, **params) → relative turning angle at arc-length s ∈ [0, 1]
        - fit_fn(s, dtheta) → {} (no-op; profile is fully specified)
    """
    nodes_array = np.array(nodes, dtype=float)
    if len(nodes_array) < 3:
        raise ValueError("Need at least 3 nodes to compute curvature profile")

    # Arc-length parameterization
    edge_vecs = np.diff(nodes_array, axis=0)
    edge_lengths = np.linalg.norm(edge_vecs, axis=1)
    total_arc = float(np.sum(edge_lengths))

    if total_arc < 1e-10:
        raise ValueError("Nodes are too close; total arc length is near zero")

    # Normalize arc-length to [0, 1]
    s_nodes = np.zeros(len(nodes_array))
    s_nodes[1:] = np.cumsum(edge_lengths) / total_arc
    s_edges = (s_nodes[:-1] + s_nodes[1:]) / 2.0

    # Compute absolute tangent angles at each edge (like theta_edges in _fit_one_dir)
    theta_edges = np.arctan2(edge_vecs[:, 0], -edge_vecs[:, 1])

    # Extract relative turning angles from the first edge (like dtheta in _fit_one_dir)
    theta_start = float(theta_edges[0])
    dtheta = (theta_edges - theta_start + np.pi) % (2 * np.pi) - np.pi

    # Normalize arc-length for the profile to [0, 1] over edge range
    s0, span = s_edges[0], s_edges[-1] - s_edges[0] + 1e-12
    s_norm_edges = (s_edges - s0) / span

    # Create cubic spline from relative turning angles
    cs = CubicSpline(s_norm_edges, dtheta, bc_type='natural', extrapolate='extrapolate')

    def profile_fn(s_norm: np.ndarray, **_params) -> np.ndarray:
        """Evaluate turning-angle profile at normalized arc-length s ∈ [0, 1]."""
        return np.asarray(cs(s_norm), dtype=float)

    def fit_fn(_s_norm: np.ndarray, _dtheta: np.ndarray) -> dict:
        """No-op: profile is fully specified by the input nodes."""
        return {}

    return profile_fn, fit_fn
