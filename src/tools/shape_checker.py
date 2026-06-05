# ---------------------------------------------------------------------------
# src/tools/shape_checker.py
# ---------------------------------------------------------------------------
# Deterministic shape validator (DDG-informed).
#
# Core principle
# --------------
# A beam shape is fully described by how its tangent angle θ changes along
# arc-length s ∈ [0, 1].  Each target shape defines an *ideal θ profile*
# θ*(s', params) where s' is normalised arc-length and θ*(0) ≡ 0 (relative
# to the start tangent).  We fit the free parameters to the provided shape,
# reconstruct the ideal geometry in the same coordinate frame, and report
# the normalised RMS node distance as the error.
#
# This is invariant to rigid-body translation and rotation by construction.
# A CONFIG flag enables reflection checking (swap which end is s=0).
#
# Consistency rule
# ----------------
# Both shape generation (test file) and ideal reconstruction use the same
# integration scheme: one step per edge, edge direction = θ*(s_edge, params).
# This guarantees exact round-trip (0 % error) for ideal test shapes.
#
# Section order:
#   1. Imports
#   2. Primitives          — extract_turning_angles, integrate_kappa (kept for LLM descriptor)
#   3. Theta profiles      — θ*(s', **params) functions and param-fitting functions
#   4. Build ideal         — build_ideal_from_edge_profile, _build_ideal_one_dir
#   5. Public API          — compute_shape_error, reconstruct_ideal
#   6. Orchestrator        — compute_shape_signature (text descriptor for LLM prompt)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 1. IMPORTS
# ---------------------------------------------------------------------------
import numpy as np
from typing import List, Tuple, Dict, Callable


# ---------------------------------------------------------------------------
# 2. PRIMITIVES  (kept for the LLM geometric descriptor)
# ---------------------------------------------------------------------------

def extract_turning_angles(nodes: List[Tuple[float, float]]) -> np.ndarray:
    """
    Discrete curvature κ_i = θ_{i+1} − θ_i at each interior node.
    Invariant to translation and rotation.  Returns array of length N−2.
    """
    obs = np.array(nodes)
    if len(obs) < 3:
        return np.array([])
    dx = np.diff(obs[:, 0])
    dy = np.diff(obs[:, 1])
    theta = np.arctan2(dx, -dy)
    kappa = np.diff(theta)
    return (kappa + np.pi) % (2 * np.pi) - np.pi


def integrate_kappa(
    kappa: np.ndarray,
    arc_length: float,
    start_pos: Tuple[float, float] = (0.0, 0.0),
    start_angle: float = 0.0,
) -> List[Tuple[float, float]]:
    """Reconstruct N = len(kappa)+2 nodes from a turning-angle sequence."""
    N = len(kappa) + 2
    ds = arc_length / (N - 1)
    nodes = [(float(start_pos[0]), float(start_pos[1]))]
    x, y, theta = float(start_pos[0]), float(start_pos[1]), float(start_angle)
    for i in range(N - 1):
        x += ds * np.sin(theta)
        y -= ds * np.cos(theta)
        nodes.append((x, y))
        if i < N - 2:
            theta += float(kappa[i])
    return nodes


# ---------------------------------------------------------------------------
# 3. THETA PROFILES  θ*(s', **params)
#
# Convention
# ----------
# s' ∈ [0, 1]  is normalised arc-length measured from the first edge midpoint
# (s'=0) to the last edge midpoint (s'=1).  θ*(0) ≡ 0 by convention; the
# returned value is the *relative* turning from the start tangent direction.
#
# Free parameters are fitted by matching θ*(s'_i, params) ≈ Δθ_i where Δθ_i
# is the provided tangent angle at edge i minus the tangent at edge 0.
# ---------------------------------------------------------------------------

# --- profile functions ---

def _profile_c(s_norm: np.ndarray, total_turn: float = 1.0) -> np.ndarray:
    """C: linear ramp — constant curvature (circular arc)."""
    return total_turn * s_norm


def _profile_u(s_norm: np.ndarray, total_turn: float = np.pi) -> np.ndarray:
    """U: linear ramp to ±π — semicircle."""
    return total_turn * s_norm


def _profile_s(s_norm: np.ndarray, amplitude: float = 0.5) -> np.ndarray:
    """S: half-sine bump — beam curves one way then back, net turn = 0."""
    return amplitude * np.sin(np.pi * s_norm)


def _profile_l(
    s_norm: np.ndarray,
    total_turn: float = np.pi / 2,
    turn_at: float = 0.7,
) -> np.ndarray:
    """L: step function — straight section then abrupt turn."""
    return np.where(s_norm >= turn_at, float(total_turn), 0.0)


THETA_PROFILES: Dict[str, Callable] = {
    "C": _profile_c,
    "U": _profile_u,
    "S": _profile_s,
    "L": _profile_l,
}


# --- fitting functions ---
# Each returns a dict of params that can be passed to the corresponding profile.

def _fit_c(s_norm: np.ndarray, dtheta: np.ndarray) -> dict:
    """Least-squares fit of total_turn for a linear ramp."""
    denom = float(np.dot(s_norm, s_norm)) + 1e-12
    return {"total_turn": float(np.dot(s_norm, dtheta) / denom)}


def _fit_u(s_norm: np.ndarray, dtheta: np.ndarray) -> dict:
    """U is π, but fit the sign from the data's dominant turning direction."""
    sign = float(np.sign(dtheta[-1])) if len(dtheta) > 0 else 1.0
    return {"total_turn": (sign or 1.0) * np.pi}


def _fit_s(s_norm: np.ndarray, dtheta: np.ndarray) -> dict:
    """Project onto sin(π s') to find the S amplitude."""
    basis = np.sin(np.pi * s_norm)
    denom = float(np.dot(basis, basis)) + 1e-12
    return {"amplitude": float(np.dot(dtheta, basis) / denom)}


def _fit_l(s_norm: np.ndarray, dtheta: np.ndarray) -> dict:
    """Grid-search for the step location; total turn from the last edge."""
    total_turn = float(dtheta[-1])
    best_rms, best_turn_at = np.inf, 0.7
    for t in np.linspace(0.05, 0.95, 19):
        residual = dtheta - np.where(s_norm >= t, total_turn, 0.0)
        rms = float(np.sqrt(np.mean(residual ** 2)))
        if rms < best_rms:
            best_rms, best_turn_at = rms, t
    return {"total_turn": total_turn, "turn_at": float(best_turn_at)}


_FIT_FNS: Dict[str, Callable] = {
    "C": _fit_c,
    "U": _fit_u,
    "S": _fit_s,
    "L": _fit_l,
}


# ---------------------------------------------------------------------------
# 4. BUILD IDEAL SHAPE
# ---------------------------------------------------------------------------

def build_ideal_from_edge_profile(
    profile_fn: Callable,
    n: int,
    arc_length: float,
    start_pos: Tuple[float, float] = (0.0, 0.0),
    start_angle: float = 0.0,
    **params,
) -> List[Tuple[float, float]]:
    """
    Generate n nodes by integrating a θ* profile at edge midpoints.

    Used in the test file to create synthetic ideal shapes.  The edge-midpoint
    integration is the same scheme used by _build_ideal_one_dir, so generation
    and evaluation are exactly consistent → 0 % RMS error for ideal shapes.

    s' is computed as (s_edge − s_edge_0) / span, i.e. the first edge midpoint
    maps to s'=0 and the last to s'=1.
    """
    ds = arc_length / (n - 1)
    s_nodes = np.linspace(0.0, 1.0, n)
    s_edges = (s_nodes[:-1] + s_nodes[1:]) / 2.0       # midpoints, length n-1

    s0 = s_edges[0]
    span = s_edges[-1] - s0 + 1e-12
    s_norm = (s_edges - s0) / span                       # ∈ [0, 1]

    dtheta_edges = profile_fn(s_norm, **params)
    theta_edges = start_angle + dtheta_edges

    nodes = [(float(start_pos[0]), float(start_pos[1]))]
    x, y = float(start_pos[0]), float(start_pos[1])
    for theta_e in theta_edges:
        x += ds * float(np.sin(theta_e))
        y -= ds * float(np.cos(theta_e))
        nodes.append((x, y))
    return nodes


def _build_ideal_one_dir(
    obs: np.ndarray,
    target_shape: str,
) -> Tuple[List[Tuple[float, float]], float, dict]:
    """
    Fit and reconstruct the ideal shape for one traversal direction.

    Returns (ideal_nodes, rms_normalised, fitted_params).
    rms_normalised = RMSE(ideal − provided) / arc_length.
    """
    N = len(obs)
    edge_vecs = np.diff(obs, axis=0)                    # (N-1, 2)
    edge_lengths = np.linalg.norm(edge_vecs, axis=1)
    arc_length = float(np.sum(edge_lengths))
    if arc_length < 1e-10 or N < 3:
        return obs.tolist(), 0.0, {}

    # Arc-length at node positions, normalised to [0, 1]
    s_nodes = np.zeros(N)
    s_nodes[1:] = np.cumsum(edge_lengths) / arc_length

    # Arc-length at edge midpoints
    s_edges = (s_nodes[:-1] + s_nodes[1:]) / 2.0

    # Shift so s'=0 at the first edge midpoint, s'=1 at the last
    s0 = s_edges[0]
    span = s_edges[-1] - s0 + 1e-12
    s_norm = (s_edges - s0) / span                       # ∈ [0, 1], length N-1

    # Provided relative turning angle at each edge (Δθ_i = θ_i − θ_0)
    theta_edges = np.arctan2(edge_vecs[:, 0], -edge_vecs[:, 1])
    theta_start = float(theta_edges[0])
    dtheta = (theta_edges - theta_start + np.pi) % (2 * np.pi) - np.pi

    # Fit profile parameters to the provided relative tangent data
    params = _FIT_FNS[target_shape](s_norm, dtheta)

    # Ideal relative tangent at each edge
    dtheta_ideal = THETA_PROFILES[target_shape](s_norm, **params)
    theta_ideal_edges = theta_start + dtheta_ideal

    # Reconstruct ideal nodes using same edge-midpoint integration
    ideal = np.empty((N, 2))
    ideal[0] = obs[0]
    for i in range(N - 1):
        ds = float(edge_lengths[i])
        ideal[i + 1, 0] = ideal[i, 0] + ds * np.sin(float(theta_ideal_edges[i]))
        ideal[i + 1, 1] = ideal[i, 1] - ds * np.cos(float(theta_ideal_edges[i]))

    # Normalised RMSE
    rms = float(np.sqrt(np.mean(np.sum((ideal - obs) ** 2, axis=1)))) / arc_length

    return ideal.tolist(), rms, params


# ---------------------------------------------------------------------------
# 5. PUBLIC API
# ---------------------------------------------------------------------------

def compute_shape_error(
    nodes: List[Tuple[float, float]],
    target_shape: str,
    check_reflection: bool = None,
) -> Dict:
    """
    Fit the ideal θ profile for target_shape to the provided nodes and
    return the normalised RMS node error (0 % = perfect, 100 % = one rod
    length of average deviation).

    Args:
        nodes:            List of (x, y) positions (N ≥ 3).
        target_shape:     "C", "U", "S", or "L".
        check_reflection: If True also try reversed traversal and keep the
                          lower error.  Defaults to config.CHECK_SHAPE_REFLECTION.
    """
    if check_reflection is None:
        try:
            import config
            check_reflection = bool(config.CHECK_SHAPE_REFLECTION)
        except Exception:
            check_reflection = False

    if target_shape not in THETA_PROFILES:
        return {
            "error_percent": 100.0,
            "confidence_percent": 0.0,
            "ideal_nodes": [],
            "rms_normalized": 1.0,
            "fitted_params": {},
            "diagnostics": f"Unknown target shape: {target_shape}",
        }

    obs = np.array(nodes)
    if len(obs) < 3:
        return {
            "error_percent": 100.0,
            "confidence_percent": 0.0,
            "ideal_nodes": [],
            "rms_normalized": 1.0,
            "fitted_params": {},
            "diagnostics": "Need at least 3 nodes.",
        }

    ideal_fwd, rms_fwd, params_fwd = _build_ideal_one_dir(obs, target_shape)
    best_ideal, best_rms, best_params = ideal_fwd, rms_fwd, params_fwd

    if check_reflection:
        ideal_rev, rms_rev, params_rev = _build_ideal_one_dir(obs[::-1], target_shape)
        if rms_rev < best_rms:
            # Reverse the ideal nodes back to original order for display
            best_ideal = ideal_rev[::-1]
            best_rms = rms_rev
            best_params = params_rev

    error_percent = min(best_rms * 100.0, 100.0)

    return {
        "error_percent": error_percent,
        "confidence_percent": 100.0 - error_percent,
        "ideal_nodes": best_ideal,
        "rms_normalized": best_rms,
        "fitted_params": best_params,
        "diagnostics": _diagnose(target_shape, error_percent, best_params),
    }


def _diagnose(target_shape: str, error_percent: float, params: dict) -> str:
    if error_percent < 2.0:
        return f"Shape matches {target_shape} well."
    detail = {k: f"{v:.3f}" for k, v in params.items()}
    return f"{target_shape} fit params: {detail}  →  {error_percent:.1f}% error."


# ---------------------------------------------------------------------------
# 6. ORCHESTRATOR DESCRIPTOR  (geometric text for LLM prompt)
# ---------------------------------------------------------------------------

def compute_shape_signature(
    observable_positions: List[Tuple[float, float]],
) -> Dict:
    """
    Compute geometric descriptors for the orchestrator's decision prompt.

    Returns tangent_angles, curvature_signs, total_turning_angle,
    tip_position, n_nodes — the same keys expected by orchestrator_agent.py.
    """
    if len(observable_positions) < 2:
        return {}

    obs = np.array(observable_positions)
    dx = np.diff(obs[:, 0])
    dy = np.diff(obs[:, 1])
    tangents = list(np.arctan2(dx, -dy).astype(float))

    kappa = extract_turning_angles(observable_positions)
    curvature_signs = [int(np.sign(k)) if abs(k) > 1e-6 else 0 for k in kappa]

    total_turning = float(tangents[-1] - tangents[0])
    while total_turning > np.pi:
        total_turning -= 2 * np.pi
    while total_turning < -np.pi:
        total_turning += 2 * np.pi

    return {
        "tangent_angles":     tangents,
        "curvature_signs":    curvature_signs,
        "total_turning_angle": total_turning,
        "tip_position":       tuple(obs[-1]),
        "n_nodes":            len(obs),
    }
