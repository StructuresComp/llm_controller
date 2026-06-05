# ---------------------------------------------------------------------------
# src/tools/shape_checker.py
# ---------------------------------------------------------------------------
# Deterministic shape validator based on discrete curvature (DDG).
#
# Core idea: a discrete curve is fully described by the turning-angle sequence
#   κ_i = θ_{i+1} − θ_i   at each interior node
# This is invariant to rigid-body translation and rotation.
# Wall position, absolute coordinates, and symmetry in XY are irrelevant.
#
# Section order:
#   1. Imports
#   2. Primitives      — extract_turning_angles, integrate_kappa
#   3. Ideal profiles  — one function per shape type
#   4. Error functions — one function per shape type
#   5. Public API      — compute_shape_error, reconstruct_ideal
#   6. Orchestrator    — compute_shape_signature (text descriptor for LLM)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 1. IMPORTS
# ---------------------------------------------------------------------------
import numpy as np
from typing import List, Tuple, Dict


# ---------------------------------------------------------------------------
# 2. PRIMITIVES
# ---------------------------------------------------------------------------

def extract_turning_angles(nodes: List[Tuple[float, float]]) -> np.ndarray:
    """
    Compute discrete curvature κ_i = θ_{i+1} − θ_i at each interior node.

    θ_i is the tangent angle of edge i, measured counter-clockwise from the
    −Y axis:  θ = arctan2(dx, −dy).

    Returns array of length N−2 (N = number of nodes).
    Invariant to rigid-body translation and rotation.
    """
    obs = np.array(nodes)
    N = len(obs)
    if N < 3:
        return np.array([])

    dx = np.diff(obs[:, 0])
    dy = np.diff(obs[:, 1])
    theta = np.arctan2(dx, -dy)         # CCW from −Y, length N−1

    kappa = np.diff(theta)              # length N−2
    kappa = (kappa + np.pi) % (2 * np.pi) - np.pi  # wrap to (−π, π]
    return kappa


def integrate_kappa(
    kappa: np.ndarray,
    arc_length: float,
    start_pos: Tuple[float, float] = (0.0, 0.0),
    start_angle: float = 0.0,
) -> List[Tuple[float, float]]:
    """
    Reconstruct node positions from a turning-angle sequence.

    N = len(kappa) + 2 nodes, uniform edge length ds = arc_length / (N−1).
    start_angle is the tangent angle of the first edge (CCW from −Y).

    Inverse of extract_turning_angles (exact round-trip for uniform spacing).
    """
    N = len(kappa) + 2
    ds = arc_length / (N - 1)

    nodes = [(float(start_pos[0]), float(start_pos[1]))]
    x, y = float(start_pos[0]), float(start_pos[1])
    theta = float(start_angle)

    for i in range(N - 1):
        x = x + ds * np.sin(theta)
        y = y - ds * np.cos(theta)
        nodes.append((x, y))
        if i < N - 2:
            theta += float(kappa[i])

    return nodes


# ---------------------------------------------------------------------------
# 3. IDEAL κ PROFILES
# ---------------------------------------------------------------------------

def _ideal_kappa_c(kappa: np.ndarray) -> np.ndarray:
    """C: best-fit constant curvature (circular arc)."""
    return np.full(len(kappa), np.mean(kappa))


def _ideal_kappa_u(kappa: np.ndarray) -> np.ndarray:
    """U: constant curvature, total turn fixed at ±π."""
    sign = float(np.sign(np.sum(kappa))) or 1.0
    return np.full(len(kappa), sign * np.pi / len(kappa))


def _ideal_kappa_s(kappa: np.ndarray) -> np.ndarray:
    """S: antisymmetric profile scaled to match the total absolute turning."""
    N = len(kappa)
    mid = N // 2
    profile = np.zeros(N)
    profile[:mid] = 1.0
    profile[N - mid:] = -1.0                            # middle = 0 for odd N
    n_nonzero = float(np.sum(np.abs(profile))) or 1.0
    scale = np.sum(np.abs(kappa)) / n_nonzero
    return profile * scale


def _ideal_kappa_l(kappa: np.ndarray) -> np.ndarray:
    """L: all turning concentrated at the node with largest |κ|."""
    ideal = np.zeros(len(kappa))
    i_max = int(np.argmax(np.abs(kappa)))
    ideal[i_max] = float(np.sum(kappa))
    return ideal


_IDEAL_FN: Dict = {
    "C": _ideal_kappa_c,
    "U": _ideal_kappa_u,
    "S": _ideal_kappa_s,
    "L": _ideal_kappa_l,
}


# ---------------------------------------------------------------------------
# 4. ERROR FUNCTIONS  (0–100 %, lower = better)
# ---------------------------------------------------------------------------

def _c_error(kappa: np.ndarray) -> float:
    """
    C: constant curvature.  Error = coefficient of variation (std / |mean|).
    0 % for a perfect circle of any radius; 100 % for an irregular curve.
    """
    mean_k = float(np.mean(kappa))
    if abs(mean_k) < 1e-8:
        return 100.0
    return min(float(np.std(kappa)) / abs(mean_k) * 100.0, 100.0)


def _u_error(kappa: np.ndarray) -> float:
    """
    U: constant curvature AND total turning ≈ ±π.
    Combines CV (uniformity) with a turning-angle penalty.
    """
    mean_k = abs(float(np.mean(kappa)))
    if mean_k < 1e-8:
        return 100.0
    cv = float(np.std(kappa)) / mean_k
    angle_err = abs(abs(float(np.sum(kappa))) - np.pi) / np.pi
    return min((0.5 * cv + 0.5 * angle_err) * 100.0, 100.0)


def _s_error(kappa: np.ndarray) -> float:
    """
    S: antisymmetric κ sequence (κ_i ≈ −κ_{N−1−i}) and net turn ≈ 0.
    Error = mean |κ + reverse(κ)| normalised by mean |κ|.
    """
    residual = float(np.mean(np.abs(kappa + kappa[::-1])))
    scale = float(np.mean(np.abs(kappa))) + 1e-8
    return min(residual / scale * 50.0, 100.0)


def _l_error(kappa: np.ndarray) -> float:
    """
    L: curvature concentrated at a single interior node.
    Error = 1 − (max|κ| / Σ|κ|), scaled to 0–100 %.
    """
    total = float(np.sum(np.abs(kappa)))
    if total < 1e-8:
        return 100.0
    concentration = float(np.max(np.abs(kappa))) / total
    return min((1.0 - concentration) * 100.0, 100.0)


_ERROR_FN: Dict = {
    "C": _c_error,
    "U": _u_error,
    "S": _s_error,
    "L": _l_error,
}


# ---------------------------------------------------------------------------
# 5. PUBLIC API
# ---------------------------------------------------------------------------

def compute_shape_error(
    nodes: List[Tuple[float, float]],
    target_shape: str,
) -> Dict:
    """
    Compute normalised shape error (0–100 %, 0 = perfect match).

    Invariant to translation, rotation, and CW/CCW orientation (the error
    functions are all symmetric under κ → −κ for the shapes defined here).

    Args:
        nodes:        List of (x, y) node positions (any number ≥ 3).
        target_shape: "C", "U", "S", or "L".

    Returns dict with:
        error_percent      — 0 (perfect) to 100 (wrong shape)
        confidence_percent — 100 − error_percent
        kappa_provided     — extracted turning angles
        kappa_ideal        — best-fit ideal turning angles
        ideal_nodes        — ideal shape anchored to nodes[0] (for overlay)
        diagnostics        — one-line explanation
    """
    if target_shape not in _ERROR_FN:
        return {
            "error_percent": 100.0, "confidence_percent": 0.0,
            "kappa_provided": [], "kappa_ideal": [], "ideal_nodes": [],
            "diagnostics": f"Unknown target shape: {target_shape}",
        }

    kappa = extract_turning_angles(nodes)
    if len(kappa) == 0:
        return {
            "error_percent": 100.0, "confidence_percent": 0.0,
            "kappa_provided": [], "kappa_ideal": [], "ideal_nodes": [],
            "diagnostics": "Need at least 3 nodes.",
        }

    error = float(_ERROR_FN[target_shape](kappa))
    kappa_ideal = _IDEAL_FN[target_shape](kappa)
    ideal_nodes = reconstruct_ideal(nodes, kappa_ideal)
    diagnostics = _diagnose(target_shape, error, kappa)

    return {
        "error_percent": error,
        "confidence_percent": 100.0 - error,
        "kappa_provided": kappa.tolist(),
        "kappa_ideal": kappa_ideal.tolist(),
        "ideal_nodes": ideal_nodes,
        "diagnostics": diagnostics,
    }


def reconstruct_ideal(
    nodes: List[Tuple[float, float]],
    kappa_ideal: np.ndarray,
) -> List[Tuple[float, float]]:
    """
    Reconstruct the ideal shape in the coordinate frame of the provided shape.

    Anchors to nodes[0] (position) and the direction of edge 0 (tangent angle),
    then integrates kappa_ideal forward with uniform arc-length spacing.
    """
    obs = np.array(nodes)
    start_pos = (float(obs[0, 0]), float(obs[0, 1]))
    dx0 = obs[1, 0] - obs[0, 0]
    dy0 = obs[1, 1] - obs[0, 1]
    start_angle = float(np.arctan2(dx0, -dy0))
    arc_length = float(np.sum(np.sqrt(np.sum(np.diff(obs, axis=0) ** 2, axis=1))))
    return integrate_kappa(kappa_ideal, arc_length, start_pos, start_angle)


def _diagnose(target_shape: str, error: float, kappa: np.ndarray) -> str:
    if error < 2.0:
        return f"Shape matches {target_shape} well."

    cv = float(np.std(kappa)) / (abs(float(np.mean(kappa))) + 1e-8)
    total = float(abs(np.sum(kappa)))

    if target_shape == "C":
        return f"Curvature not uniform (CV={cv:.2f}); C needs constant κ."
    if target_shape == "U":
        return f"Total turning {total:.2f} rad (need π={np.pi:.2f}) or CV={cv:.2f} too high."
    if target_shape == "S":
        res = float(np.mean(np.abs(kappa + kappa[::-1]))) / (float(np.mean(np.abs(kappa))) + 1e-8)
        return f"Antisymmetry residual {res:.2f} (want 0)."
    if target_shape == "L":
        conc = float(np.max(np.abs(kappa))) / (float(np.sum(np.abs(kappa))) + 1e-8)
        return f"Curvature concentration {conc:.2f} (want ~1.0)."
    return f"Error: {error:.1f}%."


# ---------------------------------------------------------------------------
# 6. ORCHESTRATOR DESCRIPTOR (geometric text for LLM prompt)
# ---------------------------------------------------------------------------

def compute_shape_signature(observable_positions: List[Tuple[float, float]]) -> Dict:
    """
    Compute geometric descriptors for the orchestrator's decision prompt.

    Returns a dict with tangent_angles, curvature_signs, total_turning_angle,
    tip_position, and n_nodes — the same fields expected by orchestrator_agent.py.
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
        "tangent_angles":    tangents,
        "curvature_signs":   curvature_signs,
        "total_turning_angle": total_turning,
        "tip_position":      tuple(obs[-1]),
        "n_nodes":           len(obs),
    }
