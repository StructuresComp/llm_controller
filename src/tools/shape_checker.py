# ---------------------------------------------------------------------------
# src/evaluation/shape_checker.py
# ---------------------------------------------------------------------------
# Deterministic geometric shape validator.
# Computes normalized shape error (0-100%, where 0 = perfect match) from
# observable node positions, agnostic to full vs. partial node counts.
# ---------------------------------------------------------------------------

import numpy as np
from typing import List, Tuple, Dict


def compute_shape_signature(observable_positions: List[Tuple[float, float]]) -> Dict:
    """
    Compute geometric features from observable node positions.

    Args:
        observable_positions: List of (x, y) tuples for observable nodes

    Returns:
        Dict with:
          - tangent_angles: list of angles (rad, CCW from -Y axis)
          - curvature_signs: sequence of +1, 0, -1 indicating local bends
          - total_turning_angle: net rotation from first to last node
          - tip_position: (x, y) of last node
          - n_nodes: number of observable nodes
    """
    if len(observable_positions) < 2:
        return {}

    obs = np.array(observable_positions)

    # Tangent angles at each node (CCW from -Y axis)
    tangents = []
    for i in range(len(obs) - 1):
        dx = obs[i + 1][0] - obs[i][0]
        dy = obs[i + 1][1] - obs[i][1]
        angle = np.arctan2(dx, -dy)  # CCW from -Y axis
        tangents.append(float(angle))

    # Curvature signs: sign of change in tangent angle
    curvature_signs = []
    for i in range(len(tangents) - 1):
        dangle = tangents[i + 1] - tangents[i]
        # Wrap to [-π, π]
        while dangle > np.pi:
            dangle -= 2 * np.pi
        while dangle < -np.pi:
            dangle += 2 * np.pi
        sign = 1 if dangle > 1e-6 else (-1 if dangle < -1e-6 else 0)
        curvature_signs.append(sign)

    # Total turning angle
    total_turning = tangents[-1] - tangents[0]
    while total_turning > np.pi:
        total_turning -= 2 * np.pi
    while total_turning < -np.pi:
        total_turning += 2 * np.pi

    return {
        "tangent_angles": tangents,
        "curvature_signs": curvature_signs,
        "total_turning_angle": float(total_turning),
        "tip_position": tuple(obs[-1]),
        "n_nodes": len(obs),
    }


def _curvature_pattern_match(computed_signs: List[int], expected_pattern: List[int], tolerance: float = 0.3) -> float:
    """
    Score curvature pattern match (0-100%, lower is better error).

    Zeros (straight segments) are excluded from the fraction computation so
    that shapes like L (straight + curved) score correctly.

    Two-element pattern (e.g. [1, -1] for S-shape): checks that BOTH curvature
    directions are present in roughly equal proportion (min fraction >= 0.3).

    Single-element pattern (e.g. [1] for C): checks that the dominant nonzero
    sign matches, and that it makes up >= ~70% of nonzero signs.
    """
    nonzero = [s for s in computed_signs if s != 0]
    if not nonzero:
        return 0.0     # all straight — no curvature mismatch

    n = len(nonzero)
    pos_count = sum(1 for s in nonzero if s > 0)
    neg_count = sum(1 for s in nonzero if s < 0)

    if len(expected_pattern) > 1:
        # S-shape style: both curvature directions must be present
        min_frac = min(pos_count, neg_count) / n
        # Perfect: 50/50 split. Error grows as split becomes more one-sided.
        threshold = 0.3
        error_percent = max(0.0, (threshold - min_frac) / threshold * 100)
        return min(error_percent, 100.0)

    expected_sign = expected_pattern[0] if expected_pattern else 0
    if expected_sign == 0:
        return 0.0

    n_matching = sum(1 for s in nonzero if s == expected_sign)
    match_fraction = n_matching / n

    if match_fraction >= 0.7:
        error_percent = (1.0 - match_fraction) * 50
    else:
        error_percent = 100.0 - match_fraction * 100

    return min(error_percent, 100.0)


def _turning_angle_error(computed_angle: float, expected_angle: float, tolerance: float = 0.3) -> float:
    """
    Score turning angle error.

    Tolerance is fractional deviation allowed (e.g., 0.3 = 30% ok).
    """
    if abs(expected_angle) < 1e-6:
        return 0.0 if abs(computed_angle) < 1e-6 else 100.0

    angle_error = abs(computed_angle - expected_angle) / abs(expected_angle)

    # If within tolerance, error is proportional; else capped at 100
    if angle_error <= tolerance:
        error_percent = (angle_error / tolerance) * 50
    else:
        error_percent = 50 + min((angle_error - tolerance) / tolerance * 50, 50)

    return min(error_percent, 100.0)


def _wall_contact_error(observable_positions: List[Tuple[float, float]], wall_h: float, expected_fraction: float, tolerance: float = 0.2) -> float:
    """
    Score wall contact fraction error.

    Count nodes within tolerance of wall_h, compare to expected fraction.
    """
    obs = np.array(observable_positions)
    y_coords = obs[:, 1]

    # Nodes touching wall: within 0.005 m of wall_h
    contact_tol = 0.005
    touching = np.sum(np.abs(y_coords - wall_h) < contact_tol)
    contact_fraction = touching / len(y_coords) if len(y_coords) > 0 else 0

    # Error in contact fraction
    contact_error = abs(contact_fraction - expected_fraction)

    if contact_error <= tolerance:
        error_percent = (contact_error / tolerance) * 50
    else:
        error_percent = 50 + min((contact_error - tolerance) / tolerance * 50, 50)

    return min(error_percent, 100.0)


def _symmetry_error(observable_positions: List[Tuple[float, float]]) -> float:
    """
    Score symmetry deviation for shapes like U.

    Compute left-right symmetry around the center x.
    """
    obs = np.array(observable_positions)
    n = len(obs)

    if n < 3:
        return 50.0

    # Check left-right symmetry: mirror image around center x
    center_x = (obs[0, 0] + obs[-1, 0]) / 2

    # Compute deviation from symmetry
    sym_error = 0.0
    for i in range(n // 2):
        left_node = obs[i]
        right_node = obs[n - 1 - i]

        # Mirror position of left node
        mirrored_left_x = 2 * center_x - left_node[0]

        # Distance between mirrored and actual right node
        dist = np.sqrt((right_node[0] - mirrored_left_x) ** 2 + (right_node[1] - left_node[1]) ** 2)
        sym_error += dist

    # Normalize by beam extent
    extent = np.linalg.norm(obs[-1] - obs[0])
    if extent < 1e-6:
        return 50.0

    normalized_error = (sym_error / (n // 2)) / extent
    error_percent = min(normalized_error * 100, 100.0)

    return error_percent


# Shape specifications.
# Turning angles and contact fractions are calibrated against the ideal synthetic
# shapes in src/tests/test_shape_checker.py, which use arc-length parametrisation.
# Real beam data will score against these references — deviation from the ideal
# registers as nonzero error.
SHAPE_SPECS = {
    "C": {
        "curvature_pattern": [1],         # all positive (curves right)
        "expected_turning_angle": 0.6955, # calibrated: R=0.15 arc to wall
        "expected_wall_contact_fraction": 0.2,
        "wall_contact_tolerance": 0.15,
        "symmetry_required": False,
        "weights": {"curvature": 0.50, "angle": 0.35, "contact": 0.15, "symmetry": 0.0},
    },
    "U": {
        "curvature_pattern": [-1],        # all negative (single-direction semicircle)
        "expected_turning_angle": -2.618, # calibrated: semicircle R=0.12
        "expected_wall_contact_fraction": 0.1429,
        "wall_contact_tolerance": 0.15,
        "symmetry_required": True,
        "weights": {"curvature": 0.35, "angle": 0.30, "contact": 0.15, "symmetry": 0.20},
    },
    "S": {
        "curvature_pattern": [1, -1],     # both directions present (alternating)
        "expected_turning_angle": 0.0,    # calibrated: net zero rotation
        "expected_wall_contact_fraction": 0.0,
        "wall_contact_tolerance": 0.5,
        "symmetry_required": False,
        "weights": {"curvature": 0.60, "angle": 0.25, "contact": 0.15, "symmetry": 0.0},
    },
    "L": {
        "curvature_pattern": [1],         # positive curvature at the bend
        "expected_turning_angle": 1.309,  # calibrated: 90° arc R=0.08
        "expected_wall_contact_fraction": 0.1667,
        "wall_contact_tolerance": 0.15,
        "symmetry_required": False,
        "weights": {"curvature": 0.45, "angle": 0.40, "contact": 0.15, "symmetry": 0.0},
    },
}


def compute_shape_error(observable_positions: List[Tuple[float, float]], target_shape: str, wall_h: float = -0.12) -> Dict:
    """
    Compute normalized shape error (0-100%, where 0 = perfect match).

    Args:
        observable_positions: List of (x, y) node positions
        target_shape: "C", "U", "S", or "L"
        wall_h: Wall position (y-coordinate)

    Returns:
        Dict with:
          - overall_error_percent: 0-100, lower is better match
          - match_confidence_percent: 100 - overall_error, for LLM consumption
          - breakdown: dict of individual error components
          - diagnostics: human-readable explanation
          - signature: computed shape signature
    """
    if target_shape not in SHAPE_SPECS:
        return {
            "overall_error_percent": 100.0,
            "match_confidence_percent": 0.0,
            "breakdown": {},
            "diagnostics": f"Unknown target shape: {target_shape}",
            "signature": {},
        }

    spec = SHAPE_SPECS[target_shape]
    sig = compute_shape_signature(observable_positions)

    if not sig:
        return {
            "overall_error_percent": 100.0,
            "match_confidence_percent": 0.0,
            "breakdown": {},
            "diagnostics": "Insufficient nodes to compute signature",
            "signature": sig,
        }

    # Compute individual error components
    curvature_error = _curvature_pattern_match(sig["curvature_signs"], spec["curvature_pattern"])
    angle_error = _turning_angle_error(sig["total_turning_angle"], spec["expected_turning_angle"], tolerance=0.4)
    contact_error = _wall_contact_error(observable_positions, wall_h, spec["expected_wall_contact_fraction"], tolerance=spec.get("wall_contact_tolerance", 0.25))

    symmetry_error = 0.0
    if spec.get("symmetry_required"):
        symmetry_error = _symmetry_error(observable_positions)

    # Weighted composite error
    weights = spec["weights"]
    overall_error = (
        weights["curvature"] * curvature_error +
        weights["angle"] * angle_error +
        weights["contact"] * contact_error +
        weights["symmetry"] * symmetry_error
    )
    overall_error = min(overall_error, 100.0)

    # Diagnostics
    diag_parts = []
    if curvature_error > 30:
        diag_parts.append("curvature pattern doesn't match target")
    if angle_error > 30:
        diag_parts.append("rotation angle is off")
    if contact_error > 30:
        diag_parts.append("wall contact is insufficient")
    if symmetry_error > 30 and spec.get("symmetry_required"):
        diag_parts.append("shape is not symmetric enough")

    if diag_parts:
        diagnostics = f"Issues: {', '.join(diag_parts)}"
    else:
        diagnostics = "Shape matches target well."

    return {
        "overall_error_percent": float(overall_error),
        "match_confidence_percent": float(100.0 - overall_error),
        "breakdown": {
            "curvature": float(curvature_error),
            "turning_angle": float(angle_error),
            "wall_contact": float(contact_error),
            "symmetry": float(symmetry_error),
        },
        "diagnostics": diagnostics,
        "signature": sig,
    }
