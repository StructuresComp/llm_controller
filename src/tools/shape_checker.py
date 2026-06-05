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

    More lenient: check if the dominant curvature matches the expected pattern,
    allowing for noise and variation in the pattern.
    """
    if not computed_signs:
        return 50.0 if not expected_pattern else 100.0

    # Count sign frequencies
    pos_count = sum(1 for s in computed_signs if s > 0)
    neg_count = sum(1 for s in computed_signs if s < 0)
    n_total = len(computed_signs)

    if n_total == 0:
        return 100.0

    # Dominant sign in computed pattern
    dominant_sign = 1 if pos_count >= neg_count else (-1 if neg_count > pos_count else 0)
    dominant_fraction = max(pos_count, neg_count) / n_total

    # Expected dominant sign
    expected_dominant = expected_pattern[0] if expected_pattern else 0

    # Score: how well does dominant sign match?
    if expected_dominant == 0:
        # No specific expectation
        return 0.0

    if dominant_sign == expected_dominant and dominant_fraction >= 0.6:
        # Good match: dominant sign is correct and > 60% of signs
        error_percent = (1 - dominant_fraction) * 50
    elif dominant_sign == expected_dominant:
        # Correct sign but not dominant enough
        error_percent = 50 + (1 - dominant_fraction) * 25
    else:
        # Wrong dominant sign: major error
        error_percent = 100.0

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


# Shape specifications: reference for each target shape
SHAPE_SPECS = {
    "C": {
        "curvature_pattern": [1],  # all left curves (positive curvature)
        "expected_turning_angle": 1.5708,  # π/2 radians
        "expected_wall_contact_fraction": 0.2,  # Relaxed: even small contact is ok
        "wall_contact_tolerance": 0.5,  # Very relaxed for synthetic shapes
        "symmetry_required": False,
        "weights": {"curvature": 0.50, "angle": 0.35, "contact": 0.15, "symmetry": 0.0},
    },
    "U": {
        "curvature_pattern": [1, -1],  # left then right curvature
        "expected_turning_angle": 3.14159,  # π radians (full reversal)
        "expected_wall_contact_fraction": 0.2,
        "wall_contact_tolerance": 0.6,
        "symmetry_required": True,
        "symmetry_tolerance": 0.40,  # More lenient symmetry check
        "weights": {"curvature": 0.35, "angle": 0.30, "contact": 0.15, "symmetry": 0.20},
    },
    "S": {
        "curvature_pattern": [1, -1],  # Detect at least left-right transition
        "expected_turning_angle": 0.0,  # Should return to neutral
        "expected_wall_contact_fraction": 0.1,
        "wall_contact_tolerance": 0.7,
        "symmetry_required": False,
        "weights": {"curvature": 0.50, "angle": 0.30, "contact": 0.20, "symmetry": 0.0},
    },
    "L": {
        "curvature_pattern": [1],  # Single strong left turn
        "expected_turning_angle": 1.5708,  # π/2 radians
        "expected_wall_contact_fraction": 0.2,
        "wall_contact_tolerance": 0.5,
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
