"""
C-shape curvature profile: linear ramp = constant curvature = circular arc.

Exports:
    profile_c(s_norm, total_turn)  — ideal turning angle at normalised arc-length s
    fit_c(s_norm, dtheta)          — fit total_turn to observed turning angles
"""

import numpy as np


def profile_c(s_norm: np.ndarray, total_turn: float = 1.0) -> np.ndarray:
    """theta*(s) = total_turn * s  — constant curvature throughout."""
    return total_turn * s_norm


# Target magnitude for the C case — must match beam_C.reference_fn's total_turn.
TARGET_TURN = np.pi / 2


def fit_c(s_norm: np.ndarray, dtheta: np.ndarray) -> dict:
    """Fixed-target fit: magnitude pinned to TARGET_TURN, sign read from the data.

    Pinning the magnitude (the fit_u pattern) makes profile_loss a fixed-target
    metric rather than a profile-family-membership metric: a straight beam
    (dtheta ≈ 0) reconstructs a full ±90° arc and scores high error, instead of
    free-fitting total_turn ≈ 0 and spuriously scoring ~0%.
    """
    sign = float(np.sign(np.dot(s_norm, dtheta))) or 1.0
    return {"total_turn": sign * TARGET_TURN}
