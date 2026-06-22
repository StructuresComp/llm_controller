"""
C-shape curvature profile: linear ramp = constant curvature = circular arc.

Exports:
    profile_c(s_norm, total_turn)  — ideal turning angle at normalised arc-length s
    fit_c(s_norm, dtheta)          — fixed-target fit pinned to TARGET_TURN (π/2)
    make_fit_c(target_turn)        — factory for fixed-target fits at any amplitude
"""

import numpy as np


def profile_c(s_norm: np.ndarray, total_turn: float = 1.0) -> np.ndarray:
    """theta*(s) = total_turn * s  — constant curvature throughout."""
    return total_turn * s_norm


# Target magnitude for the C case — must match beam_C.reference_fn's total_turn.
TARGET_TURN = np.pi / 2


def make_fit_c(target_turn: float):
    """Factory for a fixed-target C fit pinned to `target_turn` radians.

    Pinning the magnitude (the fit_u pattern) makes profile_loss a fixed-target
    metric rather than a profile-family-membership metric: a straight beam
    (dtheta ≈ 0) reconstructs a full arc at the target angle and scores high
    error, instead of free-fitting total_turn ≈ 0 and spuriously scoring ~0%.
    This factory lets scaled cases (C_30, C_60, ...) each pin their own target
    without duplicating the logic.
    """
    def _fit(s_norm: np.ndarray, dtheta: np.ndarray) -> dict:
        sign = float(np.sign(np.dot(s_norm, dtheta))) or 1.0
        return {"total_turn": sign * target_turn}
    return _fit


# Default C fit, pinned to the standard quarter-circle amplitude.
fit_c = make_fit_c(TARGET_TURN)
