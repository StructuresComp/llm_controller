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


def fit_c(s_norm: np.ndarray, dtheta: np.ndarray) -> dict:
    """Least-squares fit of total_turn to observed turning angles."""
    denom = float(np.dot(s_norm, s_norm)) + 1e-12
    return {"total_turn": float(np.dot(s_norm, dtheta) / denom)}
