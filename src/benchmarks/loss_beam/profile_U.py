"""
U-shape curvature profile: linear ramp to π = semicircle.

Exports:
    profile_u(s_norm, total_turn)  — ideal turning angle at normalised arc-length s
    fit_u(s_norm, dtheta)          — estimate total_turn from observed turning angles
"""

import numpy as np


def profile_u(s_norm: np.ndarray, total_turn: float = np.pi) -> np.ndarray:
    """theta*(s) = total_turn * s  — same form as C but total_turn = π."""
    return total_turn * s_norm


def fit_u(s_norm: np.ndarray, dtheta: np.ndarray) -> dict:
    """Sign-corrected π: total turn is always ±π; we just read the sign from the data."""
    sign = float(np.sign(dtheta[-1])) if len(dtheta) > 0 else 1.0
    return {"total_turn": (sign or 1.0) * np.pi}
