"""
L-shape curvature profile: step function — straight section then abrupt ~90° turn.

Exports:
    profile_l(s_norm, total_turn, turn_at)  — ideal turning angle at normalised arc-length s
    fit_l(s_norm, dtheta)                   — fit total_turn and turn_at to observed angles
"""

import numpy as np


def profile_l(
    s_norm: np.ndarray,
    total_turn: float = np.pi / 2,
    turn_at: float = 0.7,
) -> np.ndarray:
    """
    Step function: 0 for s < turn_at, total_turn for s >= turn_at.
    Models a beam that is straight along most of its length and then bends sharply.
    """
    return np.where(s_norm >= turn_at, float(total_turn), 0.0)


def fit_l(s_norm: np.ndarray, dtheta: np.ndarray) -> dict:
    """
    Grid-search over turn_at ∈ [0.05, 0.95] to minimise RMS residual.
    total_turn is taken directly from the last observed turning angle.
    """
    total_turn = float(dtheta[-1])
    best_rms, best_turn_at = np.inf, 0.7
    for t in np.linspace(0.05, 0.95, 19):
        residual = dtheta - np.where(s_norm >= t, total_turn, 0.0)
        rms = float(np.sqrt(np.mean(residual ** 2)))
        if rms < best_rms:
            best_rms, best_turn_at = rms, t
    return {"total_turn": total_turn, "turn_at": float(best_turn_at)}
