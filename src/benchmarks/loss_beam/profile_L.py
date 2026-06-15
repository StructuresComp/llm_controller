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


# Target geometry for the L case — must match beam_L.reference_fn.
TARGET_TURN = np.pi / 2
TARGET_TURN_AT = 0.65


def fit_l(s_norm: np.ndarray, dtheta: np.ndarray) -> dict:
    """
    Fixed-target fit: total_turn pinned to ±TARGET_TURN (sign from the last
    observed turning angle) and turn_at pinned to TARGET_TURN_AT.

    Both parameters are pinned (the fit_u pattern) so profile_loss is a
    fixed-target metric. With turn_at left free, the earlier grid search slid
    the pinned bend to the end of the beam, where it touched only one edge, so a
    straight beam still scored ~6%; pinning the bend location to the target
    makes a straight beam score high error as intended.
    """
    sign = float(np.sign(dtheta[-1])) if len(dtheta) > 0 else 1.0
    total_turn = (sign or 1.0) * TARGET_TURN
    return {"total_turn": total_turn, "turn_at": TARGET_TURN_AT}
