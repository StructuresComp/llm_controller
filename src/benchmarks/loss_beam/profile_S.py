"""
S-shape curvature profile: full-sine, one inflection point, net turn ≈ 0.

Exports:
    profile_s(s_norm, amplitude)  — ideal turning angle at normalised arc-length s
    fit_s(s_norm, dtheta)         — fit amplitude to observed turning angles
"""

import numpy as np


def profile_s(s_norm: np.ndarray, amplitude: float = 0.5) -> np.ndarray:
    """theta*(s) = amplitude * sin(2π*s)  — symmetric S, net turn = 0."""
    return amplitude * np.sin(2 * np.pi * s_norm)


# Target amplitude for the feasible S case — must match beam_S.reference_fn's amplitude.
TARGET_AMPLITUDE = 1.2


def make_fit_s(target_amplitude: float):
    """Build a fixed-target S fit pinned to `target_amplitude` (sign read from data).

    Pinning (the fit_u pattern) makes profile_loss a fixed-target metric: a
    straight beam (dtheta ≈ 0) reconstructs a full S and scores high error,
    instead of free-fitting amplitude ≈ 0 and spuriously scoring ~0%. The
    factory lets a harder cell (e.g. S_infeasible) pin a larger amplitude
    without duplicating the fit logic.
    """
    def _fit(s_norm: np.ndarray, dtheta: np.ndarray) -> dict:
        basis = np.sin(2 * np.pi * s_norm)
        sign = float(np.sign(np.dot(dtheta, basis))) or 1.0
        return {"amplitude": sign * target_amplitude}
    return _fit


# Default S fit, pinned to the feasible-cell amplitude.
fit_s = make_fit_s(TARGET_AMPLITUDE)
