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


def fit_s(s_norm: np.ndarray, dtheta: np.ndarray) -> dict:
    """Least-squares projection of dtheta onto sin(2π*s)."""
    basis = np.sin(2 * np.pi * s_norm)
    denom = float(np.dot(basis, basis)) + 1e-12
    return {"amplitude": float(np.dot(dtheta, basis) / denom)}
