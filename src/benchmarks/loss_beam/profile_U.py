"""
U-shape curvature profile: cosine bump centered at arc-length midpoint.

Exports:
    kappa_u(s_norm, rod_length, width) — curvature density as function of arc-length
"""

import numpy as np


def kappa_u(s_norm: np.ndarray, rod_length: float = 1.0, width: float = 0.65) -> np.ndarray:
    """
    Curvature function for U-shape.

    Cosine bump centered at s=0.5; exactly zero outside the bend region (compact support).
    Amplitude A = 2*pi/width ensures the total turning angle integrates to pi (180° U-turn).

    kappa = 2*tan(theta/2) / dL, normalized by rod_length.
    """
    u = (s_norm - 0.5) / (0.5 * width)
    bump = np.where(np.abs(u) <= 1.0, 0.5 * (1.0 + np.cos(np.pi * u)), 0.0)
    A = 2.0 * np.pi / width
    kappa_raw = A * bump
    return kappa_raw / rod_length
