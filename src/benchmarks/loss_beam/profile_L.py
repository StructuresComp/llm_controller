"""
L-shape curvature profile: cosine bump centered near the tip.

Exports:
    kappa_l(s_norm, rod_length, s0, width) — curvature density as function of arc-length
"""

import numpy as np


def kappa_l(s_norm: np.ndarray, rod_length: float = 1.0, s0: float = 0.65, width: float = 0.25) -> np.ndarray:
    """
    Curvature function for L-shape.

    Cosine bump centered at s0 with given width; exactly zero outside the bend region.
    Amplitude A = pi/width ensures the total turning angle integrates to pi/2 (90° L-turn).

    kappa = 2*tan(theta/2) / dL, normalized by rod_length.
    """
    u = (s_norm - s0) / (0.5 * width)
    bump = np.where(np.abs(u) <= 1.0, 0.5 * (1.0 + np.cos(np.pi * u)), 0.0)
    A = np.pi / width
    kappa_raw = A * bump
    return kappa_raw / rod_length
