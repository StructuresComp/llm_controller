"""
S-shape curvature profile: two opposite bowls via full-sine with terminal curls.

Exports:
    kappa_s(s_norm, rod_length, As, Aend, sig) — curvature density as function of arc-length
"""

import numpy as np


def kappa_s(s_norm: np.ndarray, rod_length: float = 1.0, As: float = 10.0, Aend: float = 3.0, sig: float = 0.30) -> np.ndarray:
    """
    Curvature function for S-shape.

    kappa = 2*tan(theta/2) / dL, normalized by rod_length.
    """
    base         = As * np.sin(2.0 * np.pi * s_norm)
    bottom_curl  = np.exp(-(s_norm / sig)**2)
    top_curl     = np.exp(-((1.0 - s_norm) / sig)**2)
    kappa_raw    = base + Aend * (bottom_curl - top_curl)
    return kappa_raw / rod_length
