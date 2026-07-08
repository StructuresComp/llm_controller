"""
C-shape curvature profile: realistic C with terminal curls.

Exports:
    kappa_c(s_norm, kc0, Ac, sigc) — curvature density as function of arc-length
"""

import numpy as np


def kappa_c(s_norm: np.ndarray, rod_length: float = 1.0, kc0: float = 4.5, Ac: float = 3.0, sigc: float = 0.15) -> np.ndarray:
    """
    Curvature function for C-shape.

    kappa = 2*tan(theta/2) / dL, normalized by rod_length.
    """
    kappa_raw = kc0 + Ac * (np.exp(-(s_norm/sigc)**2) + np.exp(-((1.0-s_norm)/sigc)**2))
    return kappa_raw / rod_length
