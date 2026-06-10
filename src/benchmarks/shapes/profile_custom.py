"""
Custom curvature profile: wrap a user-supplied kappa(s) function into the
(profile_fn, fit_fn) pair expected by build_ideal_from_edge_profile and profile_loss.

Usage in a case file:

    from src.benchmarks.shapes.profile_custom import make_profile

    my_profile, my_fit = make_profile(lambda s: np.sin(np.pi * s))

    class MyCase(BenchmarkCase):
        def reference_fn(self, x0=0, y0=0, theta=0):
            return build_ideal_from_edge_profile(my_profile, n=9, arc_length=0.10,
                                                 x0=x0, y0=y0, theta=theta)
        def loss_fn(self, nodes, ps=None, q_ref=None):
            return profile_loss(nodes, my_profile, my_fit)

Exports:
    make_profile(kappa_fn)  — returns (profile_fn, fit_fn)
"""

from typing import Callable
import numpy as np


def make_profile(kappa_fn: Callable[[np.ndarray], np.ndarray]):
    """
    Wrap a user-supplied kappa(s) callable into a (profile_fn, fit_fn) pair.

    kappa_fn   — callable that takes s_norm (np.ndarray ∈ [0, 1]) and returns
                 turning angles (np.ndarray, same shape).

    fit_fn is a no-op that returns {} because the profile is fully specified
    by kappa_fn and has no free parameters to fit.
    """
    def profile_fn(s_norm: np.ndarray, **_) -> np.ndarray:
        return np.asarray(kappa_fn(s_norm), dtype=float)

    def fit_fn(s_norm: np.ndarray, dtheta: np.ndarray) -> dict:
        return {}

    return profile_fn, fit_fn
