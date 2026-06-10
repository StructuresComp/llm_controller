"""
Tip-reach goal: drive the free end to a target position.

No curvature profile — only the last node's (x, y) position matters.

Exports:
    tip_reach_loss(nodes, target_x, target_y, rod_length)  — dict with error_percent + diagnostics
"""

import numpy as np


def tip_reach_loss(
    nodes: list,
    target_x: float,
    target_y: float,
    rod_length: float,
) -> dict:
    """
    Euclidean distance from the tip node to the target, normalised by rod_length.

    nodes      — list of (x, y) observable positions; nodes[-1] is the free tip.
    target_x/y — goal position.
    rod_length — used for normalisation; prevents scale-dependence.
    """
    tip_x, tip_y = nodes[-1]
    dist = float(np.hypot(tip_x - target_x, tip_y - target_y))
    return {
        "error_percent": min(dist / rod_length * 100.0, 100.0),
        "diagnostics": (
            f"tip=({tip_x:+.4f}, {tip_y:+.4f})  "
            f"target=({target_x:+.4f}, {target_y:+.4f})  "
            f"dist={dist:.4f} m"
        ),
    }
