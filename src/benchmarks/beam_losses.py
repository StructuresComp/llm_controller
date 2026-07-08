"""
src/benchmarks/beam_losses.py
------------------------------
Loss functions and their natural-language companions for beam benchmarks.

Each loss type has two functions:
    compute_*  — returns float (error_percent); used by evaluator and case.py
    describe_* — returns str  (LLM-ready description of target vs current);
                 used by prompt_builder via benchmark.describe_fn(q, ps)

    SECTION 1 — CURVATURE LOSS
        compute_curvature_loss(q, kappa_ideal, ps)  -> float
        describe_curvature_loss(q, ps, benchmark)   -> str

    SECTION 2 — NODE LOSS
        build_ideal_from_edge_profile(kappa_fn, n, rod_length, x0, y0, theta)
        compute_node_loss(q, kappa_fn, ps)          -> float
        describe_node_loss(q, ps, benchmark)        -> str

    SECTION 3 — TIP LOSS
        compute_tip_loss(nodes, target_x, target_y, rod_length) -> float
        describe_tip_loss(q, ps, benchmark)                     -> str
"""

import numpy as np
from typing import Callable

from src.core import config
from src.tools.Discrete_Elastic_Beams import compute_kappa_dimensional


# ===========================================================================
# SECTION 1 — CURVATURE LOSS
# ===========================================================================

def compute_curvature_loss(q: np.ndarray, kappa_ideal, ps=None) -> float:
    """
    RMS curvature error normalised by ideal RMS curvature.

    kappa_ideal — callable kappa_fn(s_norm, rod_length)
                  OR pre-computed ndarray (interpolated to match q's resolution).
    """
    nv         = ps.nv     if ps else len(q) // 2
    deltaL     = ps.deltaL if ps else 1.0 / (nv - 1)
    rod_length = ps.rod_length if ps else 1.0

    if nv < 3:
        return 100.0

    kappa_actual = compute_kappa_dimensional(q, nv, deltaL)
    s_values     = np.arange(1, nv - 1) / (nv - 1)

    if callable(kappa_ideal):
        ideal = kappa_ideal(s_values, rod_length=rod_length)
    else:
        arr   = np.asarray(kappa_ideal)
        s_ref = np.arange(len(arr)) / (len(arr) - 1)
        ideal = np.interp(s_values, s_ref, arr)

    actual    = kappa_actual[1:-1]
    rms_ideal = float(np.sqrt(np.mean(ideal ** 2)))
    if rms_ideal < 1e-10:
        raise ValueError(
            f"compute_curvature_loss: ideal curvature is near-zero (rms={rms_ideal:.2e}). "
            "Check kappa source."
        )
    return float(np.sqrt(np.mean((actual - ideal) ** 2))) / rms_ideal * 100.0


def describe_curvature_loss(q, ps, benchmark) -> str:
    """Target vs current curvature at observable positions, plus overall error."""
    q = np.asarray(q)
    nv  = ps.nv
    obs = ps.observable_nodes
    interior = [i for i in obs if 0 < i < nv - 1]

    kappa_actual = compute_kappa_dimensional(q, nv, ps.deltaL)
    kappa_ideal  = benchmark.reference_kappa_fn

    if interior:
        s_values = np.array([i / (nv - 1) for i in interior])
        if callable(kappa_ideal):
            ideal = kappa_ideal(s_values, rod_length=ps.rod_length)
        else:
            arr   = np.asarray(kappa_ideal)
            s_ref = np.arange(len(arr)) / (len(arr) - 1)
            ideal = np.interp(s_values, s_ref, arr)

    error = compute_curvature_loss(q, kappa_ideal, ps)

    lines = [
        f"Target: {benchmark.target_description}",
        "Loss: RMS curvature deviation normalised by ideal RMS curvature. 0% = perfect match.",
        "",
    ]
    if interior:
        lines.append("Target curvatures along the DLO (vs current):")
        for k, node_idx in enumerate(interior):
            s   = node_idx / (nv - 1)
            cur = kappa_actual[node_idx]
            tgt = ideal[k]
            lines.append(
                f"Arc-length  s={s:.2f}:  "
                f"current curvature = {cur:+.1f} /m,  "
                f"target curvature = {tgt:+.1f} /m,  "
                f"gap = {cur - tgt:+.1f} /m"
            )
    if config.SHOW_ERROR_PERCENT:
        lines.append(f"Overall error: {error:.1f}%  (0% = perfect curvature match)")
    return "\n".join(lines)


# ===========================================================================
# SECTION 2 — NODE LOSS
# ===========================================================================

def build_ideal_from_edge_profile(
    kappa_fn: Callable,
    n: int,
    rod_length: float,
    x0: float = 0.0,
    y0: float = 0.0,
    theta: float = -np.pi / 2,
    **params,
) -> np.ndarray:
    """
    Generate n reference nodes (shape (n, 2)) by integrating kappa_fn along arc length.

    kappa_fn(s_norm, rod_length, **params) → curvature at normalised arc-length s ∈ [0, 1].
    theta follows standard math convention: CCW from +X. -pi/2 = pointing straight down.
    x0, y0 anchor the shape at node 0's actual position.
    """
    ne = n - 1
    dL = rod_length / ne
    q  = np.zeros((n, 2))
    q[0] = [x0, y0]
    q[1] = q[0] + dL * np.array([np.cos(theta), np.sin(theta)])
    for i in range(2, n):
        s     = (i - 1) / ne
        kappa = kappa_fn(np.array([s]), rod_length=rod_length, **params)[0]
        phi   = 2.0 * np.arctan(kappa * dL / 2.0)
        c, s_ = np.cos(phi), np.sin(phi)
        e_prev = q[i - 1] - q[i - 2]
        q[i]   = q[i - 1] + np.array([[c, -s_], [s_, c]]) @ e_prev
    return q


def compute_node_loss(q: np.ndarray, kappa_fn: Callable, ps=None) -> float:
    """
    RMS positional error normalised by arc length of the reference shape.

    Builds reference nodes from kappa_fn anchored at actual node 0's position and slope.
    Compares all nv actual nodes to reference.
    """
    nv         = ps.nv         if ps else len(q) // 2
    rod_length = ps.rod_length if ps else 1.0

    actual = q.reshape(nv, 2)
    x0, y0 = float(actual[0, 0]), float(actual[0, 1])
    theta = float(np.arctan2(actual[1, 1] - actual[0, 1], actual[1, 0] - actual[0, 0]))

    ref        = build_ideal_from_edge_profile(kappa_fn, nv, rod_length, x0=x0, y0=y0, theta=theta)
    arc_length = float(np.sum(np.linalg.norm(np.diff(ref, axis=0), axis=1)))
    if arc_length < 1e-10:
        return 100.0
    rms = float(np.sqrt(np.mean(np.sum((actual - ref) ** 2, axis=1))))
    return rms / arc_length * 100.0


def describe_node_loss(q, ps, benchmark) -> str:
    """Target vs current node positions at observable points, plus overall error."""
    q = np.asarray(q)
    nv  = ps.nv
    obs = ps.observable_nodes

    actual = q.reshape(nv, 2)
    x0, y0 = float(actual[0, 0]), float(actual[0, 1])
    theta  = float(np.arctan2(actual[1, 1] - actual[0, 1], actual[1, 0] - actual[0, 0]))
    ref    = build_ideal_from_edge_profile(
        benchmark.reference_kappa_fn, nv, ps.rod_length, x0=x0, y0=y0, theta=theta
    )
    error = compute_node_loss(q, benchmark.reference_kappa_fn, ps)

    lines = [
        f"Target: {benchmark.target_description}",
        "Loss: RMS positional error normalised by reference arc length. 0% = perfect match.",
        "",
    ]
    lines.append("Node positions at observable points (target vs current):")
    for k, node_idx in enumerate(obs):
        pct    = node_idx / (nv - 1) * 100
        cx, cy = float(actual[node_idx, 0]), float(actual[node_idx, 1])
        tx, ty = float(ref[node_idx, 0]),    float(ref[node_idx, 1])
        dist   = float(np.hypot(cx - tx, cy - ty))
        label  = " [clamped]" if k == 0 else (" [tip]" if k == len(obs) - 1 else "")
        lines.append(
            f"  {pct:.0f}%{label}:  "
            f"current ({cx:+.4f}, {cy:+.4f}),  "
            f"target ({tx:+.4f}, {ty:+.4f}),  "
            f"dist = {dist:.4f} m"
        )
    if config.SHOW_ERROR_PERCENT:
        lines.append(f"Overall error: {error:.1f}%  (0% = perfect position match)")
    return "\n".join(lines)


# ===========================================================================
# SECTION 3 — TIP LOSS
# ===========================================================================

def compute_tip_loss(
    nodes: list,
    target_x: float,
    target_y: float,
    rod_length: float,
) -> float:
    """Euclidean distance from the tip to the target, normalised by rod_length."""
    tip_x, tip_y = nodes[-1]
    dist = np.linalg.norm([tip_x - target_x, tip_y - target_y])
    return dist / rod_length * 100.0


def describe_tip_loss(q, ps, benchmark) -> str:
    """Target vs current tip position, plus distance as % of rod length."""
    q = np.asarray(q)
    target_x, target_y = benchmark.reference_tip_fn
    cx, cy = float(q[-2]), float(q[-1])
    dist   = float(np.hypot(cx - target_x, cy - target_y))
    error  = dist / ps.rod_length * 100.0

    lines = [
        f"Target: {benchmark.target_description}",
        "Loss: Euclidean tip-to-target distance as % of rod length. 0% = tip exactly at target.",
        "",
        f"Tip position:  current ({cx:+.4f}, {cy:+.4f}) m",
        f"Target:        ({target_x:+.4f}, {target_y:+.4f}) m",
        f"Distance:      {dist:.4f} m" + (f"  ({error:.1f}% of rod length)" if config.SHOW_ERROR_PERCENT else ""),
    ]
    return "\n".join(lines)
