# ---------------------------------------------------------------------------
# src/tools/walls.py
# ---------------------------------------------------------------------------
# Wall geometry utilities shared by all physics tools and the safety agent.
#
# A "wall" is a rigid surface that the beam/rod cannot penetrate.  Each wall
# is represented internally as a half-space (defined by an outward unit normal
# n and offset d) clipped to a finite extent (the original segment or patch).
#
# Two implementations are provided:
#   2D  — line segments in the XY plane (used by Discrete_Elastic_Beams.py)
#   3D  — planar patches in XYZ space   (used by Discrete_Elastic_Rods.py — stubs only)
#
# Public API
# ----------
# 2D:
#   make_wall_2d(spec)              dict  — convert user spec to internal form
#   signed_dist_2d(px, py, wall)    float — positive = safe side, negative = penetrated
#   penetrated_2d(px, py, wall)     bool  — True if node has crossed into the wall
#   plot_wall_2d(ax, wall)                — draw one wall segment on a matplotlib Axes
#   plot_walls_2d(ax, walls)              — draw all walls
#
# 3D (stubs — implement when rod support lands):
#   make_wall_3d(spec)
#   signed_dist_3d(px, py, pz, wall)
#   penetrated_3d(px, py, pz, wall)
#   plot_wall_3d(ax, wall)
#   plot_walls_3d(ax, walls)
#
# Wall spec formats
# -----------------
# 2D user-facing:  {"P0": [x0, y0], "P1": [x1, y1]}
#   Segment from P0 to P1.  Outward normal = (P1-P0) rotated 90° CCW, so
#   traversal order determines which side is the allowed region.
#   Use large sentinel coordinates (e.g. ±100) for near-infinite walls.
#
# 3D user-facing (planned):  {"P0": [x,y,z], "P1": [x,y,z], "P2": [x,y,z]}
#   Three non-collinear points define the plane and patch extent.
#   Outward normal = (P1-P0) × (P2-P0), normalised.
# ---------------------------------------------------------------------------

import numpy as np


# ===========================================================================
# 2D — LINE SEGMENT WALLS
# ===========================================================================

def make_wall_2d(spec: dict) -> dict:
    """
    Convert a user-facing 2D wall spec to the internal {n, d, P0, P1} form.

    The outward normal is the edge vector (P1-P0) rotated 90° CCW.
    A point p is on the safe side when n·p >= d; it has penetrated when n·p < d.

    Internal fields:
      n  — outward unit normal (2D numpy array)
      d  — scalar offset: n·P0 = d
      P0 — start endpoint (2D numpy array)
      P1 — end  endpoint (2D numpy array)
    """
    P0 = np.array(spec["P0"], dtype=float)
    P1 = np.array(spec["P1"], dtype=float)
    e  = P1 - P0
    n  = np.array([-e[1], e[0]])
    n  = n / np.linalg.norm(n)
    d  = float(np.dot(n, P0))
    return {"n": n, "d": d, "P0": P0, "P1": P1}


def signed_dist_2d(px: float, py: float, wall: dict) -> float:
    """
    Signed distance from point (px, py) to the wall's half-plane.

    Positive = safe (outward-normal) side.
    Negative = penetrated (forbidden) side.
    Zero      = exactly on the wall surface.

    Does not check segment extent — use penetrated_2d for a full contact test.
    """
    n, d = wall["n"], wall["d"]
    return float(n[0] * px + n[1] * py - d)


def within_extent_2d(px: float, py: float, wall: dict) -> bool:
    """
    Return True if the node's projection lands on the wall segment (not past its endpoints).

    Scalar parameter:
        t = (p - P0) · (P1 - P0) / |P1 - P0|²
    t ∈ [0, 1] → projection lands on the segment.
    t < 0 or t > 1 → node is past an endpoint → not this wall's responsibility.

    This check is independent of which side of the wall the node is on.
    """
    P0, P1 = wall["P0"], wall["P1"]
    seg = P1 - P0
    t = ((px - P0[0]) * seg[0] + (py - P0[1]) * seg[1]) / np.dot(seg, seg)
    return 0.0 <= t <= 1.0


def penetrated_2d(px: float, py: float, wall: dict) -> bool:
    """
    Return True if node at (px, py) has penetrated this wall segment.

    Both conditions must hold:
      1. Node is on the forbidden side: signed_dist_2d < 0.
      2. Node projects onto the segment: within_extent_2d is True.
    """
    return signed_dist_2d(px, py, wall) < 0 and within_extent_2d(px, py, wall)


def plot_wall_2d(ax, wall: dict) -> None:
    """Draw one 2D wall segment on a matplotlib Axes."""
    P0, P1 = wall["P0"], wall["P1"]
    ax.plot([P0[0], P1[0]], [P0[1], P1[1]],
            color='r', linestyle='--', linewidth=1.5)


def plot_walls_2d(ax, walls: list) -> None:
    """Draw all 2D wall segments on a matplotlib Axes."""
    for wall in walls:
        plot_wall_2d(ax, wall)


# ===========================================================================
# 3D — PLANAR PATCH WALLS  (stubs — implement with rod support)
# ===========================================================================

def make_wall_3d(spec: dict) -> dict:
    """
    Convert a user-facing 3D wall spec to the internal {n, d, P0, P1, P2} form.

    Spec: {"P0": [x,y,z], "P1": [x,y,z], "P2": [x,y,z]}
    Three non-collinear points define the plane.
    Outward normal = (P1-P0) × (P2-P0), normalised.
    Extent is the triangle P0-P1-P2; extend to a rectangle or polygon as needed.

    NOT IMPLEMENTED — stub for when rod support lands.
    """
    raise NotImplementedError("3D wall support is not yet implemented.")


def signed_dist_3d(px: float, py: float, pz: float, wall: dict) -> float:
    """
    Signed distance from point (px, py, pz) to the wall's half-space.

    Same semantics as signed_dist_2d: positive = safe, negative = penetrated.

    NOT IMPLEMENTED — stub for when rod support lands.
    """
    raise NotImplementedError("3D wall support is not yet implemented.")


def penetrated_3d(px: float, py: float, pz: float, wall: dict) -> bool:
    """
    Return True if node at (px, py, pz) has penetrated this wall patch.

    Same two-test structure as penetrated_2d:
      Test 1 — half-space: signed_dist_3d < 0.
      Test 2 — patch extent: point projects inside the wall polygon.
                In 3D this requires a 2D point-in-polygon test after projecting
                onto the wall plane (e.g. barycentric coordinates for a triangle).

    NOT IMPLEMENTED — stub for when rod support lands.
    """
    raise NotImplementedError("3D wall support is not yet implemented.")


def plot_wall_3d(ax, wall: dict) -> None:
    """
    Draw one 3D wall patch on a matplotlib Axes3D.

    NOT IMPLEMENTED — stub for when rod support lands.
    """
    raise NotImplementedError("3D wall support is not yet implemented.")


def plot_walls_3d(ax, walls: list) -> None:
    """
    Draw all 3D wall patches on a matplotlib Axes3D.

    NOT IMPLEMENTED — stub for when rod support lands.
    """
    for wall in walls:
        plot_wall_3d(ax, wall)
