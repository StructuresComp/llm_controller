# ---------------------------------------------------------------------------
# src/core/observation.py
# ---------------------------------------------------------------------------
# Converts physical beam state into natural language for LLM prompts.
#
# Public API:
#   describe_walls(walls)                        -> str
#   describe_nodes(nodes, bc, level)             -> str
#   describe_history(bc_history, shape_history, level) -> str
#
# Run standalone to inspect prompt text for hand-typed configurations:
#   python -m src.core.observation
# ---------------------------------------------------------------------------

import math

import numpy as np

from src.core import config as _config


# ---------------------------------------------------------------------------
# Wall description
# ---------------------------------------------------------------------------

def describe_walls(walls: list) -> str:
    """
    Convert a list of wall dicts (processed format: n, d, P0, P1) into
    a flat list of geometric descriptions, one per segment.
    """
    if not walls:
        return "none"
    return "; ".join(_describe_segment(w) for w in walls)


def _describe_segment(w: dict) -> str:
    """Single axis-aligned wall segment → geometric string with allowed-side label."""
    n, d = w["n"], w["d"]
    P0, P1 = w["P0"], w["P1"]
    if n[1] > 0.5:       # normal points up   → floor
        return f"floor at y={d / n[1]:.4f} m, beam must stay above (x: {P0[0]:.3f} to {P1[0]:.3f})"
    elif n[1] < -0.5:    # normal points down → ceiling
        return f"ceiling at y={d / n[1]:.4f} m, beam must stay below (x: {P0[0]:.3f} to {P1[0]:.3f})"
    elif n[0] > 0.5:     # normal points right → left wall
        return f"left wall at x={d / n[0]:.4f} m, beam must stay to the right (y: {P0[1]:.3f} to {P1[1]:.3f})"
    else:                # normal points left  → right wall
        return f"right wall at x={d / n[0]:.4f} m, beam must stay to the left (y: {P0[1]:.3f} to {P1[1]:.3f})"


# ---------------------------------------------------------------------------
# Node / configuration description
# ---------------------------------------------------------------------------

def describe_nodes(
    nodes: list,
    bc: list,
    level: str = "positions",
) -> str:
    """
    Convert observable node positions + current BC into prose.

    Args:
        nodes  — list of (x, y) tuples (N_PARTIAL observable nodes)
        bc     — [x0, y0, theta] current boundary condition
        level  — "positions" | "geometric" | "full"
    """
    x0, y0, theta = bc
    lines = [f"BC: ({x0:.4f}, {y0:.4f}) θ={theta:.4f} rad"]
    for j, (x, y) in enumerate(nodes):
        lines.append(f"obs[{j}]: ({x:+.6f}, {y:+.6f})")
    if level in ("geometric", "full"):
        desc = _compute_descriptors(nodes)
        if desc:
            lines.append(_format_descriptors(desc))
    return "\n".join(lines)


def describe_history(
    bc_history: list,
    shape_history: list,
    level: str = "positions",
) -> str:
    """
    Format rolling (BC, observable nodes) history as prose.

    Args:
        bc_history    — list of [x0, y0, theta]
        shape_history — list of lists of (x, y)
        level         — observation level (same as describe_nodes)
    """
    if not bc_history:
        return "(no history yet)"
    lines = []
    for i, (bc, shape) in enumerate(zip(bc_history, shape_history)):
        x0, y0, theta = bc
        lines.append(f"Step {i+1}: node0 → ({x0:.4f}, {y0:.4f}) θ={theta:.4f} rad")
        for j, (x, y) in enumerate(shape):
            lines.append(f"         obs[{j}]: ({x:+.6f}, {y:+.6f})")
        if level in ("geometric", "full"):
            desc = _compute_descriptors(shape)
            if desc:
                lines.append(_format_descriptors(desc, indent="         "))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Geometric descriptor helpers
# ---------------------------------------------------------------------------

def _compute_descriptors(nodes: list) -> dict:
    """Tangent angles, curvature signs, total turning from observable positions."""
    if len(nodes) < 2:
        return {}
    obs = np.array(nodes)

    tangents = []
    for i in range(len(obs) - 1):
        dx = obs[i+1][0] - obs[i][0]
        dy = obs[i+1][1] - obs[i][1]
        tangents.append(float(np.arctan2(dx, -dy)))   # CCW from -Y axis

    curvature_signs = []
    for i in range(len(tangents) - 1):
        dangle = tangents[i+1] - tangents[i]
        # Wrap to [-pi, pi]
        dangle = (dangle + math.pi) % (2 * math.pi) - math.pi
        curvature_signs.append(1 if dangle > 1e-6 else (-1 if dangle < -1e-6 else 0))

    total = tangents[-1] - tangents[0]
    total = (total + math.pi) % (2 * math.pi) - math.pi

    return {
        "tangent_angles":  tangents,
        "curvature_signs": curvature_signs,
        "total_turning":   total,
    }


def _format_descriptors(desc: dict, indent: str = "") -> str:
    angles = [f"{a:.3f}" for a in desc["tangent_angles"]]
    return (
        f"{indent}tangents: {angles}\n"
        f"{indent}curvature signs: {desc['curvature_signs']}\n"
        f"{indent}total turning: {desc['total_turning']:.4f} rad"
    )


# ---------------------------------------------------------------------------
# Standalone test — python -m src.core.observation
# ---------------------------------------------------------------------------

_TEST_CASES = [
    {
        "label": "Straight beam (initial position)",
        "bc":    [0.0, 0.0, 0.0],
        "nodes": [(0.0, 0.0), (0.0, -0.025), (0.0, -0.05), (0.0, -0.075), (0.0, -0.10)],
        "walls": [{"n": [0., 1.], "d": -0.12,
                   "P0": [-10., -0.12], "P1": [10., -0.12]}],
        "level": "geometric",
    },
    {
        "label": "C-shape in progress",
        "bc":    [0.0, -0.03, 0.4],
        "nodes": [(0.0, -0.03), (0.015, -0.05), (0.025, -0.075),
                  (0.02, -0.10), (0.005, -0.115)],
        "walls": [{"n": [0., 1.], "d": -0.12,
                   "P0": [-10., -0.12], "P1": [10., -0.12]}],
        "level": "geometric",
    },
    {
        "label": "Snake-L: beam entering vertical channel",
        "bc":    [0.0, -0.04, 0.0],
        "nodes": [(0.0, -0.04), (0.0, -0.065), (0.0, -0.09),
                  (0.005, -0.105), (0.02, -0.11)],
        "walls": [
            {"n": [ 1.,  0.], "d": -0.01, "P0": [-0.01,  0.02], "P1": [-0.01, -0.12]},
            {"n": [ 0.,  1.], "d": -0.12, "P0": [-0.01, -0.12], "P1": [ 0.12, -0.12]},
            {"n": [-1.,  0.], "d": -0.12, "P0": [ 0.12, -0.12], "P1": [ 0.12, -0.10]},
            {"n": [ 0., -1.], "d":  0.10, "P0": [ 0.12, -0.10], "P1": [ 0.01, -0.10]},
            {"n": [-1.,  0.], "d": -0.01, "P0": [ 0.01, -0.10], "P1": [ 0.01,  0.02]},
        ],
        "level": "geometric",
    },
]


def main():
    sep = "=" * 60
    for tc in _TEST_CASES:
        print(f"\n{sep}")
        print(f"TEST: {tc['label']}")
        print(f"{sep}")
        print(f"Wall description:\n  {describe_walls(tc['walls'])}\n")
        print("Node description:")
        print(describe_nodes(tc["nodes"], tc["bc"], level=tc["level"]))


if __name__ == "__main__":
    main()
