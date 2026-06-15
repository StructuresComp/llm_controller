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
    nv: int = None,
) -> str:
    """
    Convert observable node positions + current BC into prose.

    Args:
        nodes  — list of (x, y) tuples (N_PARTIAL observable nodes)
        bc     — [x0, y0, theta] current boundary condition
        level  — "positions" | "geometric" | "full"
        nv     — total number of nodes (for arc-length percentage calculation)
    """
    x0, y0, theta = bc
    lines = [f"Clamped end (node 0): position ({x0:.4f}, {y0:.4f}) m, angle θ = {theta:.4f} rad"]
    lines.append("")
    lines.append("Observable points along the beam:")

    for j, (x, y) in enumerate(nodes):
        if nv is not None:
            # Compute arc-length percentage (node index as fraction of total length)
            # This requires knowing the actual observable node indices, which we don't have here.
            # So we'll just number them 1, 2, 3...
            pct = (j / (len(nodes) - 1) * 100) if len(nodes) > 1 else 0
            lines.append(f"  Point {j+1} ({pct:.0f}% along): x={x:+.6f} m, y={y:+.6f} m")
        else:
            lines.append(f"  Point {j+1}: x={x:+.6f} m, y={y:+.6f} m")

    if level in ("geometric", "full"):
        desc = _compute_descriptors(nodes)
        if desc:
            lines.append("")
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
        lines.append(f"Step {i+1}: moved node 0 → ({x0:.4f}, {y0:.4f}) θ={theta:.4f} rad")
        for j, (x, y) in enumerate(shape):
            pct = (j / (len(shape) - 1) * 100) if len(shape) > 1 else 0
            lines.append(f"         Point {j+1} ({pct:.0f}% along): x={x:+.6f} m, y={y:+.6f} m")
        if level in ("geometric", "full"):
            desc = _compute_descriptors(shape)
            if desc:
                lines.append(_format_descriptors(desc, indent="         "))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Geometric descriptor helpers
# ---------------------------------------------------------------------------

def _compute_descriptors(nodes: list) -> dict:
    """
    Compute edge tangent angles and relative turning angles (dtheta).
    Matches the calculation in profile_fitting.py for consistency with loss function.
    """
    if len(nodes) < 2:
        return {}
    obs = np.array(nodes)

    # Compute tangent angle at each edge (direction from node i to node i+1)
    edge_vecs = np.diff(obs, axis=0)
    theta_edges = np.arctan2(edge_vecs[:, 0], -edge_vecs[:, 1])  # CCW from -Y axis

    # Compute relative turning angle (dtheta) at each transition
    # This is what the loss function actually measures
    if len(theta_edges) > 1:
        theta_start = float(theta_edges[0])
        dtheta = (theta_edges - theta_start + math.pi) % (2 * math.pi) - math.pi
    else:
        dtheta = np.array([0.0])

    # Total rotation from first edge to last edge
    total_turn = float(theta_edges[-1]) - float(theta_edges[0])
    total_turn = (total_turn + math.pi) % (2 * math.pi) - math.pi

    return {
        "theta_edges":  [float(t) for t in theta_edges],
        "dtheta":       [float(d) for d in dtheta],
        "total_turning": total_turn,
    }


def _format_descriptors(desc: dict, indent: str = "") -> str:
    """Format geometric descriptors (edge tangents and relative turning angles)."""
    theta_edges = desc["theta_edges"]
    dtheta = desc["dtheta"]
    total = desc["total_turning"]

    lines = [f"{indent}Edge directions (tangent angle of each segment):"]
    for i, angle in enumerate(theta_edges):
        deg = math.degrees(angle)
        lines.append(f"{indent}  Segment {i+1}→{i+2}: {angle:.4f} rad = {deg:+.1f}°")

    lines.append(f"{indent}")
    lines.append(f"{indent}Relative turning (change in direction between segments):")
    for i, angle in enumerate(dtheta):
        deg = math.degrees(angle)
        turn_desc = "sharp left" if angle < -0.5 else ("sharp right" if angle > 0.5 else "slight")
        lines.append(f"{indent}  At segment {i+1}: {angle:.4f} rad = {deg:+.1f}° ({turn_desc})")

    lines.append(f"{indent}")
    total_deg = math.degrees(total)
    lines.append(f"{indent}Total rotation from first segment to last: {total:.4f} rad = {total_deg:+.1f}°")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Physics and constraint descriptions (for prompt_builder.py)
# ---------------------------------------------------------------------------

def describe_beam_physics(ps) -> str:
    """
    Beam properties with Euler-Bernoulli intuition (deflection under 1 N, gravity sag).
    Uses cantilever beam theory to give the LLM physical intuition about stiffness.
    """
    # Cantilever formula: δ = P * L³ / (3 * EI)
    # For P = 1 N: δ = L³ / (3 * EI)
    deflection_1N = ps.rod_length**3 / (3 * ps.EI)

    # Cantilever under own weight: δ = (m * g * L⁴) / (8 * EI)
    # where m = mass per unit length = ρ * A = ρ * π * r0²
    mass_per_length = np.pi * ps.r0**2 * ps.rho
    g = 9.8
    sag_under_gravity = (mass_per_length * g * ps.rod_length**4) / (8 * ps.EI)

    stiffness_category = "Stiff (high EI)" if deflection_1N < 0.01 else "Soft (low EI)"

    return f"""\
Beam properties:
  Length: {ps.rod_length:.4f} m
  Bending stiffness (EI): {ps.EI:.2e} N·m²
  Material: inextensible (does not stretch under tension)
  Movement: quasi-static (inertia is negligible)

Physical intuition:
  Under a 1 N horizontal tip force (no gravity): tip deflects {deflection_1N:.4f} m
  Under gravity alone (tip free): beam sags {sag_under_gravity:.4f} m
  Category: {stiffness_category} — {'harder to bend' if deflection_1N < 0.01 else 'easier to bend'}"""


def describe_wall_constraints(ps) -> str:
    """Wall description, or empty string if no walls."""
    if not ps.walls:
        return ""
    return f"Walls (physical obstacles):\n{describe_walls(ps.walls)}"


def describe_bc_constraints(ps) -> str:
    """Boundary condition explanation — concrete statement of what is controlled vs. fixed."""
    if ps.controlled_nodes == [0, 1]:
        # With walls: only clamped end is controlled; other end free but wall-constrained
        return """\
Boundary conditions:
  Controlled end (node 0): Position (x0, y0) and slope (θ) are set BY YOU.
  Free end: Deforms freely under gravity and walls. Walls prevent penetration but do NOT clamp the position."""
    else:
        # Without walls: both ends are clamped (tip is fixed in space); only the middle deforms
        # The fixed end stays at its initial position (0, -rod_length) with slope θ = 0
        fixed_x = 0.0
        fixed_y = -ps.rod_length
        fixed_theta = 0.0
        return f"""\
Boundary conditions:
  Controlled end (node 0): Position (x0, y0) and slope (θ) are set BY YOU.
  Fixed end (opposite end): Clamped at position ({fixed_x:.4f}, {fixed_y:.4f}) m with slope θ = {fixed_theta:.4f} rad (pointing straight down). This end DOES NOT MOVE."""


def describe_controller_feedback(reaction_force: list) -> str:
    """
    Controller feedback force felt at the clamped end (nodes 0 and 1).

    Args:
        reaction_force: Full reaction force vector (2*nv elements).
                       reaction_force[0:2] = force on node 0
                       reaction_force[2:4] = force on node 1

    Returns:
        String describing the net feedback force and its interpretation.
        Empty string if reaction_force is None.
    """
    if reaction_force is None or len(reaction_force) < 4:
        return ""

    # Extract forces on controlled nodes (0 and 1)
    F0 = np.array(reaction_force[0:2])  # force on node 0
    F1 = np.array(reaction_force[2:4])  # force on node 1

    # Total reaction from beam on the two controlled nodes
    F_total = F0 + F1

    # What the controller feels (negative of reaction)
    F_feedback = -F_total

    Fx, Fy = F_feedback[0], F_feedback[1]
    magnitude = np.sqrt(Fx**2 + Fy**2)

    # Direction description
    directions = []
    if Fx > 1e-6:
        directions.append("pushing right")
    elif Fx < -1e-6:
        directions.append("pushing left")

    if Fy > 1e-6:
        directions.append("pushing up")
    elif Fy < -1e-6:
        directions.append("pushing down")

    direction_str = " and ".join(directions) if directions else "no net force"

    return f"""\
Controller feedback (what the arm is feeling):
  Net force: [{Fx:+.3f}, {Fy:+.3f}] N
  Magnitude: {magnitude:.3f} N ({direction_str})"""


def describe_contact_regions(cx: list | None, cy: list | None, nv: int, walls: list) -> str:
    """
    Describe which parts of the beam are in contact with walls (by arc-length).

    Args:
        cx: Contact state for x-DOF. cx[i] = wall index or -1 (free). None if no walls.
        cy: Contact state for y-DOF. cy[i] = wall index or -1 (free). None if no walls.
        nv: Number of nodes.
        walls: List of wall dicts with keys 'n' (normal), 'd' (distance).

    Returns:
        String describing contact regions by arc-length percentage.
        Empty string if no contact or cx/cy are None.
    """
    if cx is None and cy is None:
        return ""

    if not walls:
        return ""

    # Helper to describe wall by index
    def wall_description(wall_idx):
        if wall_idx < 0 or wall_idx >= len(walls):
            return "unknown wall"
        wall = walls[wall_idx]
        n = wall["n"]
        if abs(n[1]) > abs(n[0]):  # horizontal wall
            return f"horizontal wall at y={wall['d']/n[1]:.4f} m"
        else:  # vertical wall
            return f"vertical wall at x={wall['d']/n[0]:.4f} m"

    # Convert node indices to arc-length and group regions
    lines = ["Wall contact status:"]

    # X-contact regions
    if cx is not None:
        x_contact = [i for i, wall_idx in enumerate(cx) if wall_idx >= 0]
        if x_contact:
            regions_x = []
            start = x_contact[0]
            for i in range(1, len(x_contact)):
                if x_contact[i] != x_contact[i-1] + 1:  # Gap detected
                    end = x_contact[i-1]
                    pct_start = (start / (nv - 1)) * 100
                    pct_end = (end / (nv - 1)) * 100
                    wall = cx[start]
                    regions_x.append((pct_start, pct_end, wall))
                    start = x_contact[i]
            # Final region
            end = x_contact[-1]
            pct_start = (start / (nv - 1)) * 100
            pct_end = (end / (nv - 1)) * 100
            wall = cx[start]
            regions_x.append((pct_start, pct_end, wall))

            for pct_start, pct_end, wall_idx in regions_x:
                wall_desc = wall_description(wall_idx)
                lines.append(f"  Beam at {pct_start:.0f}%-{pct_end:.0f}% arc-length: x-DOF constrained ({wall_desc})")

    # Y-contact regions
    if cy is not None:
        y_contact = [i for i, wall_idx in enumerate(cy) if wall_idx >= 0]
        if y_contact:
            regions_y = []
            start = y_contact[0]
            for i in range(1, len(y_contact)):
                if y_contact[i] != y_contact[i-1] + 1:  # Gap detected
                    end = y_contact[i-1]
                    pct_start = (start / (nv - 1)) * 100
                    pct_end = (end / (nv - 1)) * 100
                    wall = cy[start]
                    regions_y.append((pct_start, pct_end, wall))
                    start = y_contact[i]
            # Final region
            end = y_contact[-1]
            pct_start = (start / (nv - 1)) * 100
            pct_end = (end / (nv - 1)) * 100
            wall = cy[start]
            regions_y.append((pct_start, pct_end, wall))

            for pct_start, pct_end, wall_idx in regions_y:
                wall_desc = wall_description(wall_idx)
                lines.append(f"  Beam at {pct_start:.0f}%-{pct_end:.0f}% arc-length: y-DOF constrained ({wall_desc})")

    return "\n".join(lines) if len(lines) > 1 else ""


def describe_error_rejection(error_msg: str) -> str:
    """
    Format a BC rejection error to alert the LLM.
    Only shown if last BC was rejected by safety checks.
    """
    if not error_msg:
        return ""
    return f"""\
⚠️  PREVIOUS MOVE REJECTED — DO NOT REPEAT

Your last boundary condition was rejected because:
  {error_msg}

Please adjust your values to satisfy the constraint and try again."""


def describe_recent_trajectory(bc_history: list, shape_history: list, n_recent: int = 5) -> str:
    """
    Recent trajectory: last N moves showing progress or stagnation (arc-length, no node indices).
    Shows position deltas to give the LLM a sense of direction and pace of change.
    """
    if not bc_history or not shape_history:
        return "(no history yet)"

    # Get last N moves
    n = min(n_recent, len(bc_history))
    recent_bcs = bc_history[-n:]
    recent_shapes = shape_history[-n:]

    lines = [f"Recent trajectory (last {n} moves):"]

    # For each move, compute displacement of clamped end
    for i, (bc, shape) in enumerate(zip(recent_bcs, recent_shapes)):
        x0, y0, theta = bc
        lines.append(f"  Move {i+1}: BC→({x0:+.4f}, {y0:+.4f}, θ={theta:+.4f} rad)")
        if shape:
            # Compute tip displacement from previous move (if exists)
            if i > 0:
                prev_shape = recent_shapes[i-1]
                if len(shape) > 0 and len(prev_shape) > 0:
                    tip_x, tip_y = shape[-1]
                    prev_tip_x, prev_tip_y = prev_shape[-1]
                    dx = tip_x - prev_tip_x
                    dy = tip_y - prev_tip_y
                    distance = np.sqrt(dx**2 + dy**2)
                    direction = "down" if dy < -1e-6 else ("up" if dy > 1e-6 else "")
                    direction += ("-left" if dx < -1e-6 else ("-right" if dx > 1e-6 else ""))
                    lines.append(f"            → tip moved {distance:.4f} m {direction.strip()}")

    return "\n".join(lines)


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
