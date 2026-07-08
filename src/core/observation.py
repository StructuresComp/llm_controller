# ---------------------------------------------------------------------------
# src/core/observation.py
# ---------------------------------------------------------------------------
# Converts physical beam state into natural language for LLM prompts.
#
# Public API:
#   describe_wall_constraints(ps)                      -> str
#   describe_current_shape(q, ps, kappa=None)          -> str
#   describe_history(bc_history, shape_history, level) -> str
#
# Run standalone to inspect prompt text for hand-typed configurations:
#   python -m src.core.observation
# ---------------------------------------------------------------------------

import numpy as np

from src.core import config

# ---------------------------------------------------------------------------
# Node / configuration description
# ---------------------------------------------------------------------------

def describe_current_shape(q: list, ps, kappa: np.ndarray = None, bc: list = None, decision_num: int = 0) -> str:
    """
    Current beam state as natural language for the LLM prompt.

    bc           — [x0, y0, theta] active boundary condition; shown first for consistency with describe_history.
    decision_num — shown in the header as "Current step (decision N):".
    Each observable node reported at arc-length s in [0, 1].
    Curvature appended for interior nodes when kappa is provided (geometric observation level).
    """
    obs = ps.observable_nodes
    nv  = ps.nv
    lines = [f"Current step (decision number: {decision_num}):"]
    if bc is not None:
        lines.append(f"  BC (control decision): (x0={bc[0]:+.4f}, y0={bc[1]:+.4f}, theta={bc[2]:+.4f} rad)")
    lines.append("  Shape of the DLO:")
    for node_idx in obs:
        s   = node_idx / (nv - 1)
        x   = float(q[2 * node_idx])
        y   = float(q[2 * node_idx + 1])
        line = f"  - s={s:.2f}: position = ({x:+.6f}, {y:+.6f}) m"
        if kappa is not None and 0 < node_idx < nv - 1:
            line += f",  curvature = {kappa[node_idx]:+.2f} /meter"
        lines.append(line)
    return "\n".join(lines)


def describe_history(
    bc_history: list,
    shape_history: list,
    ps,
    error_history: list,
    kappa_history: list = None,
) -> str:
    """
    Previous control steps paired with their resulting beam configurations.
    The current step is excluded by the caller (shape_history[-(k+1):-1]) to avoid
    duplicating describe_current_shape, which shows the current state with richer info.

    All non-None lists must have the same length as shape_history.
    """
    if not shape_history:
        return ""

    n = len(shape_history)
    assert len(bc_history) == n, f"bc_history length {len(bc_history)} != shape_history length {n}"
    if kappa_history is not None:
        assert len(kappa_history) == n, f"kappa_history length {len(kappa_history)} != {n}"
    assert len(error_history) == n, f"error_history length {len(error_history)} != {n}"

    obs = ps.observable_nodes
    nv  = ps.nv
    lines = [
        f"Here are the previous {n} decisions (control steps) you made and the resulting DLO shapes:",
        "Use this to judge whether your moves are progressing toward the target or stagnating.",
    ]
    for i, shape in enumerate(shape_history):
        bc     = bc_history[i]
        kappas = kappa_history[i] if kappa_history is not None else None
        lines.append(f"{n - i} step(s) ago:")
        lines.append(f"  BC (control decision): ({bc[0]:+.4f}, {bc[1]:+.4f}, theta={bc[2]:+.4f} rad)")
        lines.append(f"  Your reasoning behind the decision: {bc[3]}")
        lines.append(f"  Shape of the DLO:")
        for j, (x, y) in enumerate(shape):
            node_idx = obs[j]
            s = node_idx / (nv - 1)
            line = f"  - s={s:.2f}: position = ({x:+.6f}, {y:+.6f}) m"
            if kappas is not None and 0 < node_idx < nv - 1:
                line += f",  curvature = {kappas[j]:+.2f} /meter"
            lines.append(line)
        if config.SHOW_ERROR_PERCENT:
            lines.append(f"  Error: {error_history[i]:.1f}%")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Physics and constraint descriptions (for prompt_builder.py)
# ---------------------------------------------------------------------------

def describe_beam_physics(ps) -> str:
    """
    Beam properties with Euler-Bernoulli intuition (deflection under 1 N, gravity sag).
    Uses cantilever beam theory to give the LLM physical intuition about stiffness.
    """
    # Cantilever formula: delta = P * L^3 / (3 * EI)
    # For P = 1 N: delta = L^3 / (3 * EI)
    deflection_1N = ps.rod_length**3 / (3 * ps.EI)

    # Cantilever under own weight: delta = (m * g * L^4) / (8 * EI)
    # where m = mass per unit length = ρ * A = ρ * π * r0^2
    mass_per_length = np.pi * ps.r0**2 * ps.rho
    g = abs(config.GRAVITY[1])
    sag_under_gravity = (mass_per_length * g * ps.rod_length**4) / (8 * ps.EI)

    return f"""\
Beam properties:
  Length: {ps.rod_length:.4f} m
  Bending stiffness (EI): {ps.EI:.2e} Nm^2
  Axial stiffness (EA): {ps.EA:.2e} N — resists stretching, but axial strain occurs when both ends are constrained and the geometry demands it.
  Movement: quasi-static (inertia is negligible)

Physical intuition:
  Under a 1 N horizontal tip force (no gravity): tip deflects {deflection_1N:.4f} m.
  Gravity can deform the beam — account for it in your plan if relevant.
  A cantilever held horizontal (gravity perpendicular to it): tip sags {sag_under_gravity:.4f} m under its own weight."""

def describe_wall_constraints(ps) -> str:
    """Format wall geometry and friction for the LLM prompt. Returns "" if no walls."""
    if not ps.walls:
        return ""
    inf_threshold = ps.rod_length * config.WALL_INF_MULTIPLIER
    friction_str = (
        "Perfectly sticky (beam cannot slide once in contact — both x and y constrained)"
        if ps.friction else
        "Perfectly slippery (beam slides freely along wall surfaces — only normal direction constrained)"
    )
    lines = [f"Walls (physical obstacles): {len(ps.walls)} total. Friction: {friction_str}"]
    for i, w in enumerate(ps.walls):
        n, d = w["n"], w["d"]
        P0, P1 = w["P0"], w["P1"]
        if n[1] > 0.5:
            x_range = f" (x: {P0[0]:.3f} to {P1[0]:.3f})" if abs(P1[0] - P0[0]) <= inf_threshold else ""
            desc = f"Floor at y={d / n[1]:.4f} m, outward normal along +y. Beam must stay above{x_range}."
        elif n[1] < -0.5:
            x_range = f" (x: {P0[0]:.3f} to {P1[0]:.3f})" if abs(P1[0] - P0[0]) <= inf_threshold else ""
            desc = f"Ceiling at y={d / n[1]:.4f} m, outward normal along -y. Beam must stay below{x_range}."
        elif n[0] > 0.5:
            y_range = f" (y: {P0[1]:.3f} to {P1[1]:.3f})" if abs(P1[1] - P0[1]) <= inf_threshold else ""
            desc = f"Left wall at x={d / n[0]:.4f} m, outward normal along +x. Beam must stay to the right{y_range}."
        else:
            y_range = f" (y: {P0[1]:.3f} to {P1[1]:.3f})" if abs(P1[1] - P0[1]) <= inf_threshold else ""
            desc = f"Right wall at x={d / n[0]:.4f} m, outward normal along -x. Beam must stay to the left{y_range}."
        lines.append(f"  Wall {i+1}: {desc}")
    return "\n".join(lines)

def describe_bc_constraints(ps, q: list) -> str:
    """Describes the s=1 end of the beam — the end the LLM does not control."""
    nv = ps.nv
    fx  = float(q[2 * (nv - 1)])
    fy  = float(q[2 * (nv - 1) + 1])
    dx  = fx - float(q[2 * (nv - 2)])
    dy  = fy - float(q[2 * (nv - 2) + 1])
    theta = float(np.arctan2(dy, dx))
    if ps.controlled_nodes == [0, 1]:
        return (
            f"Boundary conditions (s=1 end, current position: ({fx:.4f}, {fy:.4f}) m, theta={theta:.4f} rad):\n"
            f"  This end is free. It deforms under gravity and wall contact."
        )
    else:
        return (
            f"Boundary conditions:\n"
            f"   s=1 end is permanently clamped at ({fx:.4f}, {fy:.4f}) m, "
            f"theta={theta:.4f} rad. It DOES NOT MOVE. You only control the other end at s=0."
        )


def describe_controller_feedback(reaction_force: list, ps, q: list) -> str:
    """Reaction force and bending moment at the clamped end vs beam weight."""
    if reaction_force is None or len(reaction_force) < 4:
        return ""

    F0 = np.array(reaction_force[0:2])
    F1 = np.array(reaction_force[2:4])
    F_total = F0 + F1
    Fx, Fy  = float(F_total[0]), float(F_total[1])
    magnitude = float(np.sqrt(Fx**2 + Fy**2))

    # Bending moment about node 0 due to force on node 1: Mz = (r1 - r0) x F1
    Mz = (q[2] - q[0]) * F1[1] - (q[3] - q[1]) * F1[0]

    beam_weight = float(ps.m.sum() * abs(config.GRAVITY[1]))
    Rx, Ry  = -Fx, -Fy  # force beam exerts back on arm (Newton's 3rd law)
    Mz_arm  = -Mz        # moment felt by arm (Newton's 3rd law)
    return "\n".join([
        "Controller feedback at clamped end (s=0):",
        f"  Force applied by robotic arm on DLO: [{Fx:+.3f}, {Fy:+.3f}] N   magnitude = {magnitude:.3f} N",
        f"  Force felt by arm (DLO on arm): [{Rx:+.3f}, {Ry:+.3f}] N",
        f"  Bending moment applied by arm on DLO: {Mz:+.4f} N*m",
        f"  Bending moment felt by arm (DLO on arm): {Mz_arm:+.4f} N*m",
        f"  Beam weight: {beam_weight:.3f} N (along negative y-axis)",
        "  If the force applied by arm on DLO is close to beam weight, there is no significant external loads (wall contact, clamped tip, etc.).",
    ])


def describe_contact_regions(cx: list | None, cy: list | None, nv: int, walls: list) -> str:
    """Enumerate nodes in contact with walls by arc-length. Walls numbered as in describe_wall_constraints."""
    if not walls or (cx is None and cy is None):
        return ""
    lines = ["Wall contact:"]
    for i in range(nv):
        wx = cx[i] if cx is not None else -1
        wy = cy[i] if cy is not None else -1
        wall_idx = wx if wx >= 0 else wy
        if wall_idx >= 0:
            lines.append(f"  Contact with Wall#{wall_idx+1} at arc-length s={i / (nv - 1):.2f}")
    return "\n".join(lines) if len(lines) > 1 else ""


def describe_axial_stretch(axial_strain, ps) -> str:
    """Warn the LLM when any edge strain exceeds config.AXIAL_STRAIN_THRESHOLD. Returns "" otherwise."""
    if axial_strain is None:
        return ""
    max_strain = float(np.max(np.abs(axial_strain)))
    if max_strain < config.AXIAL_STRAIN_THRESHOLD:
        return ""
    nv = ps.nv
    lines = [
        f"Warning — axial stretching detected (max strain: {max_strain*100:.1f}%):",
        "  The DLO is being stretched or compressed significantly along its length.",
        "  This happens when the controlled end is moved to a position that cannot be reached",
        "  without deforming the beam axially. Large axial strain means the beam may behave",
        "  unpredictably.",
        # "  Per-edge strain (positive = stretched, negative = compressed):",
    ]
    # for i, strain in enumerate(axial_strain):
    #         s_mid = (i + 0.5) / (nv - 1)
    #         lines.append(f"    Edge at s={s_mid:.2f}: {strain*100:+.1f}%")
    return "\n".join(lines)


def describe_error_rejection(error_msg: str) -> str:
    """
    Format a BC rejection error to alert the LLM.
    Only shown if last BC was rejected by safety checks.
    """
    if not error_msg:
        return ""
    return f"""\
PREVIOUS MOVE REJECTED — DO NOT REPEAT

Your last boundary condition was rejected because:
  {error_msg}

Please adjust your values to satisfy the constraint and try again."""


def describe_dt_failure_rejection() -> str:
    """Static rejection message shown when a BC caused the physics solver to fail even at the smallest allowed timestep."""
    return """\
PREVIOUS BC REJECTED — DO NOT REPEAT

The previous boundary condition led to high strains in the DLO, and our numerical \
tool estimates non-realistic configurations as a result. Avoid this type of BC."""

# ---------------------------------------------------------------------------
# Standalone test — python -m src.core.observation
# ---------------------------------------------------------------------------

_TEST_CASES = [
    {
        "label": "Straight beam (initial position)",
        "bc":    [0.0, 0.0, 0.0],
        "nodes": [(0.0, 0.0), (0.0, -0.025), (0.0, -0.05), (0.0, -0.075), (0.0, -0.10)],
        "observable_nodes": [0, 1, 2, 3, 4],
        "nv": 5,
        "walls": [{"n": np.array([0., 1.]), "d": -0.12,
                   "P0": np.array([-10., -0.12]), "P1": np.array([10., -0.12])}],
        "controlled_nodes": [0, 1],
    },
    {
        "label": "C-shape in progress",
        "bc":    [0.0, -0.03, 0.4],
        "nodes": [(0.0, -0.03), (0.015, -0.05), (0.025, -0.075),
                  (0.02, -0.10), (0.005, -0.115)],
        "observable_nodes": [0, 1, 2, 3, 4],
        "nv": 5,
        "walls": [{"n": np.array([0., 1.]), "d": -0.12,
                   "P0": np.array([-10., -0.12]), "P1": np.array([10., -0.12])}],
        "controlled_nodes": [0, 1],
    },
    {
        "label": "Snake-L: beam entering vertical channel",
        "bc":    [0.0, -0.04, 0.0],
        "nodes": [(0.0, -0.04), (0.0, -0.065), (0.0, -0.09),
                  (0.005, -0.105), (0.02, -0.11)],
        "observable_nodes": [0, 1, 2, 3, 4],
        "nv": 5,
        "walls": [
            {"n": np.array([ 1.,  0.]), "d": -0.01, "P0": np.array([-0.01,  0.02]), "P1": np.array([-0.01, -0.12])},
            {"n": np.array([ 0.,  1.]), "d": -0.12, "P0": np.array([-0.01, -0.12]), "P1": np.array([ 0.12, -0.12])},
            {"n": np.array([-1.,  0.]), "d": -0.12, "P0": np.array([ 0.12, -0.12]), "P1": np.array([ 0.12, -0.10])},
            {"n": np.array([ 0., -1.]), "d":  0.10, "P0": np.array([ 0.12, -0.10]), "P1": np.array([ 0.01, -0.10])},
            {"n": np.array([-1.,  0.]), "d": -0.01, "P0": np.array([ 0.01, -0.10]), "P1": np.array([ 0.01,  0.02])},
        ],
        "controlled_nodes": [0, 1],
    },
]


class _MockPS:
    """Minimal physical system mock covering all describe_* functions."""
    rod_length      = 0.1
    friction        = False
    r0              = 1e-2
    rho             = 1000.0
    EI              = 1e6 * np.pi * (1e-2)**4 / 4   # Y * pi * r0^4 / 4

    def __init__(self, tc):
        self.walls           = tc["walls"]
        self.nv              = tc["nv"]
        self.observable_nodes = tc["observable_nodes"]
        self.controlled_nodes = tc["controlled_nodes"]
        self.m               = np.full(tc["nv"], self.rho * np.pi * self.r0**2 * self.rod_length / tc["nv"])


def main():
    sep = "=" * 60
    for tc in _TEST_CASES:
        print(f"\n{sep}")
        print(f"TEST: {tc['label']}")
        print(f"{sep}")

        ps     = _MockPS(tc)
        q_flat = [v for xy in tc["nodes"] for v in xy]
        kappa  = np.zeros(tc["nv"])  # straight beam: zero curvature everywhere

        print(describe_beam_physics(ps))
        print()
        print(describe_wall_constraints(ps) or "(no walls)")
        print()
        print(describe_bc_constraints(ps, q_flat))
        print()
        print(describe_current_shape(q_flat, ps, kappa=kappa, bc=tc["bc"]))
        print()

        # one-step dummy history (uses previous bc/shape as prior step)
        prev_bc    = [[b * 0.5 for b in tc["bc"]]]
        prev_shape = [[(x * 0.5, y * 0.5) for x, y in tc["nodes"]]]
        prev_kappa = [list(kappa)]
        prev_error = [42.0]
        print(describe_history(prev_bc, prev_shape, ps, prev_error, kappa_history=prev_kappa))

        print(describe_error_rejection("Node 0 too close to wall (clearance 0.001 m < 0.004 m)."))

    # describe_controller_feedback and describe_contact_regions — tested once with dummy data
    print(f"\n{sep}")
    print("TEST: controller feedback + contact regions")
    print(f"{sep}")
    ps0    = _MockPS(_TEST_CASES[0])
    q0     = [v for xy in _TEST_CASES[0]["nodes"] for v in xy]
    rf     = [0.1, -0.5, 0.05, -0.3] + [0.0] * (2 * ps0.nv - 4)
    print(describe_controller_feedback(rf, ps0, q0))
    print()
    cx = [-1, -1, 0, 0, -1]
    cy = [-1, -1, -1, -1, 0]
    print(describe_contact_regions(cx, cy, ps0.nv, ps0.walls))


if __name__ == "__main__":
    main()
