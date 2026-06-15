# ---------------------------------------------------------------------------
# src/agents/orchestrator_agent.py
# ---------------------------------------------------------------------------
# Orchestrator LangGraph sub-graph: planner_node → decision_node.
#
# planner_node  — writes a strategy to state["plan"] on the first call in
#                 "plan" mode; no-op otherwise.
# decision_node — always runs; appends state["plan"] to the prompt (empty
#                 string in myopic mode = no effect); calls LLM twice:
#                 (1) BC decision with _BC_SCHEMA
#                 (2) convergence assessment with _CONVERGED_SCHEMA
#                 Both results merged into a single atomic output.
#
# INPUT/OUTPUT MANIFEST:
# ---------------------------------------------------------------------------
# planner_node:
#   Input:  state["plan"], state["target_shape"], state["physical_system"]
#   Output: state["plan"]
#
# decision_node:
#   Input:  state["shape_history"], state["bc_history"], state["plan"],
#           state["eval_result"], state["physical_system"], state["target_shape"],
#           state["iteration_log"]
#   Output: state["decision"] {x0, y0, theta}
#           state["converged"] bool
#           state["orchestrator_prompt"] str (for JSONL traceability)
# ---------------------------------------------------------------------------
#
# Section order:
#   1. Imports  2. Schemas  3. Prompts  4. Helpers  5. Nodes  6. Graph  7. Main  8. Entry
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 1. IMPORTS
# ---------------------------------------------------------------------------
from langgraph.graph import END, StateGraph

from src.core import config
from src.core.observation import describe_walls, describe_nodes, describe_history
from src.llm.llm import call_llm, call_llm_with_tool
from src.models.state_llm_control import ControllerState


# ---------------------------------------------------------------------------
# 2. SCHEMAS
# ---------------------------------------------------------------------------

# BC decision schema — orchestrator is ALWAYS forced through this.
# Never ask the LLM to write JSON freely; always use call_llm_with_tool.
_BC_SCHEMA = {
    "name": "set_boundary_condition",
    "description": (
        "Set the next boundary condition for the beam simulation. "
        "Specify the target position and angle of the controlled end (node 0)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "x0": {
                "type": "number",
                "description": "Target x-position of node 0 (meters). Positive = right.",
            },
            "y0": {
                "type": "number",
                "description": "Target y-position of node 0 (meters). Negative = below origin.",
            },
            "theta": {
                "type": "number",
                "description": (
                    "Target angle at node 0 (radians, counter-clockwise from -Y axis). "
                    "0 = straight down, pi/2 = pointing right, -pi/2 = pointing left."
                ),
            },
        },
        "required": ["x0", "y0", "theta"],
    },
}

# Convergence assessment schema
_CONVERGED_SCHEMA = {
    "name": "assess_convergence",
    "description": "Assess whether the DLO has reached the target shape.",
    "input_schema": {
        "type": "object",
        "properties": {
            "converged": {
                "type": "boolean",
                "description": "True if the current shape matches the target well enough.",
            },
            "reason": {
                "type": "string",
                "description": "One sentence explaining the assessment.",
            },
        },
        "required": ["converged", "reason"],
    },
}


# ---------------------------------------------------------------------------
# 3. PROMPTS
# ---------------------------------------------------------------------------

_PLAN_SYSTEM = (
    "You are an expert in controlling elastic beams. Your task is to press a deformable "
    "beam against a wall and deform it into a target shape by manipulating the position "
    "and angle at the controlled end. Think step-by-step about beam mechanics: bending "
    "energy, gravity, contact with the wall, and which moves will curl the beam toward "
    "the desired shape."
)

_PLAN_PROMPT = """\
ROBOTIC ARM CONTROL SETUP:

PHYSICAL SYSTEM:
- Elastic beam (length L = {rod_length:.4f} m, stiffness EI = {EI:.6e})
- One end: CLAMPED and held by the arm (node 0). YOU CONTROL THIS END.
- Other end: FREE. Hangs under gravity. Can contact the wall.
- Rigid wall(s): {wall_description}

YOUR CONTROL INPUTS (arm degrees of freedom):
- Position (x0, y0): where you hold node 0 (the clamped end)
  * x0: horizontal position. x0 = 0 is centered. Positive x0 pushes right.
  * y0: vertical position. y0 = 0 is origin. Negative y0 pushes down toward wall.
- Angle theta: orientation of the clamped end (in radians, counter-clockwise from -Y axis)
  * theta = 0:    beam points straight down          [tangent: 0, -1]
  * theta = π/2:  beam points right                  [tangent: 1,  0]
  * theta = -π/2: beam points left                   [tangent: -1, 0]

TARGET: Deform the beam into a {target_shape} shape by pressing its free end against the wall and bending it.

STRATEGY (3-6 steps):
Write a step-by-step plan:
1. First move: How will you position the arm to press the beam toward the wall?
2. Rotation: What angle(s) will you apply to bend it in the right direction?
3-6. Refinement: What sequence of arm moves will shape it into {target_shape}?

Note: High EI means the beam resists bending — you need strong moves. Start gradually.
"""

_DECISION_SYSTEM = (
    "You are controlling an elastic beam pressed against a wall. Your job is to issue "
    "the next boundary condition move by calling the 'set_boundary_condition' tool.\n\n"
    "TOOL SCHEMA: set_boundary_condition\n"
    "  x0 (number): target x-position of node 0 (meters)\n"
    "  y0 (number): target y-position of node 0 (meters)\n"
    "  theta (number): target angle in radians (0 = straight down, π/2 = right, -π/2 = left)\n\n"
    "Think about beam mechanics: which moves will bend it toward the target shape? "
    "When does it contact the wall? Should you backtrack if progress stalled?"
)

_DECISION_PROMPT = """\
ROBOTIC ARM CONTROL TASK:

YOU CONTROL: Clamped end (node 0) position (x0, y0) and angle (theta, in radians).
GOAL: Bend the free end of the beam against the wall ({wall_description}) to achieve a {target_shape} shape.
BEAM PROPERTIES: Length L = {rod_length:.4f} m, Stiffness EI = {EI:.6e}.

ANGLE CONVENTION (θ counter-clockwise from -Y axis):
  θ =  0:    straight down  [0, -1]
  θ =  π/2:  pointing right [1,  0]
  θ = -π/2:  pointing left [-1,  0]

{plan_section}\
{error_section}\
RECENT ARM MOVES ({k} most recent):
{history_section}
CURRENT BEAM STATE:
{current_section}
{eval_section}\
NEXT MOVE:
Decide the arm's next position (x0, y0) and angle (theta) to progress toward {target_shape}.

Ask yourself:
1. Is the free end touching the wall? If not, move y0 more negative to push it against the wall.
2. Is the beam bending in the right direction? If not, rotate theta to correct the curve.
3. Should you backtrack? If progress stalled, reverse the last move.

Strategy: Move gradually. The beam resists bending (high EI). Incremental changes accumulate.
"""


# ---------------------------------------------------------------------------
# 4. HELPERS
# ---------------------------------------------------------------------------

def _format_plan_section(plan: str) -> str:
    """Prepend the plan to the decision prompt if one exists."""
    if not plan:
        return ""
    return f"Strategy:\n{plan}\n\n"


def _format_eval_section(eval_result: dict) -> str:
    """Format the deterministic evaluator result for the decision prompt."""
    if not eval_result:
        return ""
    pct = eval_result.get("error_percent", 100.0)
    diag = eval_result.get("diagnostics", "")
    return (
        f"GEOMETRIC SHAPE ERROR (deterministic ground truth):\n"
        f"  {pct:.1f}% error  (0% = perfect match, 100% = no match)\n"
        f"  {diag}\n\n"
    )


def _format_error_section(error: str) -> str:
    """Surface a safety rejection so the LLM knows why its last BC was refused."""
    if not error:
        return ""
    return (
        f"*** PREVIOUS BC REJECTED — DO NOT REPEAT THIS MOVE ***\n"
        f"  {error}\n"
        f"  Adjust your values to satisfy the safe limit stated above and try again.\n\n"
    )


def _detect_stagnation(bc_history: list, tol: float = 1e-4) -> bool:
    """Return True if the last few BCs are nearly identical (no progress)."""
    if len(bc_history) < 3:
        return False
    recent = bc_history[-3:]
    for i in range(1, len(recent)):
        diff = sum(abs(a - b) for a, b in zip(recent[i], recent[i-1]))
        if diff > tol:
            return False
    return True


# ---------------------------------------------------------------------------
# 5. NODES
# ---------------------------------------------------------------------------

def planner_node(state: ControllerState) -> dict:
    """Generate a high-level plan on the first call in plan mode; no-op otherwise."""
    if config.ORCHESTRATOR_MODE != "plan" or state["plan"]:
        return {}

    ps = state["physical_system"]
    prompt = _PLAN_PROMPT.format(
        wall_description = describe_walls(ps.walls),
        rod_length       = ps.rod_length,
        EI               = ps.EI,
        target_shape     = state["target_shape"] or "unspecified",
    )
    if config.DEBUG:
        print("\n" + "="*70)
        print("[PLANNER SYSTEM]")
        print("="*70)
        print(_PLAN_SYSTEM)
        print("\n[PLANNER PROMPT]")
        print("="*70)
        print(prompt)
        print("="*70 + "\n")
    plan = call_llm(prompt=prompt, system=_PLAN_SYSTEM)
    return {"plan": plan}


def decision_node(state: ControllerState) -> dict:
    """Issue the next BC decision and assess convergence."""
    ps  = state["physical_system"]
    lvl = config.OBSERVATION_LEVEL

    if state["shape_history"]:
        current_section = describe_nodes(
            state["shape_history"][-1], state["bc_current"], level=lvl,
        ) + "\n"
    else:
        current_section = "(initial straight-down configuration)\n"

    prompt = _DECISION_PROMPT.format(
        target_shape     = state["target_shape"] or "unspecified",
        wall_description = describe_walls(ps.walls),
        rod_length       = ps.rod_length,
        EI               = ps.EI,
        k                = config.K_HISTORY,
        plan_section     = _format_plan_section(state["plan"]),
        error_section    = _format_error_section(state.get("error") or ""),
        history_section  = describe_history(
                               state["bc_history"], state["shape_history"], level=lvl,
                           ) + "\n",
        current_section  = current_section,
        eval_section     = _format_eval_section(state.get("eval_result", {})),
    )

    if config.DEBUG:
        print("\n" + "="*70)
        print("[DECISION SYSTEM]")
        print("="*70)
        print(_DECISION_SYSTEM)
        print("\n[DECISION PROMPT]")
        print("="*70)
        print(prompt)
        print("="*70 + "\n")

    # BC decision — schema-enforced, never free JSON
    decision = call_llm_with_tool(
        prompt      = prompt,
        tool_schema = _BC_SCHEMA,
        tool_name   = "set_boundary_condition",
        system      = _DECISION_SYSTEM,
    )

    # Print decision in real-time
    iteration_num = len(state["iteration_log"]) + 1
    print(f"\n[Iteration {iteration_num}] Orchestrator decision:")
    print(f"  x0 = {decision['x0']:+.6f} m  |  y0 = {decision['y0']:+.6f} m  |  θ = {decision['theta']:+.6f} rad")

    # Convergence assessment
    eval_result = state.get("eval_result", {})
    convergence_prompt = (
        f"TARGET SHAPE: {state['target_shape']}\n"
        f"WALL: {describe_walls(ps.walls)}\n\n"
        f"CURRENT BEAM STATE (observable nodes):\n"
        f"{current_section}"
        + (_format_eval_section(eval_result) if eval_result else "")
        + f"\nHas the beam been deformed into a {state['target_shape']} shape? "
        f"Judge by: curvature direction, symmetry, contact with the wall, overall shape match. "
        f"Be strict: the shape must clearly match the target."
    )
    if config.DEBUG:
        print("\n" + "="*70)
        print("[CONVERGENCE SYSTEM]")
        print("="*70)
        print("You are a structural mechanics expert. Judge whether the current beam deformation matches the target shape.")
        print("\n[CONVERGENCE PROMPT]")
        print("="*70)
        print(convergence_prompt)
        print("="*70 + "\n")
    convergence = call_llm_with_tool(
        prompt = convergence_prompt,
        tool_schema = _CONVERGED_SCHEMA,
        tool_name   = "assess_convergence",
        system      = "You are a structural mechanics expert. Judge whether the current beam deformation matches the target shape.",
    )

    # Print convergence assessment in real-time
    if convergence["converged"]:
        print(f"  ✓ CONVERGED: {convergence['reason']}")
    else:
        print(f"  → Not converged: {convergence['reason']}")

    updates: dict = {
        "decision":  decision,
        "converged": convergence["converged"],
        "convergence_reason": convergence["reason"],  # recorded per iteration for stability analysis
        "orchestrator_prompt": prompt,  # Store for JSONL traceability
    }

    # Replan if stagnating and ALLOW_REPLAN is set
    if config.ALLOW_REPLAN and _detect_stagnation(state["bc_history"]):
        updates["plan"] = ""   # clears plan → planner_node will regenerate next call

    return updates


# ---------------------------------------------------------------------------
# 6. GRAPH BUILDER
# ---------------------------------------------------------------------------

def build_orchestrator_graph():
    """Compile the orchestrator sub-graph: planner_node → decision_node."""
    graph = StateGraph(ControllerState)

    graph.add_node("planner",  planner_node)
    graph.add_node("decision", decision_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "decision")
    graph.add_edge("decision", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------------------------

def main():
    """Standalone test: run one orchestrator cycle from an empty state."""
    from src.models.physical_system import make_physical_system
    from src.models.state_llm_control import initial_controller_state

    ps    = make_physical_system(config)
    state = initial_controller_state(ps, target_shape="C", bc_target=[0.0, -0.05, 0.5])

    graph   = build_orchestrator_graph()
    updates = graph.invoke(state)

    print("Plan:")
    print(updates.get("plan") or "(none — myopic mode)")
    print("\nDecision:", updates.get("decision"))
    print("Converged:", updates.get("converged"))


# ---------------------------------------------------------------------------
# 8. ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
