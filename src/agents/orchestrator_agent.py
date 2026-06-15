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
from src.core.prompt_builder import build_decision_prompt, build_plan_prompt
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

_DECISION_SYSTEM = (
    "You are controlling an elastic beam. Your job is to issue the next boundary condition "
    "move by calling the 'set_boundary_condition' tool.\n\n"
    "TOOL SCHEMA: set_boundary_condition\n"
    "  x0 (number): target x-position of clamped end (meters)\n"
    "  y0 (number): target y-position of clamped end (meters)\n"
    "  theta (number): target angle in radians (0 = straight down, π/2 = right, -π/2 = left)\n\n"
    "Think about beam mechanics: which moves will bend it toward the target shape? "
    "When does it contact constraints? Should you backtrack if progress stalled?"
)


# ---------------------------------------------------------------------------
# 4. HELPERS
# ---------------------------------------------------------------------------

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

    prompt = build_plan_prompt(state)

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
    # Build prompt from modular sub-prompts (Tier 1 only in Phase 2; Tier 2-4 TBD)
    prompt = build_decision_prompt(state)

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

    # Convergence assessment (TODO: move to prompt_builder in Phase 3)
    ps = state["physical_system"]
    eval_result = state.get("eval_result", {})

    # Build minimal convergence prompt
    convergence_prompt = f"TARGET SHAPE: {state['target_shape']}\n"
    if eval_result:
        convergence_prompt += (
            f"ERROR: {eval_result.get('error_percent', 100.0):.1f}%\n"
            f"ASSESSMENT: {eval_result.get('diagnostics', '')}\n\n"
        )
    convergence_prompt += (
        f"Has the beam been deformed into a {state['target_shape']} shape? "
        f"Judge by the error percentage and diagnostics above. "
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
