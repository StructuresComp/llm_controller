# ---------------------------------------------------------------------------
# src/agents/orchestrator_agent.py
# ---------------------------------------------------------------------------
# Orchestrator LangGraph sub-graph: planner_node → decision_node.
#
# planner_node  — writes or revises state["plan"] based on ORCHESTRATOR_MODE
#                 and iteration counter; no-op in myopic mode.
#                 "plan":   runs only at iteration 0.
#                 "replan": runs every REPLAN_FREQ iterations (overwrites plan).
# decision_node — always runs; calls LLM with _BC_SCHEMA for BC decision (retrying up
#                 to TOOL_CALL_ATTEMPTS times, then an extraction-fallback pass via a
#                 separate reliable model if every attempt fails to parse);
#                 convergence is determined deterministically from eval_result.
#
# INPUT/OUTPUT MANIFEST:
# ---------------------------------------------------------------------------
# planner_node:
#   Input:  state["plan"], state["iteration"], state["target_shape"], state["physical_system"]
#   Output: state["plan"]
#
# decision_node:
#   Input:  state["shape_history"], state["bc_history"],
#           state["eval_result"], state["physical_system"], state["target_shape"],
#           state["plan"], state["iteration_log"]
#   Output: state["decision"] {x0, y0, theta}
#           state["converged"] bool
#           state["orchestrator_prompt"] str (for JSONL traceability)
#           state["error"] None on success; set (and "decision" omitted) if the LLM
#             tool call still fails to parse after config.TOOL_CALL_ATTEMPTS retries
#             AND the extraction fallback (config.EXTRACTION_PROVIDER/EXTRACTION_MODEL)
#             also fails — workflow.py routes that straight to END.
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
from src.core.prompt_builder import build_planner_prompt, build_decision_prompt
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
        "Set the next boundary condition for the DLO. "
        "Specify the target position (x0, y0) and angle (theta) of the controlled end at s=0."
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
                    "Target angle at node 0 (radians, counter-clockwise from +X axis, standard math convention). "
                    "0 = pointing right, pi/2 = pointing up, -pi/2 = pointing straight down."
                ),
            },
            "reason": {
                "type": "string",
                "description": "One sentence explaining why this boundary condition was chosen.",
            },
        },
        "required": ["x0", "y0", "theta", "reason"],
    },
}

# Extended BC schema — experimental, not wired up yet.
# Adds two analysis fields BEFORE the BC values to encourage sequential reasoning:
#   1. error_analysis   — LLM diagnoses the dominant mismatch (region, sign, magnitude)
#   2. correction_intent — LLM states what BC change will address it
# Field order matters: autoregressive generation means these are produced before x0/y0/theta,
# softly conditioning the BC values on the prior diagnosis.
# Replaces 'reason' (which is post-hoc) with structured pre-action reasoning.
# To activate: pass _BC_SCHEMA_EXTENDED to call_llm_with_tool in decision_node.
_BC_SCHEMA_EXTENDED = {
    "name": "set_boundary_condition",
    "description": (
        "Set the next boundary condition for the DLO. "
        "First diagnose the current error, then state your correction intent, "
        "then specify the target position and angle."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "error_analysis": {
                "type": "string",
                "description": (
                    "Describe the dominant mismatch between current and target shape."
                ),
            },
            "reason": {
                "type": "string",
                "description": (
                    "State which BC variable (x0, y0, or theta) you will change, "
                    "in which direction, and why it addresses the error."
                ),
            },
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
                    "Target angle at node 0 (radians, counter-clockwise from +X axis, standard math convention). "
                    "0 = pointing right, pi/2 = pointing up, -pi/2 = pointing straight down."
                ),
            },
        },
        "required": ["error_analysis", "reason", "x0", "y0", "theta"],
    },
}

# ---------------------------------------------------------------------------
# 3. PROMPTS
# ---------------------------------------------------------------------------

_PLAN_SYSTEM = (
    "You are an expert planner for controlling an elastic deformable linear object (DLO), "
    "modeled as a beam.\n"
    "A robot controls the position and angle of the clamped end at arc-length s=0.\n"
    "The other end of the DLO has arc-length s=1 (normalized by length).\n"
    "\n"
    "The goal is to deform the DLO into a target shape or satisfy a target task specified "
    "in the user prompt.\n"
    "\n"
    "Your role is to create a high-level control plan, not to issue the next boundary "
    "condition directly.\n"
    "\n"
    "Reason in terms of mechanics: bending energy, geometric feasibility, gravity, contact, "
    "friction if provided, and how changing the controlled-end position and angle will propagate "
    "deformation along the DLO.\n"
    "Be mindful of how theta is defined — refer to the coordinate system in the user prompt.\n"
    "\n"
    "Produce a concise strategy describing how to deform the DLO toward the target.\n"
    "Leverage boundary conditions (clamped end) or walls (if any), gravity, and the elastic "
    "deformation of the DLO.\n"
    "\n"
    "Before planning, mentally reconstruct the current shape from the node positions "
    "and reason about what sequence of moves will guide it toward the target.\n"
    "\n"
    "Moving s=0 has the most direct effect on nodes near s=0; the influence attenuates "
    "toward s=1, which responds only through elastic propagation — to shape the far end, "
    "first shape the near end and let deformation propagate incrementally.\n"
    "theta sets the departure angle at s=0 and most directly controls curvature near s=0. "
    "(x0, y0) repositions the controlled end — its effect on the rest of the beam depends "
    "on other boundary conditions and constraints such as walls.\n"
    "The order of bending matters: if the beam bends in the wrong direction early, "
    "plan to reverse it before committing to the final configuration.\n"
    "\n"
    "FORMAT RULES — strictly enforced:\n"
    "- Describe a linear sequence of directional intents. Do not number them, do not label them, "
    " do not write fallback chains (e.g., 'Step 2K-ALT: if that fails, try Step 2L-ALT').\n"
    "- No conditional branching. Do not write 'if X then Y else Z'. "
    "The decision agent observes the current state fresh each iteration and executes your plan.\n"
    "- No iteration counts. Do not specify how many moves a phase should take.\n"
    "\n"
)

_DECISION_SYSTEM = (
    "You are controlling an elastic deformable linear object (DLO), modeled as a beam.\n"
    "The goal is to deform the DLO into a target shape or satisfy a target task specified "
    "in the user prompt.\n"
    "A robot holds the clamped end of the DLO at arc-length s=0.\n"
    "The other end of the DLO has arc-length (normalized by length) s=1.\n"
    "\n"
    "You must command the next boundary condition at s=0 by calling the set_boundary_condition tool.\n"
    "You will apply control actions repeatedly until the target shape is achieved — "
    "each decision moves the DLO one step closer.\n"
    "\n"
    "The controllable boundary variables are the clamped-end position (x0, y0) and angle theta at s=0.\n"
    "Be mindful of how theta is defined — refer to the coordinate system in the user prompt.\n"
    "Deformation is restricted to the X-Y plane. Gravity acts in the negative Y direction.\n"
    "\n"
    "Your role in this mode is to decide the next boundary condition, not to write a long-term plan.\n"
    "Prefer small, deliberate moves — large changes can cause unpredictable deformation, "
    "and you may need many iterations to correct course.\n"
    "Before deciding, mentally reconstruct the current shape from the node positions, "
    "then reason about how your move will change it.\n"
    "\n"
    "Moving s=0 has the most direct effect on nodes near s=0; the influence attenuates "
    "toward s=1, which responds only through elastic propagation — to shape the far end, "
    "first shape the near end and let deformation propagate incrementally.\n"
    "theta sets the departure angle at s=0 and most directly controls curvature near s=0. "
    "(x0, y0) repositions the controlled end — its effect on the rest of the beam depends "
    "on other boundary conditions and constraints such as walls.\n"
    "The order of bending matters: if the beam bends in the wrong direction early, "
    "you may need to reverse course before the target shape is reachable.\n"
    "\n"
    "You must call the set_boundary_condition tool exactly once."
)

_EXTRACTION_SYSTEM = (
    "You extract structured data faithfully. You never invent, adjust, or infer "
    "values that were not explicitly stated in the text you are given."
)

_EXTRACTION_PROMPT_TEMPLATE = (
    "A controller was asked to decide the next boundary condition for a deformable "
    "linear object and responded with the text below. Extract exactly the x0, y0, "
    "theta, and reason it stated by calling set_boundary_condition — do not invent, "
    "adjust, or infer values it did not explicitly give. If it gave no usable values "
    "at all, do your best to infer the closest reasonable call from context.\n\n"
    "Controller response:\n{free_text}"
)


# ---------------------------------------------------------------------------
# 4. HELPERS
# ---------------------------------------------------------------------------

# Leaving it as is for now for future enhancements. But we do not use the function below.
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


def _try_extraction_fallback(prompt: str) -> dict | None:
    """
    Last resort after config.TOOL_CALL_ATTEMPTS is exhausted: get a free-text response
    from the configured model (no schema constraint — an easier task than forcing valid
    JSON), then use config.EXTRACTION_PROVIDER/EXTRACTION_MODEL (a model known to be
    reliable at forced tool-calling) purely to extract the decision from that text.
    The extraction model never makes its own decision, only transcribes what the
    configured model already said. Returns None on any failure (treated as terminal
    by the caller — no separate retry budget for extraction).
    """
    try:
        free_text = call_llm(prompt=prompt, system=_DECISION_SYSTEM)
        decision = call_llm_with_tool(
            prompt      = _EXTRACTION_PROMPT_TEMPLATE.format(free_text=free_text),
            tool_schema = _BC_SCHEMA,
            tool_name   = "set_boundary_condition",
            system      = _EXTRACTION_SYSTEM,
            provider    = config.EXTRACTION_PROVIDER,
            model       = config.EXTRACTION_MODEL,
        )
        for key in ("x0", "y0", "theta"):
            decision[key] = float(decision[key])
        return decision
    except Exception as e:
        if config.DEBUG:
            print(f"  [decision] Extraction fallback failed: {e}")
        return None


# ---------------------------------------------------------------------------
# 5. NODES
# ---------------------------------------------------------------------------

def planner_node(state: ControllerState) -> dict:
    """Generate or revise plan based on mode and iteration counter; no-op in myopic mode."""

    if config.ORCHESTRATOR_MODE == "myopic":
        return {}
    if config.ORCHESTRATOR_MODE == "plan" and state["iteration"] > 0:
        return {}
    if config.ORCHESTRATOR_MODE == "replan" and state["iteration"] % config.REPLAN_FREQ != 0:
        return {}

    prompt = build_planner_prompt(state)

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
    """
    Issue the next BC decision; convergence is read from eval_result.

    Retries the LLM tool call up to config.TOOL_CALL_ATTEMPTS additional times if the
    response fails to parse (malformed JSON/tool call — see llm_wrapper's Ollama JSON-
    prompt fallback). Each attempt counts toward attempt_count.

    If every attempt fails, falls back to _try_extraction_fallback: a free-text
    response from the configured model, then a separate extraction-only pass via
    config.EXTRACTION_PROVIDER/EXTRACTION_MODEL. If that also fails, returns
    {"error": ...} instead of raising; workflow.py routes this straight to END.
    """
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

    attempt_count = state.get("attempt_count", 0)
    max_attempts  = max(config.TOOL_CALL_ATTEMPTS, 0) + 1
    decision      = None
    last_error    = None

    for attempt in range(max_attempts):
        attempt_count += 1
        try:
            # BC decision — schema-enforced, never free JSON
            decision = call_llm_with_tool(
                prompt      = prompt,
                tool_schema = _BC_SCHEMA,
                tool_name   = "set_boundary_condition",
                system      = _DECISION_SYSTEM,
            )
            # Coerce numeric fields: some providers return numbers as strings despite the schema.
            for key in ("x0", "y0", "theta"):
                decision[key] = float(decision[key])
            break
        except Exception as e:
            last_error = e
            decision   = None
            if config.DEBUG:
                if attempt + 1 < max_attempts:
                    print(f"  [decision] Tool call failed (attempt {attempt + 1}/{max_attempts}): {e} -- retrying.")
                else:
                    print(f"  [decision] Tool call failed (attempt {attempt + 1}/{max_attempts}): {e} -- giving up.")

    if decision is None:
        if config.DEBUG:
            print(f"  [decision] All {max_attempts} tool-call attempt(s) failed -- trying extraction fallback "
                  f"({config.EXTRACTION_PROVIDER}/{config.EXTRACTION_MODEL}).")
        attempt_count += 1
        decision = _try_extraction_fallback(prompt)
        if decision is None:
            if config.DEBUG:
                print("  [decision] Extraction fallback also failed -- giving up.")
            return {
                "error": f"Orchestrator: LLM tool call failed after {max_attempts} attempt(s) "
                         f"plus extraction fallback: {last_error}",
                "attempt_count": attempt_count,
            }
        if config.DEBUG:
            print("  [decision] Extraction fallback succeeded.")

    # Print decision in real-time
    iteration_num = len(state["iteration_log"]) + 1
    print(f"\n[Iteration {iteration_num}] Orchestrator decision:")
    print(f"  x0 = {decision['x0']:+.6f} m  |  y0 = {decision['y0']:+.6f} m  |  theta = {decision['theta']:+.6f} rad")
    print(f"  Reason: {decision.get('reason', '(none)')}")

    converged = state.get("eval_result", {}).get("solved", False)
    if converged:
        print(f"  checkmark CONVERGED (eval_result.solved = True)")
    else:
        print(f"  -> Not converged")

    # # Stagnation can be detected via _detect_stagnation(state["bc_history"]) and returned
    # # as {"plan": ""} to trigger a forced replan. Commented out to keep orchestrator mode
    # # fixed for the entire run — cross-LLM comparison requires a stable mode.

    return {
        "decision":           decision,
        "converged":          converged,
        "convergence_reason": decision.get("reason", ""),
        "orchestrator_prompt": prompt,  # stored for JSONL traceability
        "attempt_count":      attempt_count,
        "error":              None,  # clear any stale rejection from a previous cycle
    }


# ---------------------------------------------------------------------------
# 6. GRAPH BUILDER
# ---------------------------------------------------------------------------

def build_orchestrator_graph():
    """Compile the orchestrator sub-graph: planner_node -> decision_node."""
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
    state = initial_controller_state(ps, target_shape="C")

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
