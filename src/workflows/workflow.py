# ---------------------------------------------------------------------------
# src/workflows/workflow.py
# ---------------------------------------------------------------------------
# LangGraph master workflow.
#
# Topology:
#   [START] → orchestrator → safety ──(violation)──→ orchestrator
#                  ↑               └──(safe)──→ programmer → evaluator
#                  └──────────────────────────────────(loop)──┘
#
# The orchestrator is itself a compiled sub-graph (planner → decision).
# The loop runs until the evaluator reports solved, the programmer sets
# state["error"] (physics failure), or MAX_ITERATIONS is reached. Safety
# violations route back to the orchestrator without counting as iterations,
# but the run ends after MAX_SAFETY_REJECTIONS consecutive rejections so the
# orchestrator↔safety loop cannot spin forever.
# ---------------------------------------------------------------------------

from langgraph.graph import END, StateGraph

from src.core import config
from src.models.state_llm_control import ControllerState
from src.agents.orchestrator_agent import build_orchestrator_graph
from src.agents.safety_agent import safety_node
from src.agents.programmer_agent import programmer_node
from src.agents.evaluator_agent import evaluator_node


def _after_safety(state: ControllerState) -> str:
    """Route after safety check.

    Safe BC → programmer. Violation → retry via the orchestrator, but end the
    run once MAX_SAFETY_REJECTIONS consecutive rejections have piled up — this
    bounds the otherwise-unbounded orchestrator↔safety loop (a weak model can
    repeatedly propose wall-violating BCs and never reach an iteration).
    """
    if not state.get("error"):
        return "programmer"
    if state.get("safety_rejections", 0) >= config.MAX_SAFETY_REJECTIONS:
        return "end"
    return "orchestrator"


def _should_continue(state: ControllerState) -> str:
    """Route after evaluator: loop or end.

    Termination is controlled by the deterministic evaluator (eval_result["solved"]),
    not the LLM's convergence claim.  state["converged"] is recorded as a research
    measurement — how accurately the LLM recognises its own success.

    Hard stop: if MAX_ITERATIONS is reached, end regardless of solve status.
    """
    if state.get("error"):
        return "end"
    if state.get("eval_result", {}).get("solved", False):
        return "end"
    if len(state.get("iteration_log", [])) >= config.MAX_ITERATIONS:
        return "end"
    return "orchestrator"


def build_controller_graph():
    graph = StateGraph(ControllerState)

    graph.add_node("orchestrator", build_orchestrator_graph())
    graph.add_node("safety",       safety_node)
    graph.add_node("programmer",   programmer_node)
    graph.add_node("evaluator",    evaluator_node)

    graph.set_entry_point("orchestrator")
    graph.add_edge("orchestrator", "safety")
    graph.add_conditional_edges(
        "safety",
        _after_safety,
        {"orchestrator": "orchestrator", "programmer": "programmer", "end": END},
    )
    graph.add_edge("programmer", "evaluator")
    graph.add_conditional_edges(
        "evaluator",
        _should_continue,
        {"orchestrator": "orchestrator", "end": END},
    )

    return graph.compile()
