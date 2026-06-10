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
# The loop runs until state["converged"] is True, state["error"] is set
# by the programmer (physics failure), or LangGraph's recursion_limit is hit.
# Safety violations route back to the orchestrator — they never reach the
# programmer or evaluator, so they do not count as iterations.
# ---------------------------------------------------------------------------

from langgraph.graph import END, StateGraph

from src.models.state_llm_control import ControllerState
from src.agents.orchestrator_agent import build_orchestrator_graph
from src.agents.safety_agent import safety_node
from src.agents.programmer_agent import programmer_node
from src.agents.evaluator_agent import evaluator_node


def _after_safety(state: ControllerState) -> str:
    """Route after safety check: back to orchestrator on violation, else programmer."""
    return "orchestrator" if state.get("error") else "programmer"


def _should_continue(state: ControllerState) -> str:
    """Route after evaluator: loop or end.

    Termination is controlled by the deterministic evaluator (eval_result["solved"]),
    not the LLM's convergence claim.  state["converged"] is recorded as a research
    measurement — how accurately the LLM recognises its own success.
    """
    if state.get("error"):
        return "end"
    if state.get("eval_result", {}).get("solved", False):
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
        {"orchestrator": "orchestrator", "programmer": "programmer"},
    )
    graph.add_edge("programmer", "evaluator")
    graph.add_conditional_edges(
        "evaluator",
        _should_continue,
        {"orchestrator": "orchestrator", "end": END},
    )

    return graph.compile()
