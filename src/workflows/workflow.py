# ---------------------------------------------------------------------------
# src/workflows/workflow.py
# ---------------------------------------------------------------------------
# LangGraph master workflow.
#
# Topology:
#   [START] → orchestrator → programmer → (converged or error?) → [END]
#                    ↑___________________________|
#
# The orchestrator is itself a compiled sub-graph (planner → decision).
# The loop runs until state["converged"] is True, state["error"] is set,
# or LangGraph's recursion_limit is hit (set to MAX_ITERATIONS * 4 in cli.py
# to account for multiple node invocations per cycle).
# ---------------------------------------------------------------------------

from langgraph.graph import END, StateGraph

from src.models.state_llm_control import ControllerState
from src.agents.orchestrator_agent import build_orchestrator_graph
from src.agents.programmer_agent import programmer_node
from src.agents.evaluator_agent import evaluator_node


def _should_continue(state: ControllerState) -> str:
    """Route after evaluator: loop or end."""
    if state.get("error"):
        return "end"
    if state["converged"]:
        return "end"
    return "orchestrator"


def build_controller_graph():
    graph = StateGraph(ControllerState)

    graph.add_node("orchestrator", build_orchestrator_graph())
    graph.add_node("programmer",   programmer_node)
    graph.add_node("evaluator",    evaluator_node)

    graph.set_entry_point("orchestrator")
    graph.add_edge("orchestrator", "programmer")
    graph.add_edge("programmer",   "evaluator")
    graph.add_conditional_edges(
        "evaluator",
        _should_continue,
        {"orchestrator": "orchestrator", "end": END},
    )

    return graph.compile()
