# ---------------------------------------------------------------------------
# src/workflows/workflow.py
# ---------------------------------------------------------------------------
# LangGraph master workflow.
#
# Topology:
#   [START] → orchestrator ──(LLM tool call failed, all retries exhausted)──→ [END]
#                  ↑        └──(ok)──→ safety ──(violation)──→ orchestrator
#                  |                       └──(safe)──→ programmer ──(dt exhausted, QUIT_ON_DT_FAILURE=False)──→ orchestrator
#                  |                                        |──(dt exhausted, QUIT_ON_DT_FAILURE=True)──→ [END]
#                  |                                        └──(success)──→ evaluator
#                  └───────────────────────────────────────────────────────────────(loop)──┘
#
# The orchestrator is itself a compiled sub-graph (planner → decision). decision_node
# retries a failed-to-parse LLM tool call up to config.TOOL_CALL_ATTEMPTS times before
# giving up; exhausting all attempts ends the run immediately (skips safety/programmer/
# evaluator entirely, since there's no valid decision to act on).
# The loop runs until state["converged"] is True, an unrecoverable dt failure
# ends the run (QUIT_ON_DT_FAILURE=True), state["attempt_count"] (incremented every
# decision, including rejections) exceeds MAX_ITERATIONS_WITH_REJECTIONS, or
# LangGraph's recursion_limit is hit.
# Safety violations and recoverable dt-failure rejections both route back to the
# orchestrator — they never reach the evaluator, so they do not count as iterations.
# ---------------------------------------------------------------------------

from langgraph.graph import END, StateGraph

from src.core import config
from src.models.state_llm_control import ControllerState
from src.agents.orchestrator_agent import build_orchestrator_graph
from src.agents.safety_agent import safety_node
from src.agents.programmer_agent import programmer_node
from src.agents.evaluator_agent import evaluator_node


def _after_orchestrator(state: ControllerState) -> str:
    """Route after orchestrator: end if the LLM tool call failed after all retries
    (config.TOOL_CALL_ATTEMPTS exhausted), else safety."""
    return "end" if state.get("error") else "safety"


def _after_safety(state: ControllerState) -> str:
    """Route after safety check: back to orchestrator on violation, else programmer.

    Hard-stops if total attempts (successful iterations + rejections) exceed
    MAX_ITERATIONS_WITH_REJECTIONS, so a stuck LLM can't loop on rejections forever.
    """
    if state.get("attempt_count", 0) >= config.MAX_ITERATIONS_WITH_REJECTIONS:
        return "end"
    return "orchestrator" if state.get("error") else "programmer"


def _after_programmer(state: ControllerState) -> str:
    """Route after programmer: end on unrecoverable dt failure (or QUIT_ON_DT_FAILURE),
    back to orchestrator on a recoverable dt-failure rejection, else evaluator."""
    if state.get("error"):
        return "end" if config.QUIT_ON_DT_FAILURE else "orchestrator"
    return "evaluator"


def _should_continue(state: ControllerState) -> str:
    """Route after evaluator: loop or end.

    Termination is controlled by the deterministic evaluator (eval_result["solved"]),
    not the LLM's convergence claim.  state["converged"] is recorded as a research
    measurement — how accurately the LLM recognises its own success.

    Hard stop: if MAX_ITERATIONS is reached, end regardless of solve status.
    """
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
    graph.add_conditional_edges(
        "orchestrator",
        _after_orchestrator,
        {"safety": "safety", "end": END},
    )
    graph.add_conditional_edges(
        "safety",
        _after_safety,
        {"orchestrator": "orchestrator", "programmer": "programmer", "end": END},
    )
    graph.add_conditional_edges(
        "programmer",
        _after_programmer,
        {"orchestrator": "orchestrator", "evaluator": "evaluator", "end": END},
    )
    graph.add_conditional_edges(
        "evaluator",
        _should_continue,
        {"orchestrator": "orchestrator", "end": END},
    )

    return graph.compile()
