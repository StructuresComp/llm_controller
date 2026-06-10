# ---------------------------------------------------------------------------
# src/agents/evaluator_agent.py
# ---------------------------------------------------------------------------
# Deterministic evaluator node.
#
# Looks up the active benchmark from the registry, runs its loss function
# against the current beam configuration, and writes the result to
# state["eval_result"] so the orchestrator can include the ground-truth
# loss in its next decision prompt.
#
# Currently deterministic. Designed to be swappable for an LLM/VLM-based
# evaluator later without changing the graph wiring.
#
# Section order (CLAUDE.md convention):
#   1. Imports  2. Schemas  3. Prompts  4. Helpers  5. Nodes  7. Main  8. Entry
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 1. IMPORTS
# ---------------------------------------------------------------------------
from src.benchmarks import get_benchmark
from src.models.state_llm_control import ControllerState


# ---------------------------------------------------------------------------
# 2. SCHEMAS
# ---------------------------------------------------------------------------
# No schemas needed — deterministic, no LLM calls.


# ---------------------------------------------------------------------------
# 3. PROMPTS
# ---------------------------------------------------------------------------
# No LLM prompts needed.


# ---------------------------------------------------------------------------
# 4. HELPERS
# ---------------------------------------------------------------------------

def _latest_observable(state: ControllerState) -> list:
    """Return the most recent observable configuration, or empty list."""
    if not state["shape_history"]:
        return []
    return state["shape_history"][-1]


# ---------------------------------------------------------------------------
# 5. NODES
# ---------------------------------------------------------------------------

def evaluator_node(state: ControllerState) -> dict:
    """Look up the active benchmark and compute its loss against current shape."""
    nodes  = _latest_observable(state)
    target = state["target_shape"]

    if not nodes or not target:
        return {"eval_result": {}}

    try:
        benchmark = get_benchmark(target)
    except KeyError:
        return {"eval_result": {}}

    result = benchmark.loss_fn(nodes, state["physical_system"])
    result["solved"] = (result.get("error_percent", 100.0) / 100.0) <= benchmark.success_threshold

    pct = result["error_percent"]
    thr = benchmark.success_threshold * 100
    if result["solved"]:
        print(f"[Evaluator] SOLVED  error={pct:.2f}% ≤ threshold={thr:.1f}%")
    else:
        print(f"[Evaluator] error={pct:.2f}%  (threshold={thr:.1f}%)")

    return {"eval_result": result}


# ---------------------------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------------------------

def main():
    """Standalone test: evaluate the ideal C-shape reference (expect ~0% error)."""
    from src.benchmarks import get_benchmark

    benchmark = get_benchmark("C")
    nodes     = benchmark.reference_fn()

    state = {
        "shape_history":   [nodes],
        "target_shape":    "C",
        "physical_system": {},
    }

    result = evaluator_node(state)
    ev     = result["eval_result"]
    print(f"Shape: C  |  error: {ev['error_percent']:.2f}%  |  {ev['diagnostics']}")


# ---------------------------------------------------------------------------
# 8. ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
