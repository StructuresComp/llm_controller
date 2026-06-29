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
from src.core import config
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

# ---------------------------------------------------------------------------
# 5. NODES
# ---------------------------------------------------------------------------

def evaluator_node(state: ControllerState) -> dict:
    """Look up the active benchmark, compute its loss, and stream snapshots to disk."""
    nodes  = state["shape_history"][-1] if state["shape_history"] else []
    target = state["target_shape"]

    if not nodes or not target:
        return {"eval_result": {}}

    try:
        benchmark = get_benchmark(target)
    except KeyError:
        return {"eval_result": {}}

    result = benchmark.loss_fn(state["latest_config"], state["physical_system"])
    result["solved"] = (result.get("error_percent", 100.0) / 100.0) <= benchmark.success_threshold

    pct = result["error_percent"]
    thr = benchmark.success_threshold * 100
    if result["solved"]:
        print(f"[Evaluator] SOLVED  error={pct:.2f}% <= threshold={thr:.1f}%")
    else:
        print(f"[Evaluator] error={pct:.2f}%  (threshold={thr:.1f}%)")

    # Write q_history snapshots to JSONL with the computed error_percent
    config_writer = state.get("config_writer")
    q_history     = state.get("q_history", [])
    bc_current    = state.get("bc_current", [0.0, 0.0, 0.0])
    ctime_current = state.get("ctime_current", 0.0)
    iteration_num = len(state.get("iteration_log", []))  # programmer already added entry
    orchestrator_prompt = state.get("orchestrator_prompt", None)

    if config_writer and q_history:
        for idx, q in enumerate(q_history):
            snapshot_data = {
                "iteration": iteration_num,
                "bc": {"x0": bc_current[0], "y0": bc_current[1], "theta": bc_current[2]},
                "q": q,
                "error_percent": pct,
                "timestamp": ctime_current,
            }
            # Add prompt only to first snapshot of this iteration
            if idx == 0 and orchestrator_prompt:
                snapshot_data["orchestrator_prompt"] = orchestrator_prompt
            config_writer.append_snapshot(**snapshot_data)

    error_history = (state.get("error_history", []) + [pct])[-(config.K_HISTORY + 1):]

    return {"eval_result": result, "error_history": error_history}


# ---------------------------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------------------------

def main():
    """Standalone test: evaluate the ideal C-shape reference (expect ~0% error)."""
    from src.models.physical_system import make_physical_system

    ps        = make_physical_system(config)
    benchmark = get_benchmark("C")
    ref       = benchmark.reference_fn(ps=ps)          # ndarray (nv, 2)
    q         = ref.flatten().tolist()
    obs_nodes = [[float(ref[i, 0]), float(ref[i, 1])] for i in ps.observable_nodes]

    state = {
        "shape_history":   [obs_nodes],
        "latest_config":   q,
        "target_shape":    "C",
        "physical_system": ps,
        "error_history":   [],
        "q_history":       [],
        "config_writer":   None,
        "bc_current":      [0.0, 0.0, 0.0],
        "ctime_current":   0.0,
        "iteration_log":   [],
        "orchestrator_prompt": None,
    }

    result = evaluator_node(state)
    ev     = result["eval_result"]
    print(f"Shape: C  |  error: {ev['error_percent']:.2f}%  |  solved: {ev['solved']}")


# ---------------------------------------------------------------------------
# 8. ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
