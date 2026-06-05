# ---------------------------------------------------------------------------
# src/agents/programmer_agent.py
# ---------------------------------------------------------------------------
# Programmer LangGraph node.
#
# Receives a schema-validated BC decision from the orchestrator, runs the
# physics simulation, updates all state fields, and writes a deterministic
# natural-language report. Zero LLM calls — entirely deterministic Python.
#
# Section order (CLAUDE.md convention):
#   1. Imports  2. Schemas  3. Prompts  4. Helpers  5. Nodes  7. Main  8. Entry
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 1. IMPORTS
# ---------------------------------------------------------------------------
import config
from src.models.state_llm_control import ESLState
from src.tools.Discrete_Elastic_Beams import discreteBeam


# ---------------------------------------------------------------------------
# 2. SCHEMAS
# ---------------------------------------------------------------------------
# No schemas needed — decision arrives as a validated dict from the orchestrator.


# ---------------------------------------------------------------------------
# 3. PROMPTS
# ---------------------------------------------------------------------------
# No LLM prompts needed — report is built deterministically.


# ---------------------------------------------------------------------------
# 4. HELPERS
# ---------------------------------------------------------------------------

def extract_observable_q(q: list, ps: dict) -> list:
    """Return [x, y] pairs for each node in ps['observable_nodes']."""
    obs = []
    for i in ps["observable_nodes"]:
        obs.append([q[2*i], q[2*i + 1]])
    return obs


def format_result(result: dict, decision: dict, prev_bc: list, ps: dict) -> str:
    """Build a deterministic natural-language report from one simulation run."""
    x0_prev, y0_prev, theta_prev = prev_bc
    x0_new  = decision["x0"]
    y0_new  = decision["y0"]
    theta_new = decision["theta"]
    wall_H  = ps["wall_H"]
    nv      = ps["nv"]
    q       = result["q_new"]
    obs_idx = ps["observable_nodes"]

    lines = [
        f"Moved node 0 from ({x0_prev:.4f}, {y0_prev:.4f}) θ={theta_prev:.4f} rad "
        f"to ({x0_new:.4f}, {y0_new:.4f}) θ={theta_new:.4f} rad.",
        f"Wall is at y = {wall_H:.4f} m.",
        f"Simulation: {result['message']}",
        "",
        "DLO configuration (observable nodes):",
    ]

    for i in obs_idx:
        x, y = q[2*i], q[2*i + 1]
        label = ""
        if i == 0:
            label = "  [controlled end]"
        elif i == nv - 1:
            label = "  [tip]"
        lines.append(f"  Node {i:>3d}:  x = {x:+.6f} m,  y = {y:+.6f} m{label}")

    return "\n".join(lines)


def append_and_trim(history: list, item, max_len: int) -> list:
    """Append item to list and keep only the last max_len entries."""
    return (history + [item])[-max_len:]


def downsample_configs(configs: dict, max_configs: int) -> dict:
    """
    If len(configs) > max_configs, halve by deleting every other entry
    starting from index 1. Index 0 (initial configuration) is always kept.
    Re-indexes remaining entries as 0, 1, 2, ...
    """
    if len(configs) <= max_configs:
        return configs
    sorted_keys = sorted(configs)
    kept = [sorted_keys[0]] + [k for k in sorted_keys[1:] if sorted_keys.index(k) % 2 == 0]
    return {new_k: configs[old_k] for new_k, old_k in enumerate(kept)}


# ---------------------------------------------------------------------------
# 5. NODES
# ---------------------------------------------------------------------------

def programmer_node(state: ESLState) -> dict:
    decision = state["decision"]          # {x0, y0, theta} — schema-validated
    ps       = state["physical_system"]
    prev_bc  = list(state["bc_current"])

    # --- Run simulation ---
    result = discreteBeam(
        physical_system  = ps,
        q0               = state["q0_current"],
        u0               = state["u0_current"],
        t_start          = state["ctime_current"],
        control_position = (decision["x0"], decision["y0"]),
        control_angle    = decision["theta"],
        dt               = config.DT,
        plotImage        = config.PLOT_IMAGE,
    )

    if result["status"] == "failed_convergence":
        return {"error": f"Programmer: {result['message']}"}

    q_new = result["q_new"]

    # --- Update continuity fields ---
    bc_current = [decision["x0"], decision["y0"], decision["theta"]]

    # --- Update all_configurations (append + downsample) ---
    configs     = dict(state["all_configurations"])
    run_no      = len(configs)
    configs[run_no] = q_new
    configs     = downsample_configs(configs, config.MAX_CONFIGS)

    # --- Update rolling histories (observable nodes only) ---
    obs_q   = extract_observable_q(q_new, ps)
    shape_history = append_and_trim(state["shape_history"], obs_q,   config.K_HISTORY)
    bc_history    = append_and_trim(state["bc_history"],    bc_current, config.K_HISTORY)

    # --- Build deterministic text report ---
    report = format_result(result, decision, prev_bc, ps)

    # --- Log this iteration ---
    iteration_num = len(state["iteration_log"]) + 1
    iteration_entry = {
        "iteration": iteration_num,
        "decision": decision,
        "tip_x": q_new[-2],
        "tip_y": q_new[-1],
    }

    return {
        "q0_current":        q_new,
        "u0_current":        result["u_new"],
        "ctime_current":     result["ctime"],
        "bc_current":        bc_current,
        "all_configurations": configs,
        "shape_history":     shape_history,
        "bc_history":        bc_history,
        "result":            report,
        "iteration_log":     state["iteration_log"] + [iteration_entry],
        "error":             None,
    }


# ---------------------------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------------------------

def main():
    """Standalone test: hardcoded decision, single run from initial state."""
    import config
    from src.models.physical_system import make_physical_system
    from src.models.state_llm_control import initial_esl_state

    ps    = make_physical_system(config)
    state = initial_esl_state(ps, target_shape="C", bc_target=[0.0, -0.05, 0.5])
    state["decision"] = {"x0": 0.0, "y0": -0.03, "theta": 0.0}

    updates = programmer_node(state)

    if updates.get("error"):
        print(f"[ERROR] {updates['error']}")
    else:
        print(updates["result"])
        print(f"\nConfigs stored: {len(updates['all_configurations'])}")
        print(f"Shape history:  {len(updates['shape_history'])} entries")


# ---------------------------------------------------------------------------
# 8. ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
