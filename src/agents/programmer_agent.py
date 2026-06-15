# ---------------------------------------------------------------------------
# src/agents/programmer_agent.py
# ---------------------------------------------------------------------------
# Programmer LangGraph node.
#
# Receives a schema-validated BC decision from the orchestrator, runs the
# physics simulation, updates all state fields, and writes a deterministic
# natural-language report. Zero LLM calls — entirely deterministic Python.
#
# Section order:
#   1. Imports  2. Schemas  3. Prompts  4. Helpers  5. Nodes  7. Main  8. Entry
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 1. IMPORTS
# ---------------------------------------------------------------------------
from src.core import config
from src.models.state_llm_control import ControllerState
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

def extract_observable_q(q: list, ps) -> list:
    """Return [x, y] pairs for each node in ps.observable_nodes."""
    obs = []
    for i in ps.observable_nodes:
        obs.append([q[2*i], q[2*i + 1]])
    return obs


def format_result(result: dict, decision: dict, prev_bc: list, ps) -> str:
    """Build a deterministic natural-language report from one simulation run."""
    x0_prev, y0_prev, theta_prev = prev_bc
    x0_new  = decision["x0"]
    y0_new  = decision["y0"]
    theta_new = decision["theta"]
    nv      = ps.nv
    q       = result["q_new"]
    obs_idx = ps.observable_nodes

    if ps.walls:
        wall_str = "; ".join(
            f"y={w['d']/w['n'][1]:.4f} m" if abs(w['n'][1]) >= abs(w['n'][0])
            else f"x={w['d']/w['n'][0]:.4f} m"
            for w in ps.walls
        )
        wall_line = f"Wall(s) at: {wall_str}."
    else:
        wall_line = "No wall."

    lines = [
        f"Moved node 0 from ({x0_prev:.4f}, {y0_prev:.4f}) θ={theta_prev:.4f} rad "
        f"to ({x0_new:.4f}, {y0_new:.4f}) θ={theta_new:.4f} rad.",
        wall_line,
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


# ---------------------------------------------------------------------------
# 5. NODES
# ---------------------------------------------------------------------------

def programmer_node(state: ControllerState) -> dict:
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
        save_stride      = config.SAVE_STRIDE,
    )

    if result["status"] == "failed_convergence":
        return {"error": f"Programmer: {result['message']}"}

    q_new = result["q_new"]
    q_history = result.get("q_history", [q_new])
    reaction_force = result.get("reactionForce")
    contact_x = result.get("cx")
    contact_y = result.get("cy")

    # --- Update continuity fields ---
    bc_current = [decision["x0"], decision["y0"], decision["theta"]]

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
        "latest_config":     q_new,
        "q_history":         q_history,
        "shape_history":     shape_history,
        "bc_history":        bc_history,
        "reaction_force":    reaction_force,
        "contact_x":         contact_x,
        "contact_y":         contact_y,
        "result":            report,
        "iteration_log":     state["iteration_log"] + [iteration_entry],
        "error":             None,
    }


# ---------------------------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------------------------

def main():
    """Standalone test: hardcoded decision, single run from initial state."""
    from src.models.physical_system import make_physical_system
    from src.models.state_llm_control import initial_controller_state

    ps    = make_physical_system(config)
    state = initial_controller_state(ps, target_shape="C", bc_target=[0.0, -0.05, 0.5])
    state["decision"] = {"x0": 0.0, "y0": -0.03, "theta": 0.0}

    updates = programmer_node(state)

    if updates.get("error"):
        print(f"[ERROR] {updates['error']}")
    else:
        print(updates["result"])
        print(f"\nSnapshots generated: {len(updates['q_history'])}")
        print(f"Shape history:      {len(updates['shape_history'])} entries")


# ---------------------------------------------------------------------------
# 8. ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
