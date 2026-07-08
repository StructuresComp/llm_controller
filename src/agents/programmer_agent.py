# ---------------------------------------------------------------------------
# src/agents/programmer_agent.py
# ---------------------------------------------------------------------------
# Programmer LangGraph node.
#
# Receives a schema-validated BC decision from the orchestrator, runs the
# physics simulation, and updates all state fields.
# Zero LLM calls — entirely deterministic Python.
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

def append_and_trim(history: list, item, max_len: int) -> list:
    """Append item to list and keep only the last max_len entries."""
    return (history + [item])[-max_len:]


# ---------------------------------------------------------------------------
# 5. NODES
# ---------------------------------------------------------------------------

def programmer_node(state: ControllerState) -> dict:
    decision = state["decision"]          # {x0, y0, theta} — schema-validated
    ps       = state["physical_system"]
    iteration_num = state["iteration"] + 1  # computed early: also used for the live plot title

    # --- Run simulation (retry with smaller dt if convergence fails) ---
    dt = config.DT
    while True:
        result = discreteBeam(
            physical_system  = ps,
            q0               = state["q0_current"],
            u0               = state["u0_current"],
            t_start          = state["ctime_current"],
            control_position = (decision["x0"], decision["y0"]),
            control_angle    = decision["theta"],
            dt               = dt,
            plotImage        = config.PLOT_IMAGE,
            save_stride      = config.SAVE_STRIDE,
            iteration        = iteration_num,
        )
        if result["status"] != "failed_convergence":
            break
        if not config.ADAPT_DT:
            break
        dt *= config.DT_MULTIPLIER
        if dt < config.DT_MIN:
            break
        print(f"  [programmer] Failed at dt={dt/config.DT_MULTIPLIER:.6f}. Retrying at dt={dt:.6f}.")

    if result["status"] == "failed_convergence":
        error = f"Programmer: {result['message']}"
        if not config.QUIT_ON_DT_FAILURE:
            return {"error": error, "dt_failure_rejected": True}
        return {"error": error}

    q_new              = result["q_new"]
    q_history          = result.get("q_history", [q_new])
    reaction_force     = result.get("reactionForce")
    contact_x          = result.get("cx")
    contact_y          = result.get("cy")
    kappa_dimensional  = result["kappa_dimensional"]
    axial_strain       = result["axial_strain"]

    # --- Update continuity fields ---
    bc_current = [decision["x0"], decision["y0"], decision["theta"], decision.get("reason", "")]

    # --- Update rolling histories (observable nodes only) ---
    obs_q     = [[q_new[2*i], q_new[2*i + 1]] for i in ps.observable_nodes]
    obs_kappa = [float(kappa_dimensional[i])   for i in ps.observable_nodes]
    # K_HISTORY previous steps + 1 current = K_HISTORY + 1 entries needed so that
    # the prompt builder slice [-(K_HISTORY+1):-1] exposes exactly K_HISTORY previous steps.
    _keep = config.K_HISTORY + 1
    shape_history = append_and_trim(state["shape_history"],  obs_q,      _keep)
    kappa_history = append_and_trim(state["kappa_history"],  obs_kappa,  _keep)
    bc_history    = append_and_trim(state["bc_history"],     bc_current, _keep)

    # --- Log this iteration ---
    iteration_entry = {
        "iteration": iteration_num,
        "decision":  decision,
        "tip_x":     q_new[-2],
        "tip_y":     q_new[-1],
    }

    return {
        "q0_current":        q_new,
        "u0_current":        result["u_new"],
        "ctime_current":     result["ctime"],
        "bc_current":        bc_current,
        "latest_config":     q_new,
        "kappa_dimensional": kappa_dimensional,
        "axial_strain":      axial_strain,
        "q_history":         q_history,
        "shape_history":     shape_history,
        "kappa_history":     kappa_history,
        "bc_history":        bc_history,
        "reaction_force":    reaction_force,
        "contact_x":         contact_x,
        "contact_y":         contact_y,

        "iteration":         iteration_num,
        "iteration_log":     state["iteration_log"] + [iteration_entry],
        "error":             None,
        "dt_failure_rejected": False,
    }


# ---------------------------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------------------------

def main():
    """Standalone test: hardcoded decision, single run from initial state."""
    from src.models.physical_system import make_physical_system
    from src.models.state_llm_control import initial_controller_state

    ps    = make_physical_system(config)
    state = initial_controller_state(ps, target_shape="C")
    state["decision"] = {"x0": 0.0, "y0": -0.03, "theta": 0.0}

    updates = programmer_node(state)

    if updates.get("error"):
        print(f"[ERROR] {updates['error']}")
    else:
        print(f"Snapshots generated: {len(updates['q_history'])}")
        print(f"Shape history:      {len(updates['shape_history'])} entries")


# ---------------------------------------------------------------------------
# 8. ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
