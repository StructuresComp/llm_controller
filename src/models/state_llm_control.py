# ---------------------------------------------------------------------------
# src/models/state_llm_control.py
# ---------------------------------------------------------------------------
# Shared working state for the LLM controller pipeline.
#
# Pipeline (current):  orchestrator → programmer
# Pipeline (future):   orchestrator → programmer → (converged?) → loop back
# ---------------------------------------------------------------------------

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class ControllerState(TypedDict):

    # ============ MESSAGES
    messages: Annotated[list, add_messages]

    # ============ PHYSICAL SYSTEM
    # Pre-computed physical and derived quantities. Built once at startup
    # from config.py via make_physical_system(). Never modified during a run.
    physical_system: object  # BeamPhysicalSystem or RodPhysicalSystem

    # ============ SIMULATION CONTINUITY
    # Position, velocity, and time at the end of the last tool call.
    # Passed directly to the next discreteBeam call so runs chain seamlessly.
    q0_current:    list   # DOF vector, length 2*nv
    u0_current:    list   # velocity vector, length 2*nv
    ctime_current: float  # cumulative simulation time (seconds); used as t_start

    # ============ BOUNDARY CONDITION
    # Current active BC. Layout: [x0, y0, theta]  (stored as list for JSON serialisability)
    bc_current: list   # BC at the end of the last run

    # ============ HISTORY (rolling window, length = K_HISTORY)
    bc_history:    list   # list of [x0, y0, theta] — last K BCs issued
    shape_history: list   # list of partial configs (N_PARTIAL nodes each)

    # ============ ALL CONFIGURATIONS
    # Full DOF vector from every physics step, keyed by sequential int index.
    # Capped at MAX_CONFIGS; oldest trimmed when exceeded.
    all_configurations: dict

    # Per-frame overlay labels for the viewer.  Same keys as all_configurations.
    # Set on the last physics frame of each LLM decision so the viewer can
    # display the control action that produced each state.
    # Format: {frame_idx: "Step N  x=+0.00  y=−0.00  θ=+0.00"}
    frame_labels: dict

    # ============ TARGET & CONVERGENCE
    target_shape: str    # "C", "U", "S", etc. — set from CLI --target
    converged:    bool

    # ============ ORCHESTRATOR
    # decision is a validated dict produced by call_llm_with_tool — never a raw string.
    # Schema: {"x0": float, "y0": float, "theta": float}
    decision: dict  # BC decision from orchestrator (structured, schema-enforced)
    plan:     str   # high-level strategy (plan mode only; "" in myopic mode)
    result:   str   # programmer's natural-language summary of last run

    # ============ ITERATION LOG
    iteration_log: list  # list of {iteration, decision, tip_pos} — one per completed iteration

    # ============ EVALUATOR
    # Geometric shape error computed by evaluator_node after each programmer run.
    # Keys: error_percent, confidence_percent, rms_normalized, fitted_params, diagnostics.
    # Empty dict before first evaluation or when target_shape is unset.
    eval_result: dict

    # ============ GLOBAL
    error: str | None


def initial_controller_state(
    physical_system,
    target_shape: str = "",
) -> ControllerState:
    from src.models.physical_system import initial_q0, initial_u0
    return {
        "messages":          [],
        "physical_system":   physical_system,
        "q0_current":        initial_q0(physical_system),
        "u0_current":        initial_u0(physical_system),
        "ctime_current":     0.0,
        "bc_current":        [0.0, 0.0, 0.0],
        "bc_history":        [],
        "shape_history":     [],
        "all_configurations": {},
        "frame_labels":      {},
        "target_shape":      target_shape,
        "converged":         False,
        "decision":          {},
        "plan":              "",
        "result":            "",
        "iteration_log":     [],
        "eval_result":       {},
        "error":             None,
    }
