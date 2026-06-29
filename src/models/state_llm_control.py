# ---------------------------------------------------------------------------
# src/models/state_llm_control.py
# ---------------------------------------------------------------------------
# Shared working state for the LLM controller pipeline.
# ---------------------------------------------------------------------------

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
import numpy as np


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
    bc_history:     list   # list of [x0, y0, theta] — last K BCs issued
    shape_history:  list   # list of partial configs (N_PARTIAL nodes each)
    kappa_history:  list   # list of observable-node kappa arrays — parallel to shape_history
    error_history:  list   # list of error_percent floats — parallel to shape_history

    # ============ CURRENT SNAPSHOT BUFFER
    # q_history from the last programmer call, awaiting evaluation and disk write.
    # Cleared after evaluator writes to JSONL. List of full DOF vectors.
    q_history:     list

    # ============ LATEST CONFIGURATION
    # Final DOF vector from last programmer call. Streamed snapshots go to JSONL;
    # state keeps only the latest config for reference and LLM observations.
    latest_config:      list | None
    kappa_dimensional:  object   # np.ndarray length nv: dimensional curvature (1/m) at each node
    axial_strain:       object   # np.ndarray length nv-1: axial strain per edge (positive = stretched)

    # ============ DISK I/O
    # ConfigWriter instance for streaming trajectory snapshots to disk.
    # Initialized at CLI startup, closed at end. Created in __init__, never in state factory.
    config_writer: object  # ConfigWriter instance (not TypedDict-serializable, so type: object)

    # ============ TARGET & CONVERGENCE
    target_shape: str    # "C", "U", "S", etc. — set from CLI --target
    converged:    bool

    # ============ ORCHESTRATOR
    # decision is a validated dict produced by call_llm_with_tool — never a raw string.
    # Schema: {"x0": float, "y0": float, "theta": float}
    decision:  dict  # BC decision from orchestrator (structured, schema-enforced)
    plan:   str   # high-level strategy (plan mode only; "" in myopic mode)
    result: str   # programmer's natural-language summary of last run

    # ============ ITERATION TRACKING
    iteration:     int   # canonical iteration counter; incremented by programmer_node each cycle
    iteration_log: list  # list of {iteration, decision, tip_pos} — one per completed iteration

    # ============ EVALUATOR
    # Geometric shape error computed by evaluator_node after each programmer run.
    # Keys: error_percent, confidence_percent, rms_normalized, fitted_params, diagnostics.
    # Empty dict before first evaluation or when target_shape is unset.
    eval_result: dict

    # ============ CONTACT STATE & FEEDBACK (from discreteBeam)
    # Reaction force felt by the controller (robot arm) at the clamped end.
    # Computed as: F_feedback = -sum(reactionForce[0:4]) (first two nodes).
    # List of floats [Fx, Fy, ...] for all DOFs, or None.
    reaction_force: list | None

    # Wall contact state: which wall (if any) pins each node's x-DOF.
    # cx[i] = wall index pinning node i's x-DOF, or -1 if free. None if no walls.
    contact_x: list | None

    # Wall contact state: which wall (if any) pins each node's y-DOF.
    # cy[i] = wall index pinning node i's y-DOF, or -1 if free. None if no walls.
    contact_y: list | None

    # ============ GLOBAL
    error: str | None


def initial_controller_state(
    physical_system,
    target_shape: str = "",
    config_writer=None,
) -> ControllerState:
    from src.models.physical_system import initial_q0, initial_u0
    return {
        "messages":          [],
        "physical_system":   physical_system,
        "q0_current":        initial_q0(physical_system),
        "u0_current":        initial_u0(physical_system),
        "ctime_current":     0.0,
        "bc_current":        [0.0, 0.0, -np.pi / 2],
        "bc_history":        [],
        "shape_history":     [],
        "kappa_history":     [],
        "error_history":     [],
        "q_history":         [],
        "latest_config":     initial_q0(physical_system),
        "kappa_dimensional": np.zeros(physical_system.nv),
        "axial_strain":      np.zeros(physical_system.nv - 1),
        "config_writer":     config_writer,
        "target_shape":      target_shape,
        "converged":         False,
        "decision":          {},
        "plan":   "",
        "result": "",
        "iteration":         0,
        "iteration_log":     [],
        "eval_result":       {},
        "reaction_force":    None,
        "contact_x":         None,
        "contact_y":         None,
        "error":             None,
    }
