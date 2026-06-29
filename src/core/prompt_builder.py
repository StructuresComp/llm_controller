# ---------------------------------------------------------------------------
# src/core/prompt_builder.py
# ---------------------------------------------------------------------------
# Compose orchestrator prompts from modular observation functions.
#
# Public API:
#   build_planner_prompt(state)  -> str   (used by planner_node)
#   build_decision_prompt(state) -> str   (used by decision_node)
#
# Used by: src/agents/orchestrator_agent.py
# ---------------------------------------------------------------------------

from src.benchmarks import get_benchmark
from src.core.observation import (
    describe_beam_physics,
    describe_wall_constraints,
    describe_bc_constraints,
    describe_current_shape,
    describe_history,
    describe_error_rejection,
    describe_controller_feedback,
    describe_contact_regions,
    describe_axial_stretch,
)
from src.core import config


# ---------------------------------------------------------------------------
# STATIC TEXT CONSTANTS
# ---------------------------------------------------------------------------

_DESCRIBE_COORDINATE_SYSTEM = """\
Coordinate system:
  X-axis: points to the right
  Y-axis: points upward
  Z-axis: points out of the plane (toward you)
  theta: angle of the controlled end at s=0, measured counter-clockwise from the +X axis (standard math convention).
         0 = pointing right, pi/2 = pointing up, -pi/2 = pointing straight down (initial state), pi = pointing left."""


# ---------------------------------------------------------------------------
# INTERNAL
# ---------------------------------------------------------------------------

def _build_base_prompt(state) -> list:
    """Build Tiers 1-3: coordinate system, physics, walls, BC, shape, target, feedback, history."""
    ps        = state["physical_system"]
    benchmark = get_benchmark(state["target_shape"])
    q         = state["latest_config"]
    kappa     = state.get("kappa_dimensional") if config.INCLUDE_GEOMETRY else None

    # ── Tier 1: Static context ────────────────────────────────────────────────────────────────
    sections = [
        _DESCRIBE_COORDINATE_SYSTEM,
        describe_beam_physics(ps),
        describe_wall_constraints(ps),
        describe_bc_constraints(ps, q),
        describe_current_shape(q, ps, kappa=kappa, bc=state.get("bc_current"), decision_num=state["iteration"] + 1),
        benchmark.describe_fn(q, ps),
    ]

    # ── Tier 2: Feedback & contact (conditional on config flags) ─────────────────────────────
    if config.INCLUDE_FEEDBACK:
        feedback = describe_controller_feedback(state.get("reaction_force"), ps, q)
        if feedback:
            sections.append(feedback)

    if config.INCLUDE_CONTACT:
        contact = describe_contact_regions(state.get("contact_x"), state.get("contact_y"), ps.nv, ps.walls)
        if contact:
            sections.append(contact)

    if config.INCLUDE_AXIAL_STRETCH:
        stretch = describe_axial_stretch(state.get("axial_strain"), ps)
        if stretch:
            sections.append(stretch)

    # ── Tier 3: History ───────────────────────────────────────────────────────────────────────
    # [-(k+1):-1] gives the k steps before the current one; Python clamps if history is shorter.
    k = config.K_HISTORY
    if k > 0 and len(state["shape_history"]) > 1:
        kappa_hist = state.get("kappa_history", [])[-(k+1):-1] if config.INCLUDE_GEOMETRY else None
        error_hist = state.get("error_history", [])[-(k+1):-1] or None
        sections.append(describe_history(
            state["bc_history"][-(k+1):-1],
            state["shape_history"][-(k+1):-1],
            ps,
            error_hist,
            kappa_history=kappa_hist,
        ))

    return sections


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def build_planner_prompt(state) -> str:
    """Build the prompt for planner_node: base context + previous plan (if replanning)."""
    sections = _build_base_prompt(state)

    # In replan mode, state["plan"] holds the previous plan; show it so the planner can
    # revise rather than regenerate from scratch.
    if state.get("plan"):
        sections.append(
            "Output only the revised plan. No analysis, no preamble, no headers, no reasoning.\n\n"
            "Previous plan:\n" + state["plan"] + "\n\n"
            "The shapes above show what the worker agent produced under your plan. "
            "Revise if needed, or keep unchanged. "
            "Give only precise instructions for the worker agent."
        )

    return "\n\n".join(s for s in sections if s)


def build_decision_prompt(state) -> str:
    """Build the prompt for decision_node: base context + plan guidance + error alert."""
    sections = _build_base_prompt(state)

    if state.get("plan"):
        sections.append(
            "Plan for this task:\n"
            "A plan was generated to guide the deformation toward the target.\n"
            "Follow it, but adapt if the beam is not responding as expected.\n\n"
            + state["plan"]
        )

    if state.get("error"):
        sections.append(describe_error_rejection(state["error"]))

    return "\n\n".join(s for s in sections if s)
