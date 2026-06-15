# ---------------------------------------------------------------------------
# src/core/prompt_builder.py
# ---------------------------------------------------------------------------
# Compose orchestrator decision prompts from modular sub-prompts.
#
# This module orchestrates observation helpers to construct prompts in logical
# reading order. It does NOT generate text — it arranges pre-computed text.
#
# Public API:
#   build_decision_prompt(state) -> str
#
# Used by: src/agents/orchestrator_agent.py::decision_node()
# ---------------------------------------------------------------------------

from src.benchmarks import get_benchmark
from src.core.observation import (
    describe_beam_physics,
    describe_wall_constraints,
    describe_bc_constraints,
    describe_nodes,
    describe_history,
    describe_recent_trajectory,
    describe_error_rejection,
    describe_controller_feedback,
    describe_contact_regions,
)
from src.core import config


def build_plan_prompt(state) -> str:
    """
    Compose the planner's strategic prompt from Tier 1 sub-prompts.

    Called once per run (plan mode only) to generate a multi-step strategy.
    Reuses Tier 1 helpers (physics, walls, BC, control inputs) shared with
    the decision prompt to ensure consistent context.

    Args:
        state (ControllerState): Current controller state

    Returns:
        str: Full prompt ready for LLM with call_llm()

    Structure:
      Tier 1: Static context (once per run)
        1. Beam physics (stiffness intuition)
        2. Wall constraints (if present)
        3. BC constraints (clamped + free regions)
        4. Control inputs explanation
        5. Target task description + loss explanation
      Planning guidance
        6. Strategy development instructions
    """
    ps = state["physical_system"]
    benchmark = get_benchmark(state["target_shape"])

    parts = [
        "=" * 70,
        "ROBOTIC ARM STRATEGY PLANNING",
        "=" * 70,
        "",
    ]

    # Tier 1: Static context (reuse from decision prompt)
    parts.extend([describe_beam_physics(ps), ""])

    wall_prompt = describe_wall_constraints(ps)
    if wall_prompt:
        parts.extend([wall_prompt, ""])

    parts.extend([
        "Boundary conditions:",
        describe_bc_constraints(ps),
        "",
        "Coordinate system:",
        "  X-axis: points to the right (→)",
        "  Y-axis: points upward (↑)",
        "  Z-axis: points out of the plane (toward you)",
        "",
        "Your control inputs (sign convention):",
        "  x0 (meters): horizontal position of clamped end",
        "    x0 > 0: push right  |  x0 < 0: push left  |  x0 = 0: centered",
        "  y0 (meters): vertical position of clamped end",
        "    y0 < 0: push down (toward walls)  |  y0 > 0: pull up  |  y0 = 0: at origin",
        "  θ (radians): slope angle at clamped end (counter-clockwise from -Y axis)",
        "    θ =  0:    straight down       [direction: 0, -1]",
        "    θ =  π/2:  pointing right      [direction: +1, 0]",
        "    θ = -π/2:  pointing left       [direction: -1, 0]",
        "",
    ])

    # Target task and loss (reuse from decision)
    parts.extend([
        "Target task:",
        benchmark.target_description,
        "",
        "Success metric:",
        benchmark.loss_description,
        "",
    ])

    # Planning-specific guidance
    parts.extend([
        "STRATEGY (3-6 steps):",
        "Develop a high-level step-by-step plan:",
        "  1. Initial move: How will you position/rotate the clamped end to start bending?",
        "  2. Wall contact: When should the beam touch the wall? Where?",
        "  3-6. Refinement: What sequence of moves will shape it to match the target?",
        "",
        "Consider:",
        f"  - Stiffness: Higher EI ({ps.EI:.2e}) means the beam resists bending — expect slower progress.",
        "  - Gravity: Pulls the free end down. Can you use this to your advantage?",
        "  - Walls: Hard constraints. The beam cannot penetrate them.",
        "  - Incremental: Small moves accumulate. Plan for multiple iterations.",
    ])

    return "\n".join(parts)


def build_decision_prompt(state) -> str:
    """
    Compose the full decision prompt by concatenating sub-prompts in order.

    Args:
        state (ControllerState): Current controller state

    Returns:
        str: Full prompt ready for LLM with call_llm_with_tool()

    Structure (Tier 1 → 4):
      Tier 1: Static context (once per run) ✓ IMPLEMENTED
        1. Beam physics (stiffness intuition)
        2. Wall constraints (if present)
        3. BC constraints (clamped + free regions)
        4. Control inputs explanation
        5. Target task description + loss explanation

      Tier 2: Dynamic context (every iteration) ✓ IMPLEMENTED
        6. Current beam state (observable nodes + BC)
        7. Recent trajectory (last 5 moves, progress/stagnation)
        8. Full movement history (last K steps)
        9. Evaluation result (error %, diagnostics)

      Tier 3: Conditional guidance (conditional, every iteration) ✓ IMPLEMENTED
        10. Prior strategy (if plan mode & plan exists)
        11. Previous error (if last BC was rejected)

      Tier 4: Decision prompt
        12. Next move instructions
    """
    ps = state["physical_system"]
    benchmark = get_benchmark(state["target_shape"])

    parts = [
        "=" * 70,
        "ROBOTIC ARM CONTROL TASK",
        "=" * 70,
        "",
    ]

    # ─────────────────────────────────────────────────────────────────────
    # Tier 1: Static context (once per run)
    # ─────────────────────────────────────────────────────────────────────

    # Physics parameters with Euler-Bernoulli intuition
    parts.extend([
        describe_beam_physics(ps),
        "",
    ])

    # Wall constraints (empty if no walls — this is the point!)
    wall_prompt = describe_wall_constraints(ps)
    if wall_prompt:
        parts.extend([wall_prompt, ""])

    # BC constraints (depends on controlled_nodes)
    parts.extend([
        "Boundary conditions:",
        describe_bc_constraints(ps),
        "",
    ])

    # Coordinate system (left-hand engineering: X right, Y up, Z out of plane)
    parts.extend([
        "Coordinate system:",
        "  X-axis: points to the right (→)",
        "  Y-axis: points upward (↑)",
        "  Z-axis: points out of the plane (toward you)",
        "",
    ])

    # Control inputs explanation (sign convention)
    parts.extend([
        "Your control inputs (sign convention):",
        "  x0 (meters): horizontal position of clamped end",
        "    x0 > 0: push right  |  x0 < 0: push left  |  x0 = 0: centered",
        "  y0 (meters): vertical position of clamped end",
        "    y0 < 0: push down (toward walls)  |  y0 > 0: pull up  |  y0 = 0: at origin",
        "  θ (radians): slope angle at clamped end (counter-clockwise from -Y axis)",
        "    θ =  0:    straight down       [direction: 0, -1]",
        "    θ =  π/2:  pointing right      [direction: +1, 0]",
        "    θ = -π/2:  pointing left       [direction: -1, 0]",
        "",
    ])

    # Target task description (from benchmark)
    parts.extend([
        "Target task:",
        benchmark.target_description,
        "",
    ])

    # Loss explanation (from benchmark)
    parts.extend([
        "Success metric:",
        benchmark.loss_description,
        "",
    ])

    # ─────────────────────────────────────────────────────────────────────
    # Tier 2: Dynamic context (every iteration)
    # ─────────────────────────────────────────────────────────────────────

    # Current beam state (observable nodes + BC)
    if state["shape_history"]:
        current_state = describe_nodes(
            state["shape_history"][-1],
            state["bc_current"],
            level=config.OBSERVATION_LEVEL,
        )
    else:
        # Generate initial observable nodes: straight vertical (0, -i*deltaL)
        initial_nodes = []
        for idx in ps.observable_nodes:
            x = 0.0
            y = -idx * ps.deltaL
            initial_nodes.append((x, y))
        current_state = describe_nodes(
            initial_nodes,
            state["bc_current"],
            level=config.OBSERVATION_LEVEL,
        )

    parts.extend([
        "Current beam state:",
        current_state,
        "",
    ])

    # Recent trajectory (last 5 moves)
    if state["bc_history"]:
        trajectory = describe_recent_trajectory(
            state["bc_history"],
            state["shape_history"],
            n_recent=5,
        )
        parts.extend([
            "Recent progress (tip movement):",
            "How much has the tip moved with each of your last 5 commands? Progress or stagnation?",
            trajectory,
            "",
        ])

    # Full movement history (K most recent)
    if state["bc_history"]:
        # Limit to K_HISTORY most recent moves
        k = config.K_HISTORY
        recent_bc = state["bc_history"][-k:] if len(state["bc_history"]) > k else state["bc_history"]
        recent_shape = state["shape_history"][-k:] if len(state["shape_history"]) > k else state["shape_history"]
        history = describe_history(
            recent_bc,
            recent_shape,
            level=config.OBSERVATION_LEVEL,
        )
        parts.extend([
            "=" * 70,
            "YOUR RECENT CONTROL DECISIONS AND THEIR RESULTS",
            "=" * 70,
            "",
            f"Below are your last {config.K_HISTORY} control commands toward the target.",
            "Each step shows: (1) the control inputs you issued, and (2) the resulting beam shape.",
            "Use this to evaluate: Are your moves progressing toward the target, or stagnating?",
            "",
            history,
            "",
        ])

    # Evaluation result (error %, diagnostics)
    eval_result = state.get("eval_result", {})
    if eval_result:
        error_pct = eval_result.get("error_percent", 100.0)
        diagnostics = eval_result.get("diagnostics", "")
        parts.extend([
            "Evaluation (ground truth):",
            f"  Error: {error_pct:.1f}%",
            f"  Assessment: {diagnostics}",
            "",
        ])

    # Controller feedback force
    feedback = describe_controller_feedback(state.get("reaction_force"))
    if feedback:
        parts.extend([
            feedback,
            "",
        ])

    # Wall contact status
    contact = describe_contact_regions(
        state.get("contact_x"),
        state.get("contact_y"),
        ps.nv,
        ps.walls,
    )
    if contact:
        parts.extend([
            contact,
            "",
        ])

    # ─────────────────────────────────────────────────────────────────────
    # Tier 3: Conditional guidance (every iteration if conditions met)
    # ─────────────────────────────────────────────────────────────────────

    # Prior strategy (if plan mode and strategy was generated)
    if config.ORCHESTRATOR_MODE == "plan" and state.get("plan"):
        parts.extend([
            "="*70,
            "PLANNING AGENT'S STRATEGY (from the planning phase)",
            "="*70,
            "",
            "A planning agent examined the task and developed the following step-by-step strategy.",
            "Your job is to execute this plan by issuing boundary condition moves that follow it.",
            "If the plan is working (beam is progressing toward the target), continue following it.",
            "If progress stalls or the beam diverges from the plan, adapt the next move based on current state.",
            "",
            state["plan"],
            "",
            "="*70,
            "EXECUTION GUIDANCE",
            "="*70,
            "Use the strategy above as your roadmap. At each iteration:",
            "1. Check if your last move followed the strategy → Did the beam progress as planned?",
            "2. Decide the next move → Which step of the strategy should happen next?",
            "3. Adapt if needed → If the beam is not responding as expected, adjust your approach.",
            "",
        ])

    # Previous error (if last BC was rejected)
    error_msg = state.get("error")
    if error_msg:
        error_section = describe_error_rejection(error_msg)
        parts.extend([
            error_section,
            "",
        ])

    # ─────────────────────────────────────────────────────────────────────
    # Tier 4: Decision prompt
    # ─────────────────────────────────────────────────────────────────────

    parts.extend([
        "Next move:",
        "Decide the next boundary condition (x0, y0, θ) for the clamped end.",
        "",
        "Ask yourself:",
        "  1. Is the tip progressing toward the target, or is it stalled?",
        "  2. Is the beam in wall contact? Is that helping or hurting?",
        "  3. Should you continue the current strategy, adapt, or backtrack?",
    ])

    return "\n".join(parts)
