# ---------------------------------------------------------------------------
# cli.py
# ---------------------------------------------------------------------------
# Entry point for the LLM controller framework.
#
# Responsibilities (in order):
#   1. Parse CLI arguments
#   2. Apply optional config overlay file (must happen before any src imports
#      so that llm.py lazy-init sees the correct provider/model/RUN_NAME)
#   3. Validate --mode (beam vs. rod)
#   4. Resolve wall geometry: benchmark walls → --H override → make_physical_system
#   5. Build the LangGraph control loop and run it
#   6. Save the run JSON and print a summary
#
# Usage:
#   python3 cli.py --target C
#   python3 cli.py --target C --H 0.15
#   python3 cli.py --target tip_reach --length 0.12
#   python3 cli.py --target snake_L --config experiments/my_run.py
#   python3 cli.py --target C --mode beam
#   python3 cli.py --target C --provider ollama --model qwen2.5:14b --run-name qwen_C
# ---------------------------------------------------------------------------

import argparse
import importlib.util
import sys

from src.core import config


# ---------------------------------------------------------------------------
# ARGUMENT PARSING
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="LLM controller for deformable objects.")

    parser.add_argument(
        "--target",
        type=str,
        default="",
        metavar="SHAPE",
        help="Target benchmark name (any registered case in src/benchmarks/cases/).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        metavar="MODE",
        help='Physics type: "beam" (2D elastic beam) or "rod" (3D elastic rod). Overrides PHYSICS_TYPE in config.',
    )
    parser.add_argument(
        "--H",
        type=float,
        default=None,
        metavar="DEPTH",
        help=(
            "Add a single horizontal floor wall at y = -|H| m "
            "(e.g. --H 0.15 → floor at y = -0.15 m). "
            "Sign is ignored — wall is always placed below the origin. "
            "Overrides WALLS in config."
        ),
    )
    parser.add_argument(
        "--length",
        type=float,
        default=None,
        metavar="L",
        help="Override ROD_LENGTH (m).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="NAME",
        help="Override config.MODEL (e.g. 'qwen2.5:14b', 'gpt-5.4', 'claude-haiku-4-5').",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        metavar="NAME",
        help="Override config.LLM_PROVIDER (e.g. 'anthropic', 'openai', 'ollama').",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        metavar="NAME",
        help="Override config.RUN_NAME — output file stem: logs/<NAME>.jsonl + logs/<NAME>.log.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="FILE",
        help=(
            "Path to a Python config overlay file. "
            "Any attribute defined there overrides the matching setting in config.py."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print LLM prompts and responses to terminal for debugging.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# CONFIG OVERLAY
# ---------------------------------------------------------------------------

def _apply_config_overlay(path: str) -> None:
    """
    Load a user-supplied Python file and copy every non-dunder attribute onto
    the live config module.  Called before any other src imports so that
    provider, model, and RUN_NAME are in place before llm.py initialises.
    """
    spec = importlib.util.spec_from_file_location("_user_config", path)
    user_cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(user_cfg)
    for key, value in vars(user_cfg).items():
        if not key.startswith("_"):
            setattr(config, key, value)


# ---------------------------------------------------------------------------
# STARTUP SUMMARY
# ---------------------------------------------------------------------------

def _print_startup_summary(args, ps, cfg) -> None:
    """
    Print a one-screen summary of what the controller is about to do.

    Shows: target, model, orchestrator mode, wall geometry, which nodes the
    arm drives vs. which are clamped, rod length, and safety check status.

    arm_nodes  — always [0, 1]; these are moved by the LLM command each cycle.
    fixed_nodes — tip nodes clamped only when there are no walls; without this
                  clamp, moving node 0 would just translate the whole beam under
                  gravity instead of bending it.
    """
    sep = "=" * 65
    tol = getattr(cfg, "TOLERANCE_DISTANCE", None) or ps.rod_length * 1e-3

    print(sep)
    print(f"LLM CONTROLLER  target={args.target or '(none)'}  provider={cfg.LLM_PROVIDER}  model={cfg.MODEL}  mode={cfg.ORCHESTRATOR_MODE}")
    print(sep)

    if ps.walls:
        h_note = f"  [--H {abs(args.H)} -> y = -{abs(args.H):.4f} m]" if args.H is not None else ""
        print(f"Walls        : {len(ps.walls)} segment(s){h_note}")
    else:
        print("Walls        : none")

    print("Control topology:")
    print("  Clamped end (node 0): YOU control position (x0, y0) and slope (theta) each iteration")
    if ps.walls:
        print("  Free end: Can deform freely; walls prevent penetration")
    else:
        print("  Tip region (nodes nv-2, nv-1): Fixed in space — prevents unconstrained translation under gravity")

    print(f"Rod length   : {ps.rod_length:.4f} m")

    if ps.walls:
        print(f"Safety check : clamped end checked against walls; tolerance {tol:.4f} m")
    else:
        print("Safety check : disabled (no walls)")

    print(sep)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Step 1 — config overlay must come first; llm.py opens its log file on the
    # first LLM call and reads RUN_NAME/provider/model from config at that point.
    if args.config is not None:
        _apply_config_overlay(args.config)

    # Apply --debug flag to config
    if args.debug:
        config.DEBUG = True

    # CLI overrides for LLM provider/model/run-name — applied after any --config
    # overlay so these flags always win. Meant for parallel sweep scripts that
    # vary model/provider/run_name per subprocess without generating overlay files.
    if args.model is not None:
        config.MODEL = args.model
    if args.provider is not None:
        config.LLM_PROVIDER = args.provider
    if args.run_name is not None:
        config.RUN_NAME = args.run_name

    # Downstream imports are deferred to here so the overlay is in effect before
    # any src module reads from config at import time.
    from src.core.config_writer import ConfigWriter
    from src.benchmarks import get_benchmark
    from src.models.physical_system import make_physical_system
    from src.models.state_llm_control import initial_controller_state
    from src.workflows.workflow import build_controller_graph

    # Step 2 — physics mode.  "beams"/"rods" accepted as aliases.
    if args.mode is not None:
        normalised = args.mode.lower().rstrip("s")   # "beams" → "beam", "rods" → "rod"
        if normalised == "beam":
            config.PHYSICS_TYPE = "beam"
        elif normalised == "rod":
            print("Rod support is under development. Exiting.")
            sys.exit(0)
        else:
            print(f"Unknown --mode '{args.mode}'. Valid options: beam, rod.")
            sys.exit(1)

    # Step 3 — scalar overrides that feed into make_physical_system.
    if args.length is not None:
        config.ROD_LENGTH = args.length

    # Step 4 — wall geometry resolution (order matters):
    #   a. Benchmark-defined walls (e.g. snake_L carries its own L-tube geometry).
    #   b. --H overrides everything with a single horizontal floor at y = -|H|.
    # Both must land on config.WALLS *before* make_physical_system so that
    # controlled_nodes is derived correctly (walls present → [0,1] only;
    # no walls → [0,1,nv-2,nv-1]).
    bm = get_benchmark(args.target) if args.target else None
    if bm is not None and bm.walls:
        print(f"Adding {len(bm.walls)} wall(s) specific to case '{args.target}' on top of config.WALLS.")
        config.WALLS = config.WALLS + bm.walls

    if args.H is not None:
        H = abs(args.H)
        config.WALLS = [{"P0": [-100., -H], "P1": [100., -H]}]

    # Step 5 — build the physical system and print what we're about to run.
    ps = make_physical_system(config)
    _print_startup_summary(args, ps, config)

    # Step 6 — initialize trajectory disk writer and build initial state.
    # target_tip: generic BenchmarkCase field, set for any loss_type="tip" case
    # (tip_reach, snake_L); None for curvature-type cases -- not tip_reach-specific.
    config_writer = ConfigWriter(run_name=config.RUN_NAME)
    config_writer.write_header(ps, args.target, target_tip=getattr(bm, "reference_tip_fn", None))
    state  = initial_controller_state(physical_system=ps, target_shape=args.target, config_writer=config_writer)

    # Step 6b — pre-flight check: fail cleanly if the initial (straight-hanging) q0
    # already penetrates a wall, before any LLM call or graph tick happens. This can
    # happen when a case's wall geometry doesn't fit the configured ROD_LENGTH.
    from src.tools.walls import check_q_against_walls
    violations = check_q_against_walls(state["q0_current"], ps.nv, ps.walls, ps.controlled_nodes)
    if violations:
        print(f"\n[SETUP ERROR] Initial configuration already penetrates {len(violations)} "
              f"wall(s) before any control step -- this target is not runnable as configured:")
        for v in violations:
            print(f"  Node {v['node']} at ({v['x']:.4f}, {v['y']:.4f}) penetrates wall "
                  f"(n={v['wall']['n']}, d={v['wall']['d']:.4f})")
        print("Likely cause: this case's wall geometry doesn't fit ROD_LENGTH. "
              "Try a shorter --length, or adjust the case's walls.")
        config_writer.close()
        sys.exit(1)

    # Step 7 — run the control loop.
    graph  = build_controller_graph()
    try:
        result = graph.invoke(state, {"recursion_limit": 10_000})
    finally:
        config_writer.close()

    # Step 8a — print orchestrator plan (if in plan mode)
    if config.ORCHESTRATOR_MODE == "plan" and result.get("plan"):
        print("\n" + "="*70)
        print("ORCHESTRATOR PLAN (from LLM planning phase)")
        print("="*70)
        print(result["plan"])
        print("="*70)

    # Step 8b — print end-of-run summary.
    print("\n" + "="*70)
    print("CONTROL LOOP SUMMARY")
    print("="*70)
    solved        = result.get("eval_result", {}).get("solved", False)
    final_err     = result.get("eval_result", {}).get("error_percent", None)
    iterations_done = len(result["iteration_log"])
    attempts_used = result.get("attempt_count", 0)
    print(f"\nTarget     : {result['target_shape'] or '(none)'}")
    print(f"Solved     : {solved}" + (f"  (error={final_err:.2f}%)" if final_err is not None else ""))
    print(f"Iterations : {iterations_done} / {config.MAX_ITERATIONS}")
    print(f"Attempts   : {attempts_used} / {config.MAX_ITERATIONS_WITH_REJECTIONS}  "
          f"(includes retried/rejected/extraction-fallback decisions, not just completed iterations)")

    # If we stopped short of MAX_ITERATIONS without solving or hitting a fatal error,
    # the only remaining termination path is _after_safety's attempt-cap check in
    # workflow.py -- surface that clearly instead of leaving it to be inferred.
    if not solved and not result.get("error") and iterations_done < config.MAX_ITERATIONS \
            and attempts_used >= config.MAX_ITERATIONS_WITH_REJECTIONS:
        print(
            f"\n[TERMINATED] Hit MAX_ITERATIONS_WITH_REJECTIONS: {attempts_used} total LLM "
            f"attempts (successful + retried + rejected) reached the cap of "
            f"{config.MAX_ITERATIONS_WITH_REJECTIONS} after only "
            f"{iterations_done}/{config.MAX_ITERATIONS} real iterations completed. "
            f"Raise MAX_ITERATIONS_WITH_REJECTIONS in config.py to allow more retry/rejection "
            f"overhead per run."
        )

    if result["iteration_log"]:
        print("\nIteration log:")
        for entry in result["iteration_log"]:
            iteration = entry["iteration"]
            decision  = entry["decision"]
            tip_x     = entry["tip_x"]
            tip_y     = entry["tip_y"]
            status = " [CONVERGED]" if (iteration == len(result["iteration_log"]) and result["converged"]) else ""
            print(f"  {iteration:3d}.  x0={decision['x0']:+.4f}  y0={decision['y0']:+.4f}  "
                  f"theta={decision['theta']:+.4f}  -> tip: ({tip_x:+.6f}, {tip_y:+.6f}){status}")

    print(f"\nTrajectory snapshots   : {config_writer.snapshot_count}  (streamed to {config_writer.filepath.name})")

    if result.get("error"):
        print(f"\n[ERROR] {result['error']}")

    print("="*70)

    if result.get("error"):
        sys.exit(1)


if __name__ == "__main__":
    main()
