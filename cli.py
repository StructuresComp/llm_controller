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
        help="Target benchmark name: C, U, S, L, snake_L, tip_reach, or any registered case.",
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
        "--config",
        type=str,
        default=None,
        metavar="FILE",
        help=(
            "Path to a Python config overlay file. "
            "Any attribute defined there overrides the matching setting in config.py."
        ),
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
    from src.core.observation import describe_walls
    sep = "─" * 65
    tol = getattr(cfg, "TOLERANCE_DISTANCE", None) or ps.rod_length * 1e-3

    print(sep)
    print(f"LLM CONTROLLER  target={args.target or '(none)'}  model={cfg.MODEL}  mode={cfg.ORCHESTRATOR_MODE}")
    print(sep)

    if ps.walls:
        h_note = f"  [--H {abs(args.H)} → y = -{abs(args.H):.4f} m]" if args.H is not None else ""
        print(f"Walls        : {len(ps.walls)} segment(s) — {describe_walls(ps.walls)}{h_note}")
    else:
        print("Walls        : none")

    arm_nodes   = [0, 1]
    fixed_nodes = [n for n in ps.controlled_nodes if n not in arm_nodes]
    print(f"Arm nodes    : {arm_nodes}  (position + angle commanded each iteration)")
    if fixed_nodes:
        print(f"Fixed nodes  : {fixed_nodes}  (tip clamped — no walls, prevents rigid-body translation)")
    else:
        print("Fixed nodes  : none  (tip is free; walls constrain the beam)")

    print(f"Rod length   : {ps.rod_length:.4f} m")

    if ps.walls:
        print(f"Safety check : nodes {arm_nodes} checked against walls; tolerance {tol:.4f} m")
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

    # Downstream imports are deferred to here so the overlay is in effect before
    # any src module reads from config at import time.
    from src.core.save_state import save_run
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
    if args.target:
        bm = get_benchmark(args.target)
        if bm.walls is not None:
            config.WALLS = bm.walls

    if args.H is not None:
        H = abs(args.H)
        config.WALLS = [{"P0": [-100., -H], "P1": [100., -H]}]

    # Step 5 — build the physical system and print what we're about to run.
    ps = make_physical_system(config)
    _print_startup_summary(args, ps, config)

    # Step 6 — assemble the LangGraph state and run the control loop.
    state  = initial_controller_state(physical_system=ps, target_shape=args.target)
    graph  = build_controller_graph()
    result = graph.invoke(state, {"recursion_limit": 10_000})

    save_run(result, ps)

    # Step 7 — print end-of-run summary.
    print("\n" + "="*70)
    print("CONTROL LOOP SUMMARY")
    print("="*70)
    solved     = result.get("eval_result", {}).get("solved", False)
    final_err  = result.get("eval_result", {}).get("error_percent", None)
    print(f"\nTarget     : {result['target_shape'] or '(none)'}")
    print(f"Solved     : {solved}" + (f"  (error={final_err:.2f}%)" if final_err is not None else ""))
    print(f"LLM judged : {result['converged']}  (recorded for analysis; does not control termination)")
    print(f"Iterations : {len(result['iteration_log'])} / {config.MAX_ITERATIONS}")

    if result["iteration_log"]:
        print("\nIteration log:")
        for entry in result["iteration_log"]:
            iteration = entry["iteration"]
            decision  = entry["decision"]
            tip_x     = entry["tip_x"]
            tip_y     = entry["tip_y"]
            status = " [CONVERGED]" if (iteration == len(result["iteration_log"]) and result["converged"]) else ""
            print(f"  {iteration:3d}.  x0={decision['x0']:+.4f}  y0={decision['y0']:+.4f}  "
                  f"θ={decision['theta']:+.4f}  → tip: ({tip_x:+.6f}, {tip_y:+.6f}){status}")

    print(f"\nConfigurations stored : {len(result['all_configurations'])}")

    if result.get("error"):
        print(f"\n[ERROR] {result['error']}")

    print("="*70)


if __name__ == "__main__":
    main()
