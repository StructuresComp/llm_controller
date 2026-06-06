# ---------------------------------------------------------------------------
# cli.py
# ---------------------------------------------------------------------------
# Entry point. Parses arguments, builds the graph, runs it, prints results.
#
# Usage:
#   python3 cli.py --target C
#   python3 cli.py --target U --H 0.15
# ---------------------------------------------------------------------------

import argparse

from src.core import config
from src.models.physical_system import make_physical_system
from src.models.state_llm_control import initial_controller_state
from src.workflows.workflow import build_controller_graph


def parse_args():
    parser = argparse.ArgumentParser(description="LLM controller for deformable objects.")

    # Goal
    parser.add_argument(
        "--target",
        type=str,
        default="",
        metavar="SHAPE",
        help='Target shape label, e.g. "C", "U", "S".',
    )

    # Environment overrides
    parser.add_argument(
        "--H",
        type=float,
        default=None,
        metavar="DEPTH",
        help="Wall depth override: wall sits at y = -H (m). Overrides WALL_H in config.py.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Build physical system from config, apply any CLI overrides
    ps = make_physical_system(config)
    if args.H is not None:
        ps["wall_H"] = -abs(args.H)   # ensure negative (wall is below origin)

    # Assemble initial state
    state = initial_controller_state(
        physical_system = ps,
        target_shape    = args.target,
    )

    # Run the graph
    # recursion_limit accounts for multiple node invocations per cycle:
    # each cycle = orchestrator sub-graph (planner + decision) + programmer + evaluator = ~5 steps.
    graph  = build_controller_graph()
    result = graph.invoke(state, {"recursion_limit": config.MAX_ITERATIONS * 5})

    # Print summary
    print("\n" + "="*70)
    print("CONTROL LOOP SUMMARY")
    print("="*70)
    print(f"\nTarget shape: {result['target_shape'] or '(none)'}")
    print(f"Converged: {result['converged']}")
    print(f"Iterations: {len(result['iteration_log'])} / {config.MAX_ITERATIONS}")

    # Print iteration-by-iteration summary
    if result["iteration_log"]:
        print("\nIteration log:")
        for entry in result["iteration_log"]:
            iteration = entry["iteration"]
            decision = entry["decision"]
            tip_x = entry["tip_x"]
            tip_y = entry["tip_y"]
            status = " [CONVERGED]" if (iteration == len(result["iteration_log"]) and result["converged"]) else ""
            print(f"  {iteration}. x0={decision['x0']:+.4f}  y0={decision['y0']:+.4f}  θ={decision['theta']:+.4f}  "
                  f"→ tip: ({tip_x:+.6f}, {tip_y:+.6f}){status}")

    n_configs = len(result["all_configurations"])
    print(f"\nConfigurations stored: {n_configs}")

    if result.get("error"):
        print(f"\n[ERROR] {result['error']}")

    print("="*70)


if __name__ == "__main__":
    main()
