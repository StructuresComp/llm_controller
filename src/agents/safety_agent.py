# ---------------------------------------------------------------------------
# src/agents/safety_agent.py
# ---------------------------------------------------------------------------
# BC safety check — sits between orchestrator and programmer.
#
# Checks whether the proposed node-0 position (x0, y0) from state["decision"]
# would place the controlled end within TOLERANCE_DISTANCE of any wall.
# On violation: sets state["error"] with a descriptive message and routes
# back to the orchestrator. On pass: clears state["error"] and routes to
# the programmer.
#
# No LLM calls — entirely deterministic geometry.
#
# Section order:
#   1. Imports  2. Schemas  3. Prompts  4. Helpers  5. Nodes  7. Main  8. Entry
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 1. IMPORTS
# ---------------------------------------------------------------------------
import math

from src.core import config
from src.models.state_llm_control import ControllerState
from src.tools.walls import signed_dist_2d, within_extent_2d


# ---------------------------------------------------------------------------
# 2. SCHEMAS
# ---------------------------------------------------------------------------
# No schemas needed — no LLM calls.


# ---------------------------------------------------------------------------
# 3. PROMPTS
# ---------------------------------------------------------------------------
# No LLM prompts needed.


# ---------------------------------------------------------------------------
# 4. HELPERS
# ---------------------------------------------------------------------------

def check_bc_wall_safety(
    px: float,
    py: float,
    walls: list,
    tolerance: float,
) -> list[str]:
    """
    Return a list of human-readable violation strings (empty = safe).

    A violation occurs when both:
      1. The node projects onto the wall segment (within_extent_2d).
      2. signed_dist_2d < tolerance  (too close or already penetrated).

    Checking extent first means we never flag a node that is "behind" the
    infinite plane extension of a wall but past its endpoints (corner region).
    Checking dist < tolerance (not just dist < 0) catches near-misses before
    the node has actually crossed — which is the whole point of the buffer.
    """
    violations = []
    for wall in walls:
        if not within_extent_2d(px, py, wall):
            continue
        dist = signed_dist_2d(px, py, wall)
        if dist >= tolerance:
            continue
        n, d = wall["n"], wall["d"]
        if abs(n[1]) >= abs(n[0]):
            wall_desc = f"horizontal wall at y={d / n[1]:.4f} m"
            safe_coord = (d + tolerance) / n[1]
            safe_hint  = f"; y must be {'>=' if n[1] > 0 else '<='} {safe_coord:.4f} m"
        else:
            wall_desc = f"vertical wall at x={d / n[0]:.4f} m"
            safe_coord = (d + tolerance) / n[0]
            safe_hint  = f"; x must be {'>=' if n[0] > 0 else '<='} {safe_coord:.4f} m"
        violations.append(
            f"{wall_desc} (signed distance = {dist:+.4f} m, required >= {tolerance:.4f} m{safe_hint})"
        )
    return violations


# ---------------------------------------------------------------------------
# 5. NODES
# ---------------------------------------------------------------------------

def safety_node(state: ControllerState) -> dict:
    """
    Validate the proposed BC against wall proximity constraints.

    Checks both arm nodes (0 and 1). Node 1's target position is derived from
    the commanded (x0, y0, theta): node1 = (x0 + deltaL*sin(theta), y0 - deltaL*cos(theta)).

    Returns {"error": None} on pass, {"error": <message>} on violation.
    The workflow routes violations back to the orchestrator.
    """
    decision = state["decision"]
    ps       = state["physical_system"]
    x0, y0   = decision["x0"], decision["y0"]
    theta    = decision["theta"]

    tol = getattr(config, "TOLERANCE_DISTANCE", None)
    if tol is None:
        import warnings
        warnings.warn(
            "TOLERANCE_DISTANCE is not set in config — wall safety check running with zero clearance. "
            "Set TOLERANCE_DISTANCE = 0.1 * ROD_LENGTH in config.py.",
            stacklevel=2,
        )
        tol = 0.0

    if not ps.walls:
        return {"error": None}

    x1 = x0 + ps.deltaL * math.sin(theta)
    y1 = y0 - ps.deltaL * math.cos(theta)

    v0 = [f"[node 0 at ({x0:+.4f}, {y0:+.4f})] {v}"
          for v in check_bc_wall_safety(x0, y0, ps.walls, tol)]
    v1 = [f"[node 1 at ({x1:+.4f}, {y1:+.4f})] {v}"
          for v in check_bc_wall_safety(x1, y1, ps.walls, tol)]
    violations = v0 + v1

    if not violations:
        return {"error": None}

    iter_num = len(state.get("iteration_log", [])) + 1
    print(f"\n[Safety] BC REJECTED  (before iteration {iter_num})")
    print(f"         x0={x0:+.6f} m  y0={y0:+.6f} m  theta={theta:+.6f} rad")
    for v in violations:
        print(f"         • {v}")
    print("         Routing back to orchestrator.\n")

    msg = (
        f"Safety: proposed BC (x0={x0:+.4f}, y0={y0:+.4f}, theta={theta:+.4f} rad) "
        f"places a controlled node too close to a wall. "
        f"{'; '.join(violations)}. "
        f"Keep nodes 0 and 1 at least {tol:.4f} m from all walls and reissue the BC."
    )
    return {"error": msg, "safety_rejections": state.get("safety_rejections", 0) + 1}


# ---------------------------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------------------------

def main():
    """Standalone test with a few hand-crafted BC proposals."""
    import types
    from src.models.physical_system import make_physical_system

    # Build a physical system with the snake-L walls
    cfg = types.SimpleNamespace(**{
        k: getattr(config, k) for k in dir(config) if not k.startswith("_")
    })
    cfg.WALLS = [
        {"P0": [-0.01,  0.02], "P1": [-0.01, -0.12]},
        {"P0": [-0.01, -0.12], "P1": [ 0.12, -0.12]},
        {"P0": [ 0.12, -0.12], "P1": [ 0.12, -0.10]},
        {"P0": [ 0.12, -0.10], "P1": [ 0.01, -0.10]},
        {"P0": [ 0.01, -0.10], "P1": [ 0.01,  0.02]},
    ]
    ps  = make_physical_system(cfg)
    tol = ps.rod_length * 1e-3

    tests = [
        (( 0.00, -0.05), "safe — centre of vertical channel"),
        ((-0.009, -0.05), "close to left wall (should trigger)"),
        (( 0.00, -0.121), "just above floor (should trigger)"),
        (( 0.05, -0.05), "outside channel — no extent match"),
    ]

    print(f"Tolerance: {tol:.4f} m\n")
    for (x0, y0), label in tests:
        v = check_bc_wall_safety(x0, y0, ps.walls, tol)
        status = "SAFE" if not v else f"VIOLATION: {'; '.join(v)}"
        print(f"  ({x0:+.4f}, {y0:+.4f})  [{label}]\n    → {status}\n")


# ---------------------------------------------------------------------------
# 8. ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
