# ---------------------------------------------------------------------------
# src/tests/test_beam_chaining.py
# ---------------------------------------------------------------------------
# Verifies that two consecutive discreteBeam calls chain correctly:
#   Run 1 — start from rest, move to control_position_1 / angle_1
#   Run 2 — start from Run 1's final (q, u), move to control_position_2 / angle_2
#
# Run from project root:
#   python -m src.tests.test_beam_chaining
# ---------------------------------------------------------------------------

import numpy as np
from src.core import config
from src.models.physical_system import make_physical_system
from src.tools.Discrete_Elastic_Beams import discreteBeam


def run():
    ps = make_physical_system(config)

    # ------------------------------------------------------------------
    # Run 1: straight down from rest, push node 0 to (0, -0.03)
    # ------------------------------------------------------------------
    cp1, ca1 = (0.0, -0.03), 0.0
    print(f"Controller decision:  position = {cp1},  angle = {ca1:.3f} rad")
    r1 = discreteBeam(
        physical_system  = ps,
        control_position = cp1,
        control_angle    = ca1,
        dt               = config.DT,
        plotImage        = config.PLOT_IMAGE,
    )
    assert r1["status"] == "success", f"Run 1 failed: {r1['message']}"
    print(f"Simulation result:    {r1['message']}  |  tip: x={r1['q_new'][-2]:.6f} m, y={r1['q_new'][-1]:.6f} m")
    print()

    # ------------------------------------------------------------------
    # Run 2: continue from Run 1's final state, tilt to angle 0.4 rad
    # ------------------------------------------------------------------
    cp2, ca2 = (0.0, -0.05), 0.4
    print(f"Controller decision:  position = {cp2},  angle = {ca2:.3f} rad")
    r2 = discreteBeam(
        physical_system  = ps,
        q0               = r1["q_new"],
        u0               = r1["u_new"],
        t_start          = r1["ctime"],
        control_position = cp2,
        control_angle    = ca2,
        dt               = config.DT,
        plotImage        = config.PLOT_IMAGE,
    )
    assert r2["status"] == "success", f"Run 2 failed: {r2['message']}"
    print(f"Simulation result:    {r2['message']}  |  tip: x={r2['q_new'][-2]:.6f} m, y={r2['q_new'][-1]:.6f} m")
    print()

    # ------------------------------------------------------------------
    # Continuity check: Run 2's starting q0 equals Run 1's q_new
    # ------------------------------------------------------------------
    q0_run2 = np.array(r1["q_new"])
    q_start_run2 = np.array(r2["q_new"])  # this is the end, not start — checked via input
    # The real check: pass r1["q_new"] as q0 to Run 2 and confirm it differs from
    # a fresh-start run (i.e., chaining actually changed the result)
    r2_fresh = discreteBeam(
        physical_system  = ps,
        control_position = (0.0, -0.05),
        control_angle    = 0.4,
        dt               = config.DT,
        plotImage        = 0,
    )
    tip_chained = np.array(r2["q_new"][-2:])
    tip_fresh   = np.array(r2_fresh["q_new"][-2:])
    tip_diff    = np.linalg.norm(tip_chained - tip_fresh)

    print(f"\nChaining check:")
    print(f"  Chained tip : x = {tip_chained[0]:.6f},  y = {tip_chained[1]:.6f}")
    print(f"  Fresh tip   : x = {tip_fresh[0]:.6f},  y = {tip_fresh[1]:.6f}")
    print(f"  Tip diff    : {tip_diff:.6f} m  (should be > 0 — runs started from different states)")
    assert tip_diff > 1e-6, "Chaining had no effect — Run 2 looks identical to a fresh start"

    print("\nAll checks passed.")


if __name__ == "__main__":
    run()
