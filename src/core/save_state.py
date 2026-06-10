"""
Save a completed control run to a raw JSON file.

Called from cli.py after graph.invoke() returns.
Stores everything in physical units — no normalisation.
The viewer's export.py adapter reads this file and produces data.js.

─────────────────────────────────────────────────────────────
RAW JSON SCHEMA  (contract between llm_controller and viewer)
─────────────────────────────────────────────────────────────
{
  "physics_type":       "beam" | "rod",
  "nv":                 int,            # nodes per frame
  "rod_length":         float,          # metres — used for normalisation in export.py
  "r0":                 float,          # cross-section radius (metres)
  "walls": [                             # list of wall dicts; empty = no wall
    {"n": [nx, ny], "d": float, "P0": [x, y], "P1": [x, y]}
  ],
  "target_shape":       str,            # e.g. "C", "U", "S"
  "all_configurations": {               # physics frames
      "0":  [float, ...],               # q vector, length 2*nv (beam) or 3*nv (rod)
      "1":  [float, ...],
      ...
  },
  "frame_labels": {                     # viewer overlay text — same keys as all_configurations
      "42": "Step 1  x=+0.0100  y=−0.0300  θ=+0.500 rad",
      ...                               # only the last frame of each LLM decision has an entry
  },
  "bc_history":    [[x0, y0, theta], ...],   # last K boundary conditions issued
  "iteration_log": [                         # one entry per LLM decision
      {
          "iteration": int,
          "decision":  {"x0": float, "y0": float, "theta": float},
          "tip_x":     float,
          "tip_y":     float
      },
      ...
  ]
}
─────────────────────────────────────────────────────────────
Breaking changes to this schema require updates in both repos.
─────────────────────────────────────────────────────────────
"""

import json
import os
import time
from pathlib import Path

from src.core import config
from src.core.paths import LOGS_DIR


def save_run(result: dict, ps) -> Path:
    """
    Persist a finished control run to LOGS_DIR/run_<timestamp>.json.

    Args:
        result  — final ControllerState dict returned by graph.invoke()
        ps      — BeamPhysicalSystem or RodPhysicalSystem

    Returns the path where the file was written.
    """
    payload = {
        "physics_type":       ps.type,
        "nv":                 ps.nv,
        "rod_length":         float(ps.rod_length),
        "r0":                 float(getattr(ps, "r0", 0.003)),
        "walls":              [
                                  {"n": w["n"].tolist(), "d": float(w["d"]),
                                   "P0": w["P0"].tolist(), "P1": w["P1"].tolist()}
                                  for w in getattr(ps, "walls", [])
                              ],
        "target_shape":       result.get("target_shape", ""),
        "all_configurations": result.get("all_configurations", {}),
        "frame_labels":       result.get("frame_labels", {}),
        "bc_history":         result.get("bc_history", []),
        "iteration_log":      result.get("iteration_log", []),
    }

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    stem = getattr(config, "RUN_NAME", "") or f"run_{int(time.time())}_{os.getpid()}"
    out = LOGS_DIR / f"{stem}.json"
    out.write_text(json.dumps(payload))
    print(f"Run saved → {out}")
    return out
