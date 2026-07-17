"""
Export a finished llm_controller run to mechview-compatible data.js.

mechview: https://github.com/StructuresComp/mechview

Reads the raw run JSON written by src/core/save_state.py and converts it to
the data.js format that mechview's index.html reads directly in Chrome.
All coordinates are normalised by rod_length so the camera setup is
scale-independent.

The raw JSON schema is documented in src/core/save_state.py — that docstring
is the authoritative contract between the two repos.

Usage:
    python mechview_export.py logs/run_<timestamp>.json
    python mechview_export.py logs/run_<timestamp>.json --out path/to/data.js
"""

import argparse
import json
import sys
from pathlib import Path


# Visual constants — all in normalised units (rod_length = 1).
# generate_synthetic.py imports these so wall appearance stays consistent.
WALL_WIDTH         = 2.0   # obstacle box x-extent
WALL_DEPTH         = 2.0   # obstacle box z-extent
WALL_THICKNESS     = 0.04  # obstacle box y-extent
TUBE_RADIUS_SCALE  = 0.5   # r0/rod_length multiplied by this for the visual tube
TARGET_MARKER_SIZE = 0.05  # small cube marking a tip-type case's target (x, y); cube edge length


def _nodes_from_q(q: list, nv: int, physics_type: str) -> list:
    if physics_type == "beam":
        return [[q[2*i], q[2*i + 1], 0.0] for i in range(nv)]
    elif physics_type == "rod":
        return [[q[3*i], q[3*i + 1], q[3*i + 2]] for i in range(nv)]
    else:
        raise ValueError(f"Unknown physics_type: {physics_type!r}")


def convert(raw: dict, output_path: Path, frame_stride: int = 1, target_tip=None) -> None:
    physics_type = raw["physics_type"]
    nv           = raw["nv"]
    rod_length   = float(raw["rod_length"])
    r0           = float(raw.get("r0", 0.003))
    walls_raw    = raw.get("walls", [])        # list of {n, d, P0, P1} dicts; empty = no wall
    frame_stride = max(frame_stride, 1)        # keep every Nth frame; final frame always kept

    scale = 1.0 / rod_length

    # Frames — load from JSONL file (streaming trajectory)
    # If raw["all_configurations"] exists (old format), use that for backward compatibility
    frames = []
    if "all_configurations" in raw:
        # Fallback for old-format JSON files
        frame_labels = raw.get("frame_labels", {})
        for idx in sorted(raw["all_configurations"].keys(), key=int):
            q_raw   = raw["all_configurations"][idx]
            nodes   = _nodes_from_q(q_raw, nv, physics_type)
            nodes_n = [[x * scale, y * scale, z * scale] for x, y, z in nodes]
            frame   = {"t": int(idx), "nodes": nodes_n}

            label = frame_labels.get(str(idx)) or frame_labels.get(idx)
            if label:
                frame["label"] = label

            frames.append(frame)
    else:
        # New format: read from JSONL trajectory file
        trajectory_file = raw.get("trajectory_file", "")
        if trajectory_file:
            from src.core.paths import LOGS_DIR
            jsonl_path = LOGS_DIR / f"{trajectory_file}.jsonl"
            if jsonl_path.exists():
                snapshots = []
                with open(jsonl_path, "r") as f:
                    for line in f:
                        if line.strip():
                            snap = json.loads(line)
                            if snap.get("type") == "header":
                                continue
                            snapshots.append(snap)

                last_idx = len(snapshots) - 1
                frame_idx = 0
                for i, snap in enumerate(snapshots):
                    if i % frame_stride != 0 and i != last_idx:
                        continue  # thinned out — frame_stride keeps every Nth, plus the final one

                    q_raw = snap["q"]
                    nodes = _nodes_from_q(q_raw, nv, physics_type)
                    nodes_n = [[x * scale, y * scale, z * scale] for x, y, z in nodes]
                    frame = {"t": frame_idx, "nodes": nodes_n}

                    # Optionally annotate with BC command
                    bc = snap.get("bc", {})
                    error = snap.get("error_percent", 0)
                    frame["label"] = (
                        f"Iter {snap.get('iteration')}  "
                        f"x={bc.get('x0', 0):+.4f}  y={bc.get('y0', 0):+.4f}  "
                        f"θ={bc.get('theta', 0):+.3f}  error={error:.1f}%"
                    )

                    frames.append(frame)
                    frame_idx += 1
            else:
                print(f"Warning: trajectory file {jsonl_path} not found", file=sys.stderr)

    # Obstacles — one box per wall; top surface placed at the physics contact plane.
    # P0/P1 are always present; large sentinel coordinates (used for "infinite" walls)
    # are capped to WALL_WIDTH so the visual box stays a reasonable size.
    obstacles = []
    for w in walls_raw:
        n, d = w["n"], w["d"]
        P0, P1 = w["P0"], w["P1"]
        if abs(n[1]) >= abs(n[0]):                         # horizontal wall
            wall_y   = (d / n[1]) * scale
            center_y = wall_y - WALL_THICKNESS / 2
            cx       = (P0[0] + P1[0]) / 2 * scale
            width    = min(abs(P1[0] - P0[0]) * scale, WALL_WIDTH)
            obstacles.append({
                "center": [cx, center_y, 0.0],
                "size":   [width, WALL_THICKNESS, WALL_DEPTH],
            })
        else:                                              # vertical wall
            wall_x   = (d / n[0]) * scale
            center_x = wall_x
            cy       = (P0[1] + P1[1]) / 2 * scale
            height   = min(abs(P1[1] - P0[1]) * scale, WALL_WIDTH)
            obstacles.append({
                "center": [center_x, cy, 0.0],
                "size":   [WALL_THICKNESS, height, WALL_DEPTH],
            })

    rod_radius_visual = r0 * scale * TUBE_RADIUS_SCALE

    # Target marker — only for tip-type cases (tip_reach, snake_L); target_tip is
    # None for curvature-type cases (C/S/U/L/arc/sine), which have no single target
    # point, so nothing is added and the render is unaffected. Z is offset slightly
    # positive (2x the visual rod radius) so the marker doesn't get visually
    # absorbed inside the beam tube, which is rendered at z=0.
    if target_tip is not None:
        tx, ty = target_tip
        obstacles.append({
            "center": [tx * scale, ty * scale, 2 * rod_radius_visual],
            "size":   [TARGET_MARKER_SIZE, TARGET_MARKER_SIZE, TARGET_MARKER_SIZE],
        })

    data = {
        "object_type": physics_type,   # "beam" or "rod" — mechview renders both as a tube
        "nv":          nv,
        "rod_radius":  rod_radius_visual,
        "obstacles":   obstacles,
        "frames":      frames,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("const MECHVIEW_DATA = " + json.dumps(data) + ";")
    print(f"Exported {len(frames)} frames ({physics_type}, nv={nv}) → {output_path}")


def main():
    p = argparse.ArgumentParser(description="Convert raw run JSON to mechview data.js.")
    p.add_argument("input", help="path to raw run JSON (e.g. logs/run_1234567890.json)")
    p.add_argument("--out", default="data.js", help="output path (default: ./data.js)")
    p.add_argument("--frame-stride", type=int, default=1, metavar="N",
                    help="keep every Nth frame (default: 1 = every frame). Final frame always kept.")
    args = p.parse_args()

    raw_path = Path(args.input)
    if not raw_path.exists():
        print(f"Error: {raw_path} not found.", file=sys.stderr)
        sys.exit(1)

    raw = json.loads(raw_path.read_text())
    convert(raw, Path(args.out), frame_stride=args.frame_stride)


if __name__ == "__main__":
    main()
