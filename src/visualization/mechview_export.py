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
WALL_WIDTH        = 2.0   # obstacle box x-extent
WALL_DEPTH        = 2.0   # obstacle box z-extent
WALL_THICKNESS    = 0.04  # obstacle box y-extent
TUBE_RADIUS_SCALE = 0.5   # r0/rod_length multiplied by this for the visual tube


def _nodes_from_q(q: list, nv: int, physics_type: str) -> list:
    if physics_type == "beam":
        return [[q[2*i], q[2*i + 1], 0.0] for i in range(nv)]
    elif physics_type == "rod":
        return [[q[3*i], q[3*i + 1], q[3*i + 2]] for i in range(nv)]
    else:
        raise ValueError(f"Unknown physics_type: {physics_type!r}")


def convert(raw: dict, output_path: Path) -> None:
    physics_type = raw["physics_type"]
    nv           = raw["nv"]
    rod_length   = float(raw["rod_length"])
    r0           = float(raw.get("r0", 0.003))
    walls_raw    = raw.get("walls", [])        # list of {n, d, P0, P1} dicts; empty = no wall
    frame_labels = raw.get("frame_labels", {}) # {str(frame_idx): label} — may be absent

    scale = 1.0 / rod_length

    # Frames — normalise coordinates, attach per-frame label if present
    frames = []
    for idx in sorted(raw["all_configurations"].keys(), key=int):
        q_raw   = raw["all_configurations"][idx]
        nodes   = _nodes_from_q(q_raw, nv, physics_type)
        nodes_n = [[x * scale, y * scale, z * scale] for x, y, z in nodes]
        frame   = {"t": int(idx), "nodes": nodes_n}

        # frame_labels keys are stored as strings in JSON
        label = frame_labels.get(str(idx)) or frame_labels.get(idx)
        if label:
            frame["label"] = label

        frames.append(frame)

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

    data = {
        "object_type": physics_type,   # "beam" or "rod" — mechview renders both as a tube
        "nv":          nv,
        "rod_radius":  r0 * scale * TUBE_RADIUS_SCALE,
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
    args = p.parse_args()

    raw_path = Path(args.input)
    if not raw_path.exists():
        print(f"Error: {raw_path} not found.", file=sys.stderr)
        sys.exit(1)

    raw = json.loads(raw_path.read_text())
    convert(raw, Path(args.out))


if __name__ == "__main__":
    main()
