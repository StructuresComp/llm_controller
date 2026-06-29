#!/usr/bin/env python3
"""
Render a finished llm_controller run in interactive 3D (mechview).

Reads a run JSON from logs/, converts it to mechview format, and drops
index.html + data.js into viz/ (VISUALIZATION_DIR) ready to open in Chrome.

Usage:
    python3 render_3D.py                      # latest run in logs/
    python3 render_3D.py logs/run_123.json    # specific run
"""

import importlib.resources
import json
import shutil
import sys
from pathlib import Path

from src.core.paths import LOGS_DIR, VISUALIZATION_DIR
from src.visualization.mechview_export import convert


def latest_run(logs_dir: Path) -> Path | None:
    runs = sorted(logs_dir.glob("*.jsonl"))
    return runs[-1] if runs else None


def load_run(path: Path) -> dict:
    """Support both legacy .json and new .jsonl formats."""
    if path.suffix == ".jsonl":
        from src.core import config
        from src.tools.walls import make_wall_2d
        walls = [make_wall_2d(w) for w in getattr(config, "WALLS", [])]
        return {
            "physics_type":    getattr(config, "PHYSICS_TYPE", "beam"),
            "nv":              config.NV,
            "rod_length":      config.ROD_LENGTH,
            "r0":              config.R0,
            "walls":           walls,
            "trajectory_file": path.stem,
        }
    return json.loads(path.read_text())


def main():
    if len(sys.argv) > 1:
        raw_path = Path(sys.argv[1])
        if not raw_path.exists():
            print(f"Error: {raw_path} not found.")
            sys.exit(1)
    else:
        raw_path = latest_run(LOGS_DIR)
        if raw_path is None:
            print("No run JSONL files found in logs/.")
            print("Run the controller first:  python3 cli.py --target <shape>")
            sys.exit(0)
        print(f"Using latest run: {raw_path.name}")

    VISUALIZATION_DIR.mkdir(exist_ok=True)

    raw = load_run(raw_path)
    convert(raw, VISUALIZATION_DIR / "data.js")

    src = importlib.resources.files("mechview") / "index.html"
    with importlib.resources.as_file(src) as p:
        shutil.copy(p, VISUALIZATION_DIR / "index.html")

    print(f"Open in Chrome: {(VISUALIZATION_DIR / 'index.html').resolve()}")


if __name__ == "__main__":
    main()
