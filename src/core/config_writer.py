# ---------------------------------------------------------------------------
# src/core/config_writer.py
# ---------------------------------------------------------------------------
# Stream trajectory snapshots to a single JSONL file during a run.
#
# Each line is one DOF snapshot with metadata:
#   {"iteration": N, "bc": {...}, "q": [...], "error_percent": X, "timestamp": T}
#
# File is flushed to disk after every write — memory usage is O(1).
# Post-run, analysis_plots.py reads this file to reconstruct and plot trajectories.
# ---------------------------------------------------------------------------

import json
import time
import os
from pathlib import Path

from src.core.paths import LOGS_DIR


class ConfigWriter:
    """
    Stream DOF snapshots to a single JSONL trajectory file.
    One instance per run; created at startup, closed at end.
    """

    def __init__(self, run_name: str = ""):
        """
        Initialize ConfigWriter and create the JSONL file.

        Args:
            run_name: optional prefix for filename (stem only, no .jsonl).
                     If empty, auto-generate run_<timestamp>_<pid>.
        """
        if not run_name:
            timestamp = int(time.time() * 1000)  # milliseconds
            pid = os.getpid()
            run_name = f"run_{timestamp}_{pid}"

        self.run_name = run_name
        self.filepath = LOGS_DIR / f"{run_name}.jsonl"

        # Ensure LOGS_DIR exists
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        # Open in append mode; allows resume if interrupted mid-run
        self.file = open(self.filepath, "a")
        self.snapshot_count = 0

    def append_snapshot(
        self, iteration: int, bc: dict, q: list, error_percent: float, timestamp: float, **kwargs
    ):
        """
        Write one DOF snapshot to the JSONL file.

        Args:
            iteration: orchestrator cycle number (1-indexed).
            bc: boundary condition dict with keys {x0, y0, theta}.
            q: DOF vector; list of floats (length = 2*nv for beams).
            error_percent: geometric shape error from evaluator (0–100).
            timestamp: cumulative simulation time (seconds).
            **kwargs: optional fields (e.g., orchestrator_prompt for first snapshot of iteration).
        """
        snapshot = {
            "iteration": iteration,
            "bc": bc,
            "q": q,
            "error_percent": error_percent,
            "timestamp": timestamp,
        }
        # Add any optional fields (e.g., orchestrator_prompt)
        snapshot.update(kwargs)
        self.file.write(json.dumps(snapshot) + "\n")
        self.file.flush()  # force to disk immediately
        self.snapshot_count += 1

    def close(self):
        """Flush and close the file handle."""
        if self.file and not self.file.closed:
            self.file.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __repr__(self):
        return f"ConfigWriter(file={self.filepath}, snapshots={self.snapshot_count})"
