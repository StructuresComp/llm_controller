# ---------------------------------------------------------------------------
# paths.py
# ---------------------------------------------------------------------------
# All filesystem path constants. No logic — pure definitions.
# Import named constants from here; never hardcode paths elsewhere.
# ---------------------------------------------------------------------------

from pathlib import Path

# Project root — the directory containing this file.
PROJECT_ROOT = Path(__file__).resolve().parent

# Runtime outputs
DATA_DIR          = PROJECT_ROOT / "data"           # saved simulation outputs (q_new, configs)
VISUALIZATION_DIR = PROJECT_ROOT / "visualizations" # saved plots and figures
LOGS_DIR          = PROJECT_ROOT / "logs"           # run logs
