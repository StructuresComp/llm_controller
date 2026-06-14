# ---------------------------------------------------------------------------
# src/core/paths.py
# ---------------------------------------------------------------------------
# All filesystem path constants. No logic — pure definitions.
# Import named constants from here; never hardcode paths elsewhere.
# ---------------------------------------------------------------------------

from pathlib import Path

# Project root — two levels up from src/core/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Runtime outputs — all three are gitignored; created on first use by the code that writes them
LOGS_DIR          = PROJECT_ROOT / "logs"   # LLM call log + run JSON (save_state.py, llm.py)
VISUALIZATION_DIR = PROJECT_ROOT / "viz"    # mechview output: index.html + data.js (render_3D.py)
