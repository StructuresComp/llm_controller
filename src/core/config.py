# ---------------------------------------------------------------------------
# config.py
# ---------------------------------------------------------------------------
# Central configuration for the LLM controller framework.
# All tuneable settings live here. No other file should hardcode these values.
#
# Sections (in order):
#   1. LLM
#   2. Agent Perception & Control
#   3. Simulation & Visualization
# ---------------------------------------------------------------------------


# ===========================================================================
# 1. LLM
# ===========================================================================
# Provider choices: "anthropic", "openai", "ollama", "fake"
# Provider can also be inferred from the model name (claude-* → anthropic,
# gpt-/o1-/o3-/o4-* → openai, gemini-* → gemini).

LOG_LLM_CALLS = True   # True = log every prompt + response alongside the run JSON
RUN_NAME      = ""     # stem for output files: <RUN_NAME>.json + <RUN_NAME>.log
                       # "" = auto-generate run_<timestamp>_<pid> for each file
DEBUG         = False  # True = print LLM prompts and responses to terminal

qwen = False  # if True, use Qwen models (only available via Ollama for now)
if qwen:
    LLM_PROVIDER = "ollama"
    MODEL        = "qwen2.5:14b"
    MAX_TOKENS   = 4096
    # Ollama-specific settings (ignored when provider is not "ollama")
    OLLAMA_BASE_URL    = "https://assure-roots-hostels-excessive.trycloudflare.com"
    OLLAMA_TIMEOUT     = 300   # seconds
    OLLAMA_TEMPERATURE = 0.0
else:
    LLM_PROVIDER = "anthropic"
    MODEL        = "claude-haiku-4-5"
    MAX_TOKENS   = 2048   # hard upper bound on output tokens per LLM call

# ===========================================================================
# 2. AGENT PERCEPTION & CONTROL
# ===========================================================================

# ---------------------------------------------------------------------------
# 2a. Orchestrator observation level
#     Controls what geometric information is sent to the orchestrator.
#     "positions"  — raw (x, y) coordinates of observable nodes only
#     "geometric"  — positions + tangent angles + curvature signs + total turning angle
#     "full"       — geometric + wall contact status per node + arc length distribution
# ---------------------------------------------------------------------------
OBSERVATION_LEVEL = "geometric"

# ---------------------------------------------------------------------------
# 2b. Orchestrator perception & history
# ---------------------------------------------------------------------------
N_PARTIAL  = 5    # nodes sent to orchestrator (subsampled from full NV under "Physical parameters")
K_HISTORY  = 5    # number of (shape, BC) pairs kept in rolling history

# ---------------------------------------------------------------------------
# 2d. Control loop
# ---------------------------------------------------------------------------
CHECK_SHAPE_REFLECTION = False  # if True, also test reversed traversal and take min error

ORCHESTRATOR_MODE = "plan"   # "plan" (grounded) or "myopic" (reactive)
ALLOW_REPLAN      = False    # if True, orchestrator can discard and redo the plan
MAX_ITERATIONS    = 10       # hard stop on orchestrator → programmer cycles
MAX_SAFETY_REJECTIONS = 10   # consecutive safety rejections of a BC before the run ends
                             # (bounds the orchestrator↔safety loop; see workflows/workflow.py)

# ===========================================================================
# 3. SIMULATION & VISUALIZATION
# ===========================================================================

# ---------------------------------------------------------------------------
# 3a. Physical parameters
#     These define the beam/DLO. Change here to try different materials or
#     geometries. make_physical_system(config) reads these once at startup.
# ---------------------------------------------------------------------------
PHYSICS_TYPE = "beam"  # "beam" (2D elastic beam) | "rod" (3D elastic rod, under development)
NV          = 21       # number of vertices (nodes); total DOFs = 2 * NV
RHO         = 10000    # density (kg/m³)
Y           = 1e6      # Young's modulus (Pa)
R0          = 1e-2     # cross-sectional radius (m)
ROD_LENGTH  = 0.1      # total beam length (m)
VISC        = 1000.0   # viscous damping coefficient (Pa·s)

# ---------------------------------------------------------------------------
# 3b. Environment
# ---------------------------------------------------------------------------
WALLS = []  # list of wall dicts; empty = no wall
# WALLS = [{"P0": [-10, -0.12], "P1": [10, -0.12]}]
# Each wall is a directed segment {"P0": [x0,y0], "P1": [x1,y1]}.
# Outward normal = (P1-P0) rotated 90° CCW — traversal order sets which side is blocked.
# Use large sentinels (±10) for walls that should span the full simulation domain.
# Empty list = no wall.

TOLERANCE_DISTANCE = 0.1 * ROD_LENGTH  # min clearance (m) the controlled nodes must keep from any wall
                                        # safety agent flags a violation if a node comes within this distance
                                        # 10% of rod length is a reasonable default; set to 0 to disable

BOUNDARY_VELOCITY       = 0.05   # translational speed of node 0 (m/s)
BOUNDARY_ANGLE_VELOCITY = 0.05   # rotational speed of node 0 (rad/s)
DT                      = 0.01   # time step size for physics integration (s); smaller = more accurate but slower
FRICTION                = False  # True = wall fixes x and y; False = wall fixes y only (frictionless)

# ---------------------------------------------------------------------------
# 3c. Visualization
# ---------------------------------------------------------------------------
PLOT_IMAGE = 1   # 1 = show beam plots during simulation, 0 = suppress

# ---------------------------------------------------------------------------
# 3d. Data storage
#     Controls how much simulation history is retained for the viewer.
#
#     SAVE_STRIDE  — save every Kth physics step within each LLM decision.
#                    1 = save every step (full resolution).
#                    N = save every Nth step (reduces data by factor N).
#                    The final step of each decision is always saved regardless
#                    of stride, since that is the state the LLM observes.
#
#     MAX_CONFIGS  — hard cap on total configurations stored in state.
#                    If exceeded, the oldest entries are trimmed from the front
#                    so recent history is always preserved.
# ---------------------------------------------------------------------------
SAVE_STRIDE = 50     # save every 50th physics step per decision
MAX_CONFIGS = 1000   # hard cap; oldest trimmed if exceeded
