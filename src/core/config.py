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
DEBUG         = True  # True = print LLM prompts and responses to terminal

qwen = False  # if True, use Qwen models (only available via Ollama for now)
if qwen:
    LLM_PROVIDER = "ollama"
    MODEL        = "qwen2.5:14b"
    MAX_TOKENS   = 8000
    # Ollama-specific settings (ignored when provider is not "ollama")
    OLLAMA_BASE_URL    = "https://assure-roots-hostels-excessive.trycloudflare.com"
    OLLAMA_TIMEOUT     = 300   # seconds
    OLLAMA_TEMPERATURE = 0.0
else:
    LLM_PROVIDER = "anthropic"
    MODEL        = "claude-haiku-4-5"
    # MODEL        = "claude-sonnet-4-6"
    MAX_TOKENS   = 8000   # hard upper bound on output tokens per LLM call

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
INCLUDE_GEOMETRY       = True    # True = include tangent angles + curvature in the prompt
INCLUDE_FEEDBACK       = False   # include controller reaction force in the prompt
INCLUDE_CONTACT        = False   # include wall contact regions in the prompt
INCLUDE_AXIAL_STRETCH  = True    # warn the LLM when axial strain exceeds AXIAL_STRAIN_THRESHOLD
AXIAL_STRAIN_THRESHOLD = 0.05    # fraction (0.05 = 5%): warn if any edge strain exceeds this

# ---------------------------------------------------------------------------
# 2b. Orchestrator perception & history
# ---------------------------------------------------------------------------
N_PARTIAL  = 7   # nodes sent to orchestrator (subsampled from full NV under "Physical parameters")
K_HISTORY  = 3    # number of (shape, BC) pairs kept in rolling history; set to 0 or negative to disable

# ---------------------------------------------------------------------------
# 2d. Control loop
# ---------------------------------------------------------------------------
ORCHESTRATOR_MODE    = "myopic" # "plan" (grounded), "myopic" (reactive), or "replan" (fresh plan every iteration)
REPLAN_FREQ          = 5        # replan every Nth iteration in "replan" mode (1 = every iteration)
MAX_ITERATIONS       = 1000     # hard stop on orchestrator → programmer cycles

# ===========================================================================
# 3. SIMULATION & VISUALIZATION
# ===========================================================================

# ---------------------------------------------------------------------------
# 3a. Physical parameters
#     These define the beam/DLO. Change here to try different materials or
#     geometries. make_physical_system(config) reads these once at startup.
# ---------------------------------------------------------------------------
PHYSICS_TYPE = "beam"  # "beam" (2D elastic beam) | "rod" (3D elastic rod, under development)
NV          = 15       # number of vertices (nodes); total DOFs = 2 * NV
RHO         = 1000     # density (kg/m³)
Y           = 1e6      # Young's modulus (Pa)
R0          = 1e-2     # cross-sectional radius (m)
ROD_LENGTH  = 0.2      # total beam length (m)
VISC        = 10.0   # viscous damping coefficient (Pa·s)
GRAVITY     = [1e-3, -9.8]  # gravitational acceleration vector [gx, gy] (m/s^2); small gx breaks symmetry

# ---------------------------------------------------------------------------
# 3b. Environment
# ---------------------------------------------------------------------------
WALLS = []  # list of wall dicts; empty = no wall
# WALLS = [{"P0": [-10, -0.12], "P1": [10, -0.12]}]
# Each wall is a directed segment {"P0": [x0,y0], "P1": [x1,y1]}.
# Outward normal = (P1-P0) rotated 90° CCW — traversal order sets which side is blocked.
# Use large sentinels (±10) for walls that should span the full simulation domain.
# Empty list = no wall.

TOLERANCE_DISTANCE = 0.02 * ROD_LENGTH  # min clearance (m) the controlled nodes must keep from any wall
                                        # safety agent flags a violation if a node comes within this distance
                                        # 10% of rod length is a reasonable default; set to 0 to disable

BOUNDARY_VELOCITY       = 0.05   # translational speed of node 0 (m/s)
BOUNDARY_ANGLE_VELOCITY = 0.05   # rotational speed of node 0 (rad/s)
FRICTION                = False  # True = wall fixes x and y; False = wall fixes y only (frictionless)
WALL_INF_MULTIPLIER     = 10     # wall spans > this × ROD_LENGTH are treated as unbounded in prompts

# ---------------------------------------------------------------------------
# 3c. Time stepping
#     DT        — base time step for physics integration; smaller = more accurate but slower.
#     ADAPT_DT  — if True, retry a failed-convergence call with DT * DT_MULTIPLIER until
#                 DT drops below DT_MIN, then give up.
#     DT_MULTIPLIER — factor applied to DT on each retry (must be < 1).
#     DT_MIN        — smallest DT allowed before the retry loop gives up (must be < DT).
# ---------------------------------------------------------------------------
DT            = 0.01   # base time step (s)
ADAPT_DT      = True    # retry with smaller dt on failed convergence
DT_MULTIPLIER = 0.5     # halve dt on each retry
DT_MIN        = 0.0001  # give up below this dt

# ---------------------------------------------------------------------------
# 3d. Visualization
# ---------------------------------------------------------------------------
PLOT_IMAGE = 5   # 0 = no plots; N >= 1 = plot every Nth physics step

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
SAVE_STRIDE = 10     # save every 50th physics step per decision
MAX_CONFIGS = 1000   # hard cap; oldest trimmed if exceeded
