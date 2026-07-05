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

ollama = True  # if True, use ollama models
if ollama:
    LLM_PROVIDER = "ollama"
    MODEL        = "olmo2:13b" # "qwen2.5-coder:14b"
    # Ollama models available now (as of 07/05/2026). Local/smaller models are
    # less reliable at tool-calling than frontier models (Claude, GPT, Gemini) --
    # TOOL_CALL_ATTEMPTS below exists to retry past occasional malformed tool-call
    # responses from these; it should rarely be needed with frontier models.
    # "qwen2.5:14b"        -- reliable tool-calling # Aggressive
    # "qwen2.5-coder:14b"  -- unreliable tool-calling
    # "phi4"               -- no native tool-calling support; falls back to JSON-prompt emulation
    # "mistral-nemo"       -- reliable tool-calling
    # "gemma3:12b"         -- reliable tool-calling # Conservative
    # "olmo2:13b"          -- reliable tool-calling # Aggressive

    # Ollama-specific settings (ignored when provider is not "ollama")
    OLLAMA_BASE_URL    = "https://passenger-dod-cafe-placed.trycloudflare.com"
    OLLAMA_TIMEOUT     = 300   # seconds
    OLLAMA_TEMPERATURE = 0.0
else:
    LLM_PROVIDER = "anthropic"
    MODEL        = "claude-haiku-4-5"
    MODEL        = "claude-opus-4-8" # "claude-sonnet-4-6"

    LLM_PROVIDER = "gemini"
    MODEL        = "gemini-3.5-flash"
    # LLM_PROVIDER = "openai"
    # MODEL        = "gpt-5.5"

MAX_TOKENS   = 20000 # Maximum allowed tokens per LLM call
TOOL_CALL_ATTEMPTS = 5   # 0 (or <1) = quit immediately if a tool-call LLM response fails to parse
                         # >=1 = retry that many additional times before quitting
                         # Mainly needed for local Ollama models (some occasionally fail to
                         # produce a valid tool call); frontier models rarely need retries.

# ===========================================================================
# 2. AGENT PERCEPTION & CONTROL
# ===========================================================================

# ---------------------------------------------------------------------------
# 2a. Orchestrator observation level
#     Controls what geometric information is sent to the orchestrator.
#     Raw (x, y) coordinates of observable nodes are always included.
#     The following flags control whether additional info is included in the prompt.
# ---------------------------------------------------------------------------
INCLUDE_GEOMETRY       = True    # True = include tangent angles + curvature in the prompt
INCLUDE_AXIAL_STRETCH  = True    # warn the LLM the DLO is stretching too much (axial strain>=AXIAL_STRAIN_THRESHOLD)
INCLUDE_FEEDBACK       = False   # include controller reaction force in the prompt
INCLUDE_CONTACT        = False   # include wall contact regions in the prompt
AXIAL_STRAIN_THRESHOLD = 0.05    # fraction (0.05 = 5%): warn if any edge's axial strain exceeds this
SHOW_ERROR_PERCENT     = True    # True = show error % in prompts; False = hide it (forces geometric reasoning)

# ---------------------------------------------------------------------------
# 2b. Orchestrator perception & history
# ---------------------------------------------------------------------------
N_PARTIAL  = 15  # nodes sent to orchestrator (subsampled from full NV under "Physical parameters")
K_HISTORY  = 5   # number of (shape, BC) pairs kept in rolling history; set to 0 or negative to disable

# ---------------------------------------------------------------------------
# 2d. Control loop
# ---------------------------------------------------------------------------
ORCHESTRATOR_MODE    = "myopic" # "plan" (grounded), "myopic" (reactive), or "replan" (fresh plan every iteration)
REPLAN_FREQ          = 5        # replan every Nth iteration in "replan" mode (1 = every iteration)
MAX_ITERATIONS       = 100      # hard stop on orchestrator → programmer cycles
MAX_ITERATIONS_WITH_REJECTIONS = MAX_ITERATIONS * 2  # hard stop on ALL orchestrator decisions, including rejected BCs

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
WALL_INF_MULTIPLIER     = 10     # wall spans > this x ROD_LENGTH are treated as unbounded in prompts

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
DT_MIN        = 0.002   # give up below this dt or retry based on QUIT_ON_DT_FAILURE setting
QUIT_ON_DT_FAILURE = False # True = end the run if dt bottoms out without converging
                           # False = reject the BC instead and let the orchestrator retry

# ---------------------------------------------------------------------------
# 3d. Visualization
# ---------------------------------------------------------------------------
PLOT_IMAGE = 0  # 0 = no plots; N >= 1 = plot every Nth physics step

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
