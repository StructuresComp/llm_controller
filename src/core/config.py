# ---------------------------------------------------------------------------
# config.py
# ---------------------------------------------------------------------------
# Central configuration for the LLM controller framework.
# All tuneable settings live here. No other file should hardcode these values.
#
# Sections (in order):
#   1. LLM
#   2. Simulation & Visualization
#   3. Agent Perception & Control
# ---------------------------------------------------------------------------


# ===========================================================================
# 1. LLM
# ===========================================================================
# Provider choices: "anthropic", "openai", "ollama", "fake"
# Provider can also be inferred from the model name (claude-* → anthropic,
# gpt-/o1-/o3-/o4-* → openai, gemini-* → gemini).

qwen = True  # if True, use Qwen models (only available via Ollama for now)
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
# 2. SIMULATION & VISUALIZATION
# ===========================================================================

# ---------------------------------------------------------------------------
# 2a. Physical parameters
#     These define the beam/DLO. Change here to try different materials or
#     geometries. make_physical_system(config) reads these once at startup.
# ---------------------------------------------------------------------------
NV          = 21       # number of vertices (nodes); total DOFs = 2 * NV
RHO         = 10000    # density (kg/m³)
Y           = 1e6      # Young's modulus (Pa)
R0          = 1e-2     # cross-sectional radius (m)
ROD_LENGTH  = 0.1      # total beam length (m)
VISC        = 1000.0   # viscous damping coefficient (Pa·s)

# ---------------------------------------------------------------------------
# 2b. Environment
# ---------------------------------------------------------------------------
WALL_H                  = -0.12  # wall position: wall sits at y = WALL_H (m); must satisfy |WALL_H| > ROD_LENGTH
BOUNDARY_VELOCITY       = 0.05   # translational speed of node 0 (m/s)
BOUNDARY_ANGLE_VELOCITY = 0.05   # rotational speed of node 0 (rad/s)
DT                      = 0.001   # time step size for physics integration (s); smaller = more accurate but slower
FRICTION                = True  # True = wall fixes x and y; False = wall fixes y only (frictionless)

# ---------------------------------------------------------------------------
# 2c. Visualization
# ---------------------------------------------------------------------------
PLOT_IMAGE = 1   # 1 = show beam plots during simulation, 0 = suppress


# ===========================================================================
# 3. AGENT PERCEPTION & CONTROL
# ===========================================================================

# ---------------------------------------------------------------------------
# 3a. Orchestrator observation level
#     Controls what geometric information is sent to the orchestrator.
#     "positions"  — raw (x, y) coordinates of observable nodes only
#     "geometric"  — positions + tangent angles + curvature signs + total turning angle
#     "full"       — geometric + wall contact status per node + arc length distribution
# ---------------------------------------------------------------------------
OBSERVATION_LEVEL = "geometric"  # change to "geometric" or "full" for richer input

# ---------------------------------------------------------------------------
# 3b. Orchestrator perception & history
# ---------------------------------------------------------------------------
N_PARTIAL  = 5    # nodes sent to orchestrator (subsampled from full NV)
K_HISTORY  = 5    # number of (shape, BC) pairs kept in rolling history

# ---------------------------------------------------------------------------
# 3c. Control loop
# ---------------------------------------------------------------------------
CHECK_SHAPE_REFLECTION = False  # if True, also test reversed traversal and take min error

ORCHESTRATOR_MODE = "plan"   # "plan" (grounded) or "myopic" (reactive)
ALLOW_REPLAN      = False    # if True, orchestrator can discard and redo the plan
MAX_ITERATIONS    = 50       # hard stop on orchestrator → programmer cycles
MAX_CONFIGS       = 1000     # max configurations stored in state before downsampling
