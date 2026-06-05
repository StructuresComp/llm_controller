# ESL Agent Framework — LLM Control of Deformable Objects

A two-agent LangGraph framework for using large language models (and future vision models) to control deformable objects. The system uses an LLM as a reactive or strategic controller: it observes the current configuration, reasons about physics, and issues control commands to steer the system toward target shapes.

**Research question:** How well can LLMs (and later VLMs) perform as controllers in physical systems they cannot directly observe or manipulate?

## Problem Setup

**The Task:**
- A deformable elastic beam hangs vertically under gravity.
- One end (node 0) is held by a robotic arm that you control via position (x0, y0) and angle (θ).
- The other end is free and can contact a wall.
- Goal: deform the beam into a target shape (C, U, S, L) by commanding the arm.

**The Controller:**
- An LLM observes the current beam configuration (positions of observable nodes).
- It optionally has access to a prior strategy ("plan mode") or decides reactively ("myopic mode").
- It issues the next boundary condition: where to move the arm and what angle to hold.
- A physics simulator executes the command and returns the new configuration.
- The LLM assesses whether the shape matches the target. If not, the loop repeats.

**Key insight:** The LLM's ability to recognize convergence is intentionally *not* overridden by a deterministic checker. The LLM's judgment (correct or incorrect) is part of what is being measured.

## Architecture

```
┌─────────────────┐
│   ESLState      │
│  (observations) │
└────────┬────────┘
         │
         ▼
    ┌─────────────────────────────────────────┐
    │  Orchestrator Sub-graph                 │
    │  ┌──────────────┐                       │
    │  │ planner_node │  (plan mode only)     │
    │  └──────┬───────┘                       │
    │         ▼                               │
    │  ┌──────────────────┐                   │
    │  │ decision_node    │  LLM BC decision  │
    │  └──────┬───────────┘                   │
    │         │ decision, converged           │
    └─────────┼──────────────────────────────┘
              │
              ▼
    ┌─────────────────────────────────────┐
    │  Programmer Node                    │
    │  (Deterministic Python)             │
    │  → discreteBeam() physics tool      │
    │  → update state                     │
    └──────────┬────────────────────────┘
               │
         converged?
         /          \
       YES          NO
        │            │
      [END]     (loop back)
```

## Quick Start

```bash
git clone <repo-url>
cd agents_ESL
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and add your API key:
```bash
ANTHROPIC_API_KEY=your_anthropic_key_here
# OPENAI_API_KEY=your_openai_key_here
```

If you use **Ollama**, skip the `.env` step and set these in `config.py` instead:
```python
LLM_PROVIDER_FAST  = "ollama"
FAST_MODEL         = "qwen2.5:14b"
OLLAMA_BASE_URL    = "http://localhost:11434"
```

### 3. Test Physics Tool (Optional)

Verify beam simulator chaining works (two consecutive runs where the second starts from the first's final state):

```bash
python -m src.tests.test_beam_chaining
```

### 4. Run the Agent Loop

```bash
python cli.py --target C
```

**CLI Arguments:**
- `--target SHAPE` — target shape label: `C`, `U`, `S`, `L`
- `--H DEPTH` — override wall depth (e.g., `--H 0.15` sets wall at y = -0.15 m)

**Example:**
```bash
python cli.py --target U --H 0.15
```

### 5. Observe Real-Time Output

The orchestrator prints decisions and convergence reasoning at each iteration:

```
[Iteration 1] Orchestrator decision:
  x0 = +0.000000 m  |  y0 = -0.060000 m  |  θ = +0.000000 rad
  → Not converged: The beam is in its initial straight-down configuration...

[Iteration 2] Orchestrator decision:
  x0 = +0.000000 m  |  y0 = -0.100000 m  |  θ = +0.000000 rad
  → Not converged: The beam shows positive curvature but lacks the symmetry...

[Iteration 3] Orchestrator decision:
  x0 = +0.040000 m  |  y0 = -0.100000 m  |  θ = +0.300000 rad
  ✓ CONVERGED: The beam exhibits the correct C-shape characteristics...
```

## Configuration

All settings live in `config.py`, organized into three sections:

### Section 1: LLM

| Setting | Default | Description |
|---|---|---|
| `LLM_PROVIDER_FAST` | `"anthropic"` | Provider: `"anthropic"`, `"openai"`, `"ollama"`, `"fake"` |
| `LLM_PROVIDER_SMART` | `"anthropic"` | Provider for heavy reasoning (planning) |
| `FAST_MODEL` | `"claude-haiku-4-5"` | Model for BC decisions & convergence checks |
| `SMART_MODEL` | `"claude-haiku-4-5"` | Model for planning (swap to `claude-sonnet-4-6`) |
| `MAX_TOKENS_FAST` | `512` | Token cap for fast calls |
| `MAX_TOKENS_SMART` | `2048` | Token cap for planning |
| `LOG_LLM_CALLS` | `False` | If True, log all LLM calls to `logs/llm_calls.jsonl` |

### Section 2: Simulation & Visualization

**2a. Physical Parameters** (describe the beam):

| Setting | Default | Description |
|---|---|---|
| `NV` | `21` | Number of beam vertices (nodes) |
| `RHO` | `10000` | Density (kg/m³) |
| `Y` | `1e6` | Young's modulus (Pa) |
| `R0` | `1e-2` | Cross-sectional radius (m) |
| `ROD_LENGTH` | `0.1` | Total beam length (m) |
| `VISC` | `1000.0` | Viscous damping (Pa·s) |

**2b. Environment:**

| Setting | Default | Description |
|---|---|---|
| `WALL_H` | `-0.12` | Wall position: y = WALL_H (m, negative) |
| `BOUNDARY_VELOCITY` | `0.01` | Speed of arm translational motion (m/s) |
| `BOUNDARY_ANGLE_VELOCITY` | `0.01` | Speed of arm rotational motion (rad/s) |
| `DT` | `0.01` | Physics timestep (s) — smaller = more accurate, slower |
| `FRICTION` | `False` | Wall friction: True = fixes x and y; False = y only |

**2c. Visualization:**

| Setting | Default | Description |
|---|---|---|
| `PLOT_IMAGE` | `1` | 1 = show plots during simulation, 0 = suppress |

### Section 3: Agent Perception & Control

**3a. Observation Level** (controls what info the orchestrator receives):

| Setting | Default | Description |
|---|---|---|
| `OBSERVATION_LEVEL` | `"geometric"` | `"positions"` = raw (x,y) only; `"geometric"` = + tangent angles + curvature signs + total turning angle; `"full"` = reserved for future |

**3b. History & Perception:**

| Setting | Default | Description |
|---|---|---|
| `N_PARTIAL` | `5` | Number of observable nodes shown to orchestrator (subsampled from NV) |
| `K_HISTORY` | `5` | Rolling window of (shape, BC) pairs kept in context |

**3c. Control Loop:**

| Setting | Default | Description |
|---|---|---|
| `ORCHESTRATOR_MODE` | `"plan"` | `"plan"` (grounded strategy) or `"myopic"` (reactive) |
| `ALLOW_REPLAN` | `False` | If True, clear the plan if stagnation is detected |
| `MAX_ITERATIONS` | `50` | Hard stop on orchestrator→programmer cycles |
| `MAX_CONFIGS` | `1000` | Cap on stored beam configurations (downsampled beyond this) |

## Key Features

### Observation Levels

The orchestrator can receive shape information at different abstraction levels:

**`"positions"`** (simplest)
```
Observable nodes: [(x₀, y₀), (x₁, y₁), ..., (x₄, y₄)]
→ LLM infers shape from numbers alone
```

**`"geometric"`** (richer, recommended)
```
Observable nodes: [(x₀, y₀), ..., (x₄, y₄)]
Tangent angles: [θ₀, θ₁, ..., θ₄] rad (counter-clockwise from -Y axis)
Curvature signs: [+1, +1, -1, ...] (local left/straight/right bends)
Total turning angle: 1.57 rad
Wall contact: 60%
→ LLM receives the language of mechanics
```

### Control Modes

**Plan Mode** (`ORCHESTRATOR_MODE = "plan"`):
- First iteration: LLM writes a multi-step strategy based on target shape and beam properties.
- Subsequent iterations: LLM appends the strategy to the decision prompt, anchoring reasoning.
- Risk: wrong initial plan commits the controller. Set `ALLOW_REPLAN = True` to recover.

**Myopic Mode** (`ORCHESTRATOR_MODE = "myopic"`):
- No plan. LLM sees only the last K (shape, BC) pairs and reacts.
- More exploration, but prone to oscillation and longer convergence.

### Iteration Logging

After each successful iteration, the system logs:
```python
{
  "iteration": 1,
  "decision": {"x0": 0.0, "y0": -0.06, "theta": 0.0},
  "tip_x": 0.067,
  "tip_y": -0.120
}
```

All iterations are printed at the end in a summary table.

## Debugging & Logs

### LLM Call Logging

Set `LOG_LLM_CALLS = True` to capture all LLM interactions:

```bash
tail -5 logs/llm_calls.jsonl | jq
```

Each entry contains:
```json
{
  "timestamp": "2026-06-04T17:15:23.456789",
  "call_type": "text" or "tool",
  "model": "claude-haiku-4-5",
  "system": "You are ...",
  "prompt": "...",
  "response": "...",
  "tool_name": "set_boundary_condition" or null
}
```

Useful queries:
```bash
# Find all BC decisions
grep '"tool_name": "set_boundary_condition"' logs/llm_calls.jsonl | jq .response

# Find all convergence assessments
grep '"tool_name": "assess_convergence"' logs/llm_calls.jsonl | jq .response
```

### Shape Error Checking

After a run, you can evaluate how close the final shape is to the target using the shape checker:

```python
from src.tools.shape_checker import compute_shape_error

error = compute_shape_error(final_observable_nodes, target_shape='C')
print(f"Match confidence: {error['match_confidence_percent']:.1f}%")
print(f"Diagnostics: {error['diagnostics']}")
```

## Project Structure

```
agents_ESL/
├── cli.py                              # Entry point
├── config.py                           # All settings (3 sections)
├── paths.py                            # Directory paths (DATA_DIR, LOGS_DIR, etc.)
├── requirements.txt                    # Dependencies
│
├── logs/
│   └── llm_calls.jsonl                 # LLM call log (if LOG_LLM_CALLS=True)
│
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── state_llm_control.py        # ESLState TypedDict + initial_esl_state()
│   │   └── physical_system.py          # PhysicalSystem TypedDict + make_physical_system()
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── orchestrator_agent.py       # planner_node + decision_node
│   │   └── programmer_agent.py         # discreteBeam executor
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── llm.py                      # call_llm(), call_llm_with_tool()
│   │   └── providers/                  # anthropic, openai, ollama, fake
│   │
│   ├── tools/
│   │   ├── Discrete_Elastic_Beams.py   # Physics simulator (quasi-static)
│   │   └── shape_checker.py            # Geometric shape validator
│   │
│   └── tests/
│       ├── test_beam_chaining.py       # Physics tool integration test
│       └── test_shape_checker.py       # Shape error computation test
│
└── workflows/
    ├── __init__.py
    └── workflow.py                     # LangGraph: orchestrator → programmer
```

## Physics Tool: `discreteBeam`

The core simulation engine. Signature:

```python
discreteBeam(
    physical_system,           # dict: pre-computed beam + environment
    q0=None, u0=None,         # Initial DOF and velocity (auto-init if None)
    t_start=0.0,              # Cumulative simulation time (for chaining)
    control_position,         # (x0, y0) target for node 0
    control_angle,            # θ target angle (radians)
    dt=0.01,                  # Physics timestep
    totalTime=None,           # Simulation duration (auto-computed if None)
    plotImage=0               # 0 = no plot, 1 = show
) → dict
```

Returns:
```python
{
    "status": "success" or "failed_convergence",
    "message": "Simulation completed 500 steps to t = 5.0 s",
    "q_new": [...],          # Final DOF vector
    "u_new": [...],          # Final velocity vector
    "ctime": 5.0             # Cumulative time at end
}
```

**Key behaviors:**
- BC interpolates linearly from current to target over `totalTime`.
- Wall contact is handled via predictor/corrector Newton iteration.
- High `EI` (bending stiffness) resists deformation — moves must be gradual.

## Adding a New LLM Provider

1. Create `src/llm/providers/my_provider.py` inheriting from `BaseProvider`:
```python
from src.llm.providers.base_provider import BaseProvider

class MyProvider(BaseProvider):
    def call_text(self, prompt, system, max_tokens, model):
        # Return: str (the response)
        ...
    
    def call_tool(self, prompt, tool_schema, tool_name, system, max_tokens, model):
        # Return: dict (validated structured output)
        ...
```

2. Register in `src/llm/llm.py`:
```python
PROVIDERS = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "ollama": OllamaProvider,
    "my_provider": MyProvider,  # Add here
}
```

3. Set in `config.py`:
```python
LLM_PROVIDER_FAST = "my_provider"
```

## Next Steps

See `TODO.md` for the full roadmap:

- **Item 11:** Rename ESL throughout (framework is now generic LLM control, not energy-learning specific)
- **Item 12:** Deterministic shape checker (✓ implemented) — compute shape error independently
- **Item 13:** Visualization module — render real-time + iteration summary frames, compose video
- **Item 14:** Evaluation harness — run multiple trials, measure convergence rate and LLM self-assessment accuracy
- **Item 15:** VLM integration — replace text observation with rendered images

## Research Context

This framework investigates whether LLMs can serve as effective controllers for physical systems through:
1. **Grounded planning** — does prior strategy help vs. reactive control?
2. **Observation abstraction** — does geometric reasoning (curvature, angles) outperform raw coordinates?
3. **Self-assessment** — can the LLM judge convergence accurately, or do we need deterministic validators?
4. **VLM capabilities** — can vision outperform text for shape judgment?

The convergence decision is **intentionally left to the LLM**. This is the research question, not a limitation to work around.
