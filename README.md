# LLM Controller — LLM Control of Deformable Objects

A LangGraph framework for using large language models (and future vision models) as controllers for deformable linear objects (DLOs). The LLM observes the current beam configuration, reasons about the physics, and issues boundary condition commands to steer the beam toward a target shape.

**Research question:** How well can LLMs (and later VLMs) perform as controllers in physical systems they cannot directly observe or manipulate?

---

## Problem Setup

- A deformable elastic beam hangs vertically under gravity.
- One end (node 0) is held by a robotic arm. You control its position (x0, y0) and angle (θ).
- The other end is free and can contact a rigid wall below.
- **Goal:** deform the beam into a target shape (C, U, S, L) by commanding the arm.
- The LLM observes a subset of beam nodes, issues the next BC, and judges whether convergence is reached. Its convergence judgment is part of what is being measured.

---

## Architecture

```
[START] → orchestrator sub-graph → programmer_node → evaluator_node → (done?) → [END]
                  ↑___________________________________________________|
```

**Orchestrator sub-graph** (`planner_node → decision_node`):
- `planner_node` — writes a multi-step strategy on the first iteration (plan mode only).
- `decision_node` — always runs; calls LLM with a tool schema to produce the next BC `{x0, y0, theta}` and assess convergence.

**Programmer node** — deterministic, zero LLM calls. Runs `discreteBeam()` physics, updates state.

**Evaluator node** — deterministic, zero LLM calls. Looks up the active benchmark, runs `loss_fn()`, writes `state["eval_result"]`.

See **[docs/CONTRIBUTING.md](docs/CONTRIBUTING.md)** for how to add new benchmark cases, run tests, and understand the codebase structure.

---

## Quick Start

**Requires Python 3.11+.**

```bash
git clone <repo-url>
cd llm_controller
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Verify you are inside the venv:
```bash
which python3   # must show .../llm_controller/venv/bin/python3
```

Edit `.env` and add your API key:
```
ANTHROPIC_API_KEY=your_key_here
```

### Run the agent loop

```bash
python3 cli.py --target C
```

**CLI arguments:**
- `--target` — shape label: `C`, `U`, `S`, `L`
- `--H` — override wall depth (e.g., `--H 0.15` sets wall at y = −0.15 m)

### Run benchmark validation

```bash
python3 -m src.tests.test_all_shapes
```

### Run physics tool test

```bash
python3 -m src.tests.test_beam_chaining
```

---

## Configuration

All settings live in `src/core/config.py`.

### Section 1 — LLM

| Setting | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `"anthropic"` | Provider: `"anthropic"`, `"openai"`, `"ollama"`, `"fake"` |
| `MODEL` | `"claude-haiku-4-5"` | Model used for all LLM calls |
| `MAX_TOKENS` | `2048` | Hard token cap per LLM call |

To use Ollama, also set `OLLAMA_BASE_URL` in config.py and `LLM_PROVIDER = "ollama"`.

To enable LLM call logging, add `LOG_LLM_CALLS=1` to your `.env`.

### Section 2 — Simulation & Visualization

| Setting | Default | Description |
|---|---|---|
| `NV` | `21` | Number of beam vertices |
| `RHO` | `10000` | Density (kg/m³) |
| `Y` | `1e6` | Young's modulus (Pa) |
| `R0` | `1e-2` | Cross-sectional radius (m) |
| `ROD_LENGTH` | `0.1` | Total beam length (m) |
| `VISC` | `1000.0` | Viscous damping (Pa·s) |
| `WALL_H` | `-0.12` | Wall position: y = WALL_H (m, negative) |
| `BOUNDARY_VELOCITY` | `0.05` | Arm translational speed (m/s) |
| `BOUNDARY_ANGLE_VELOCITY` | `0.05` | Arm rotational speed (rad/s) |
| `DT` | `0.001` | Physics timestep (s) |
| `FRICTION` | `True` | True = wall fixes x+y; False = y only |
| `PLOT_IMAGE` | `1` | 1 = show plots during simulation, 0 = suppress |

### Section 3 — Agent Perception & Control

| Setting | Default | Description |
|---|---|---|
| `OBSERVATION_LEVEL` | `"geometric"` | `"positions"` = raw (x,y); `"geometric"` = + tangent angles + curvature signs |
| `N_PARTIAL` | `5` | Observable nodes shown to orchestrator |
| `K_HISTORY` | `5` | Rolling window of (shape, BC) pairs |
| `ORCHESTRATOR_MODE` | `"plan"` | `"plan"` (grounded strategy) or `"myopic"` (reactive) |
| `ALLOW_REPLAN` | `False` | If True, clear the plan if stagnation is detected |
| `MAX_ITERATIONS` | `50` | Hard stop on control cycles |

---

## Project Structure

```
llm_controller/
├── cli.py                               # Entry point: python3 cli.py --target C
├── requirements.txt
├── .env.example
├── docs/
│   ├── cases.md                         # Task definitions, losses, metrics
│   └── paper_outline.md
│
└── src/
    ├── core/
    │   ├── config.py                    # All tuneable settings
    │   └── paths.py                     # PROJECT_ROOT, DATA_DIR, LOGS_DIR, etc.
    │
    ├── models/
    │   ├── state_llm_control.py         # ControllerState TypedDict + initial_controller_state()
    │   └── physical_system.py           # PhysicalSystem TypedDict + make_physical_system()
    │
    ├── agents/
    │   ├── orchestrator_agent.py        # planner_node + decision_node + build_orchestrator_graph()
    │   ├── programmer_agent.py          # deterministic physics executor — zero LLM calls
    │   └── evaluator_agent.py           # deterministic evaluator — runs benchmark loss_fn
    │
    ├── workflows/
    │   └── workflow.py                  # build_controller_graph() — master LangGraph
    │
    ├── benchmarks/
    │   ├── __init__.py                  # Benchmark dataclass + register() + get_benchmark()
    │   ├── losses.py                    # Shared loss primitives
    │   ├── visualization.py             # plot_shape(), plot_grid()
    │   └── shapes/
    │       ├── beam_C.py
    │       ├── beam_U.py
    │       ├── beam_S.py
    │       └── beam_L.py
    │
    ├── llm/
    │   └── llm.py                       # Thin adapter over llm_wrapper pip package
    │
    ├── tools/
    │   ├── Discrete_Elastic_Beams.py    # discreteBeam() — physics + wall contact
    │   └── shape_checker.py             # compute_shape_error(), θ-profile fitting
    │
    └── tests/
        ├── test_beam_chaining.py
        └── test_all_shapes.py
```

---

## LLM Layer

Provider implementations live in the [`llm_wrapper`](https://github.com/agentic-writing/llm_wrapper) package (installed via pip). `src/llm/llm.py` is a thin adapter that reads `LLM_PROVIDER`, `MODEL`, and `MAX_TOKENS` from config and delegates to it.

To add a new provider, see `CONTRIBUTING.md` in the llm_wrapper repo.

---

## Benchmark System

Each benchmark is a `Benchmark` dataclass registered in `src/benchmarks/__init__.py`:
- `loss_fn(nodes, ps) -> dict` — deterministic loss function; returns `error_percent`
- `reference_fn(n, arc_length) -> list` — generates the ideal target shape
- `success_threshold` — `error_percent` below which the task is considered solved

Standalone test for a single shape:
```bash
python3 -m src.benchmarks.shapes.beam_C
```

Full registry validation + grid plot:
```bash
python3 -m src.tests.test_all_shapes
```

---

## Research Context

This framework investigates whether LLMs can serve as effective controllers for physical systems:

1. **Plan vs. myopic** — does grounding in a prior strategy improve convergence?
2. **Observation abstraction** — does geometric reasoning (curvature, tangent angles) outperform raw coordinates?
3. **Self-assessment accuracy** — can the LLM judge convergence correctly?
4. **LLM vs. VLM** — can vision outperform text for shape judgment?

The convergence decision is **intentionally left to the LLM**. The deterministic evaluator runs in parallel as ground truth, but the LLM's own convergence call is a measured quantity — its accuracy is part of what is being studied.
