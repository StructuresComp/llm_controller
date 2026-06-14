# LLM Controller — LLM Control of Deformable Objects

A LangGraph framework for using large language models (and future vision models) as controllers for deformable linear objects (DLOs). The LLM observes the current beam configuration, reasons about the physics, and issues boundary condition commands to steer the beam toward a target shape.

Right now, it only supports beams (2D DLO). Khalid is working on the rods (3D DLOs).

Recommendation for students to get up and running:
- Follow the instructions to install it.
- Run this case ``python3 cli.py --target C --H 0.15`` where the LLM will try to bend the DLO against a wall at ``y=-0.15`` and give it a C-shape.
- To visualize it, run ``python3 render_3D.py`` and open ``viz/index.html`` (Click play in your browser)

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

---

## Quick Start

**Requires Python 3.11+.**

```bash
git clone <repo-url>
cd llm_controller
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run the controller

```bash
python3 cli.py --target C --H 0.15               # shape + custom wall depth
python3 cli.py --target C                         # predefined shape
python3 cli.py --target tip_reach                # tip-reach benchmark
python3 cli.py --target C --length 0.15           # override rod length
python3 cli.py --target snake_L --config my_run.py  # custom config overlay
python3 cli.py --target C --mode beam            # explicit physics type
```

**CLI arguments:**

| Argument | Description |
|---|---|
| `--target SHAPE` | Benchmark name: `C`, `U`, `S`, `L`, `snake_L`, `tip_reach`, or any registered case. |
| `--mode MODE` | Physics type: `beam` (default) or `rod` (under development). Overrides `PHYSICS_TYPE` in config. |
| `--H DEPTH` | Add a single horizontal floor at `y = -\|H\|` m. Sign is ignored — `--H 0.15` and `--H -0.15` both place the floor at y = −0.15 m. Overrides `WALLS` in config. |
| `--length L` | Override `ROD_LENGTH` (m). |
| `--config FILE` | Path to a Python config overlay file (see [Configuration](#configuration)). |

Set `PLOT_IMAGE = 0` in `src/core/config.py` for a headless run (no matplotlib windows).

### Visualize a run

**Option 1: Interactive 3D rendering (mechview)**

```bash
python3 render_3D.py             # latest run in logs/
python3 render_3D.py logs/run_123.jsonl   # specific run
```

Writes `viz/index.html` + `viz/data.js`. Open `viz/index.html` in Chrome.

**Option 2: Analysis plots (error curves, trajectories)**

```bash
python3 analysis_plots.py --all              # error curve, trajectories, shapes
python3 analysis_plots.py --iter 5           # detailed view of iteration 5
python3 analysis_plots.py --animate          # play full trajectory animation
```

See `analysis_plots.py --help` for all options.

### Run a benchmark standalone

Each benchmark case in `src/benchmarks/cases/` has a built-in standalone test. Run any of them directly to inspect its reference geometry and loss function without launching the full controller:

```bash
python3 -m src.benchmarks.cases.beam_C
python3 -m src.benchmarks.cases.beam_U
python3 -m src.benchmarks.cases.beam_S
python3 -m src.benchmarks.cases.beam_L
python3 -m src.benchmarks.cases.beam_sine
python3 -m src.benchmarks.cases.beam_snake_L
```

To also evaluate a real configuration, set `TEST_CONFIG` at the top of the shape file to a list of `(x, y)` tuples (one per observable node) and re-run. It will overlay the actual shape on the reference and print the loss.

To validate all registered benchmarks at once:

```bash
python3 -m src.tests.test_all_shapes
```

---

## Configuration

All settings live in `src/core/config.py`. Nothing is hardcoded elsewhere.

### Section 1 — LLM

| Setting | Default | Description |
|---|---|---|
| `LOG_LLM_CALLS` | `True` | Write every prompt + response to `logs/<RUN_NAME>.log` |
| `RUN_NAME` | `""` | Stem for output files: `<RUN_NAME>.json` + `<RUN_NAME>.log`. Empty = auto `run_<timestamp>_<pid>` |
| `LLM_PROVIDER` | `"ollama"` | Provider: `"anthropic"`, `"openai"`, `"ollama"`, `"fake"` |
| `MODEL` | `"qwen2.5:14b"` | Model used for all LLM calls |
| `MAX_TOKENS` | `4096` | Hard token cap per LLM call |

For Ollama: set `OLLAMA_BASE_URL`, `OLLAMA_TIMEOUT`, `OLLAMA_TEMPERATURE` in config.  
For Anthropic: set `ANTHROPIC_API_KEY` in `.env`.

### Section 2 — Agent Perception & Control

| Setting | Default | Description |
|---|---|---|
| `OBSERVATION_LEVEL` | `"geometric"` | `"positions"` = raw (x,y); `"geometric"` = + tangent angles + curvature signs |
| `N_PARTIAL` | `5` | Observable nodes shown to orchestrator |
| `K_HISTORY` | `5` | Rolling window of (shape, BC) pairs |
| `ORCHESTRATOR_MODE` | `"plan"` | `"plan"` (grounded strategy) or `"myopic"` (reactive) |
| `ALLOW_REPLAN` | `False` | If True, clear the plan on stagnation |
| `MAX_ITERATIONS` | `50` | Hard stop on control cycles |

### Section 3 — Simulation & Visualization

| Setting | Default | Description |
|---|---|---|
| `NV` | `21` | Number of beam vertices |
| `RHO` | `10000` | Density (kg/m³) |
| `Y` | `1e6` | Young's modulus (Pa) |
| `R0` | `1e-2` | Cross-sectional radius (m) |
| `ROD_LENGTH` | `0.1` | Total beam length (m) |
| `VISC` | `1000.0` | Viscous damping (Pa·s) |
| `WALLS` | `[{"P0": [-10,-0.12], "P1": [10,-0.12]}]` | List of axis-aligned wall segments (horizontal or vertical only). Each `{"P0":[x,y], "P1":[x,y]}` — segment must be parallel to X or Y axis. Outward normal points into the allowed region. Empty list = no wall. |
| `BOUNDARY_VELOCITY` | `0.05` | Arm translational speed (m/s) |
| `BOUNDARY_ANGLE_VELOCITY` | `0.05` | Arm rotational speed (rad/s) |
| `DT` | `0.001` | Physics timestep (s) |
| `FRICTION` | `False` | True = wall fixes x+y; False = y only (frictionless) |
| `PLOT_IMAGE` | `1` | 1 = show plots during simulation, 0 = suppress |
| `SAVE_STRIDE` | `5` | Save every Nth physics step per decision (1 = all steps) |
| `MAX_CONFIGS` | `1000` | Hard cap on stored configurations; oldest trimmed if exceeded |

---

## Project Structure

```
llm_controller/
├── cli.py                               # Entry point: python3 cli.py --target C
├── render_3D.py                         # 3D viewer: python3 render_3D.py
├── analysis_plots.py                    # Analysis plots: python3 analysis_plots.py --all
├── requirements.txt
├── docs/
│   ├── cases.md                         # Task definitions, losses, metrics
│   └── paper_outline.md
│
└── src/
    ├── core/
    │   ├── config.py                    # All tuneable settings
    │   ├── paths.py                     # PROJECT_ROOT, LOGS_DIR, etc.
    │   └── save_state.py                # Persist finished run to logs/run_<timestamp>.json
    │
    ├── models/
    │   ├── state_llm_control.py         # ControllerState TypedDict + initial_controller_state()
    │   └── physical_system.py           # BeamPhysicalSystem + make_physical_system()
    │
    ├── agents/
    │   ├── orchestrator_agent.py        # planner_node + decision_node + build_orchestrator_graph()
    │   ├── programmer_agent.py          # deterministic physics executor — zero LLM calls
    │   └── evaluator_agent.py           # deterministic evaluator — runs benchmark loss_fn
    │
    ├── workflows/
    │   └── workflow.py                  # build_controller_graph() — master LangGraph
    │
    ├── visualization/
    │   ├── run_time_visual.py           # matplotlib plots during simulation
    │   └── mechview_export.py           # convert run JSON → mechview data.js
    │
    ├── benchmarks/
    │   ├── __init__.py                  # BenchmarkCase ABC + register() + get_benchmark()
    │   ├── case.py                      # BenchmarkCase base class
    │   ├── profile_fitting.py           # Shared profile math: build_ideal_from_edge_profile, profile_loss
    │   ├── loss_beam/                   # Reusable profiles and loss functions
    │   │   ├── profile_C.py             # Constant curvature (circular arc)
    │   │   ├── profile_S.py             # S-curve (full sine)
    │   │   ├── profile_U.py             # Semicircle
    │   │   ├── profile_L.py             # L-shape (step function)
    │   │   ├── profile_tip_reach.py     # Tip-reach loss function
    │   │   └── profile_custom.py        # Wrap custom curvature functions
    │   └── cases/                       # Benchmark cases (auto-discovered, auto-registered)
    │       ├── beam_C.py
    │       ├── beam_S.py
    │       ├── beam_U.py
    │       ├── beam_L.py
    │       ├── beam_sine.py
    │       ├── beam_snake_L.py
    │       └── beam_tip_reach.py
    │
    ├── llm/
    │   └── llm.py                       # Thin adapter over llm_wrapper; handles logging
    │
    ├── tools/
    │   └── Discrete_Elastic_Beams.py    # discreteBeam() — physics + wall contact
    │
    └── tests/
        ├── test_beam_chaining.py
        ├── test_beam_FDM.py
        ├── test_shape_checker.py
        └── test_all_shapes.py
```

---

## Output files

Every run writes to `logs/`. The stem is `RUN_NAME` if set, otherwise `run_<timestamp>_<pid>`:

| File | Description |
|---|---|
| `logs/<stem>.jsonl` | Streaming trajectory: one line per iteration (DOF snapshot, BC, error, timestamp, orchestrator prompt/reasoning) |
| `logs/<stem>.log` | Every LLM prompt and response (when `LOG_LLM_CALLS = True`) |

Set `RUN_NAME` in your config overlay so parallel runs produce paired, identifiable files:
```python
# experiments/qwen_plan_C.py
MODEL             = "qwen2.5:14b"
ORCHESTRATOR_MODE = "plan"
RUN_NAME          = "qwen_plan_C"
```
```bash
python3 cli.py --target C --config experiments/qwen_plan_C.py
# → logs/qwen_plan_C.jsonl + logs/qwen_plan_C.log
```

`render_3D.py` reads the run JSONL and writes:

| File | Description |
|---|---|
| `viz/index.html` | mechview viewer — open in Chrome |
| `viz/data.js` | Animation data in mechview format |

---

## LLM Layer

Provider implementations live in the [`llm_wrapper`](https://github.com/agentic-writing/llm_wrapper) package (installed via pip). `src/llm/llm.py` is a thin adapter that reads provider/model from config and delegates to it. When `LOG_LLM_CALLS = True`, every prompt and response is appended to `logs/llm_<timestamp>.log`.

---

## Visualization

**3D interactive rendering:** Post-run 3D animation is powered by [`mechview`](https://github.com/StructuresComp/mechview) (installed via pip). `render_3D.py` calls `src/visualization/mechview_export.py` to normalise coordinates and write `data.js`, then copies `index.html` from the installed mechview package into `viz/`.

**2D analysis plots:** `analysis_plots.py` reads the streaming JSONL trajectory file and generates error curves, trajectory animations, and iteration-by-iteration shape evolution. Use this to inspect control decisions and convergence behavior.

---

## Physics Tool: `discreteBeam`

`src/tools/Discrete_Elastic_Beams.py` — implicit time-integration of a 2D discrete elastic beam.

**Signature:**
```python
discreteBeam(physical_system, q0=None, u0=None,
             control_position, control_angle,
             t_start=0.0, dt=0.01, totalTime=None,
             plotImage=0, plotStep=5, save_stride=1) -> dict
```

**Key behaviours:**

- `q0=None` / `u0=None` → self-initialises to straight-vertical / zero velocity.
- `totalTime=None` → computed from `max(delta_pos / BOUNDARY_VELOCITY, delta_angle / BOUNDARY_ANGLE_VELOCITY)`.
- BC interpolates linearly from current position/angle to target over `totalTime`.
- `t_start` offsets cumulative display time for chaining runs; physics is unaffected.
- Returns `{status, message, q_new, u_new, ctime, q_history}`.

**Wall behaviour (set via `physical_system.walls`):**

Walls are **axis-aligned** line segments defined by `{"P0": [x0,y0], "P1": [x1,y1]}` — each segment must be either horizontal (same y) or vertical (same x). Diagonal walls are not supported. The outward normal is the edge vector `(P1−P0)` rotated 90° CCW, which for axis-aligned walls reduces to one of ±X or ±Y and encodes which side is allowed. Use large sentinel coordinates (e.g. `±10`) to approximate an infinite wall.

- Compatibility check: `discreteBeam` raises `ValueError` at startup if any free node in `q0` already penetrates a wall. Fix `q0` or wall positions before calling.
- Contact is resolved with a **predictor/corrector** scheme: the predictor detects new contacts; the corrector re-solves with those nodes pinned to the wall surface.
- `is_wall_contact[i]` tracks which wall each node is pinned to (−1 = free). Nodes 0 and 1 are never subject to wall logic.

**No-wall behaviour (`walls == []`):**

The tip nodes (nv−2 and nv−1) are **also fixed** throughout the run. Without this, moving node 0 simply translates the beam under gravity — a trivial result. With both ends clamped, driving node 0 produces genuine elastic bending.

---

## Benchmark System

Each benchmark is a subclass of `BenchmarkCase` in `src/benchmarks/cases/`. Adding a new case = one new file in `cases/`. Nothing else changes.

- `loss_fn(nodes, ps, q_ref) -> dict` — deterministic; returns at minimum `{error_percent, diagnostics}`
- `reference_fn(x0, y0, theta) -> list` — ideal target shape anchored at node 0's pose
- `success_threshold` — `error_percent` below which the task is solved

See `docs/CONTRIBUTING.md` for a complete guide to adding new benchmark cases.

Standalone visualiser for a single shape:
```bash
python3 -m src.benchmarks.cases.beam_C
```

Full registry validation:
```bash
python3 -m src.tests.test_all_shapes
```

---

## Research Axes

1. **Plan vs. myopic** — does grounding in a prior strategy improve convergence?
2. **Observation abstraction** — does geometric reasoning outperform raw coordinates?
3. **Self-assessment accuracy** — can the LLM judge convergence correctly?
4. **LLM vs. VLM** — can vision outperform text for shape judgment?

The convergence decision is **intentionally left to the LLM**. The deterministic evaluator runs in parallel as ground truth, but the LLM's own convergence call is a measured quantity — its accuracy is part of what is being studied.
