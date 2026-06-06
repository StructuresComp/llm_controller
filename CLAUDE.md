# LLM Controller — CLAUDE.md

---

## Non-Negotiable Engineering Rules

### 1. Never trust an LLM to produce structured output without a tool schema.

**Always use `call_llm_with_tool(prompt, tool_schema, tool_name)` when the output
must be parsed by code.** Never use `call_llm` and then `json.loads()` on the result.

LLMs are non-deterministic. Even with explicit JSON instructions, they will occasionally
add markdown fences, prose, trailing commas, or wrong field names. Tool schemas force
the model to emit a validated structure via the provider's native tool-calling API —
this is the only reliable way to get machine-readable output from an LLM.

**In practice:**
- Orchestrator BC decision → `call_llm_with_tool` with `_BC_SCHEMA` → Python dict
- Convergence assessment → `call_llm_with_tool` with `_CONVERGED_SCHEMA` → Python dict
- Free-text (planning, summaries) → `call_llm` only (text, never parsed by code)

### 2. Agent file structure (strict order, separated by dashed comment headers)

Every agent file must follow this exact section order:

```
1. Imports      — stdlib, langgraph, src.llm, src.models, src.benchmarks, src.core.config
2. Schemas      — tool schema dicts for call_llm_with_tool; Pydantic models if needed
3. Prompts      — module-level string constants (NEVER inline inside node functions)
4. Helpers      — pure functions called by nodes; one-line docstring each
5. Nodes        — LangGraph node functions: (state: ControllerState) -> dict
6. Graph builder— build_X_graph() if the agent has a sub-graph; omit if single node
7. Main         — main() for standalone testing
8. Entry point  — if __name__ == "__main__": main()
```

---

## What This Project Is

**Research question:** How well can LLMs (and later VLMs) perform as controllers
in physical manipulation tasks?

The framework implements a closed-loop control pipeline for deformable linear objects
(DLO — elastic beams and rods). The LLM acts as the controller: it observes the current
configuration, reasons about the physics, and issues boundary condition updates to
steer the beam toward a target shape. The target shapes (C, U, S, L, etc.) are
benchmarks — the real subject of study is the LLM's ability to plan, adapt, and
recognize success in a physical system it cannot directly see or touch.

**Research axes:**
- LLM plan mode vs. myopic mode — does grounding in a prior strategy help?
- LLM vs. VLM — does seeing a rendered image outperform reading geometric descriptors?
- Convergence rate, iteration count, backtracking frequency — quantitative evaluation

**The convergence judgment is intentionally left to the LLM/VLM.** The deterministic
evaluator runs in parallel as ground truth and controls termination, but the LLM's
own convergence assessment is recorded as a separate measured quantity — its accuracy
at recognizing success is part of what is being studied.

**Physical setup:**
- DLO initially hangs straight down along −Y from the origin.
- X is horizontal, Y is vertical. Gravity acts along −Y.
- A rigid wall sits at y = WALL_H (negative). Beam length L < |WALL_H|.
- The agent controls boundary condition (position + slope angle) at the held end.
- Wall contact mechanics (predictor/corrector, `is_wall_contact` flag per node) live in the physics tool.

---

## Architecture

```
[START] → orchestrator sub-graph → programmer_node → evaluator_node → (done?) → [END]
                    ↑__________________________________________________|
```

### Orchestrator sub-graph (`build_orchestrator_graph()`)
```
planner_node → decision_node → END
```
- **`planner_node`**: In "plan" mode and `state["plan"] == ""` → calls LLM to write a
  multi-step strategy, stores it in `state["plan"]`. No-op in all other cases.
- **`decision_node`**: Always runs. Builds a prompt from the last K (shape, BC) pairs,
  appends `state["plan"]` if non-empty, includes `state["eval_result"]` (geometric ground
  truth from previous cycle), and calls `call_llm_with_tool` with `_BC_SCHEMA` to produce
  `state["decision"]`. Also calls LLM convergence assessment to update `state["converged"]`.

### Programmer node (`programmer_node`)
Pure deterministic Python — zero LLM calls.
1. Reads `state["decision"]` (validated dict `{x0, y0, theta}`).
2. Calls `discreteBeam(physical_system, q0_current, u0_current, t_start=ctime_current, ...)`.
3. On failure: writes `state["error"]`, returns early.
4. Updates: `q0_current`, `u0_current`, `ctime_current`, `bc_current`, `all_configurations`,
   `shape_history` (observable nodes only), `bc_history`, `iteration_log`.
5. Writes a deterministic text report to `state["result"]`.

### Evaluator node (`evaluator_node`)
Pure deterministic Python — zero LLM calls.
1. Looks up the active benchmark in the registry via `get_benchmark(state["target_shape"])`.
2. Calls `benchmark.loss_fn(nodes, ps)` against the latest observable configuration.
3. Writes result to `state["eval_result"]` (error_percent, diagnostics, etc.).
4. The routing function `_should_continue` reads `state["converged"]` (LLM judgment)
   and `state["error"]` to decide whether to loop or end.

### Orchestrator observation levels
Controlled by `OBSERVATION_LEVEL` in `src/core/config.py`:

**`"positions"`** (default) — raw (x, y) coordinates of the N_PARTIAL observable nodes.

**`"geometric"`** — positions + tangent angles + curvature signs + total turning angle.
Tangent angles in radians, counter-clockwise from the −Y axis.

**`"full"`** (reserved) — geometric + wall contact status per node. Not yet implemented.

---

## State: `ControllerState`

Defined in `src/models/state_llm_control.py`.

```
messages            Annotated[list, add_messages]   LangGraph message history
physical_system     dict                            PhysicalSystem (built once, never modified)
q0_current          list[float]                     Full DOF vector at end of last tool call
u0_current          list[float]                     Velocity vector at end of last tool call
ctime_current       float                           Cumulative simulation time (t_start for next call)
bc_current          list[float]                     [x0, y0, theta] — active BC at end of last run
bc_history          list                            Last K BCs issued
shape_history       list                            Last K observable configurations (N_PARTIAL nodes)
all_configurations  dict[int, list[float]]          All q_new snapshots; downsampled at MAX_CONFIGS
target_shape        str                             "C", "U", "S", etc. — from CLI --target
converged           bool                            LLM's own convergence judgment
eval_result         dict                            Deterministic evaluator output: error_percent, etc.
decision            dict                            {x0, y0, theta} — schema-validated BC decision
plan                str                             High-level strategy (plan mode); "" in myopic
result              str                             Programmer's deterministic text report
iteration_log       list                            One entry per completed cycle
error               str | None
```

---

## PhysicalSystem

Defined in `src/models/physical_system.py`. Built once at startup by `make_physical_system(config)`.
Stored in state and passed directly to physics tools. Never modified during a run.

**Key fields:**
- Raw: `nv`, `rho`, `Y`, `r0`, `rod_length`, `visc`, `wall_H`, `friction`,
  `boundary_velocity`, `boundary_angle_velocity`
- Derived (computed once): `EI`, `EA`, `deltaL`, `m`, `W`, `C_diag`
- `observable_nodes` — node indices shown to orchestrator. Computed once from `NV` and
  `N_PARTIAL`: node 0, `N_PARTIAL−2` interior nodes (linspace), node `nv−1`.
  **Set at initialization, never changed during a run.**

---

## Config (`src/core/config.py`)

### Section 1 — LLM
| Flag | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `"anthropic"` | Provider for all LLM calls |
| `MODEL` | `"claude-haiku-4-5"` | Model used for all calls (planning, BC decision, convergence) |
| `MAX_TOKENS` | `2048` | Hard upper bound on output tokens per call |

### Section 2 — Simulation & Visualization
| Flag | Default | Meaning |
|---|---|---|
| `NV` | 11 | Number of beam vertices |
| `RHO`, `Y`, `R0`, `ROD_LENGTH`, `VISC` | see config | Physical parameters |
| `WALL_H` | -0.12 | Wall position: y = WALL_H (m, negative) |
| `BOUNDARY_VELOCITY` | 0.02 | Translational speed of node 0 (m/s) |
| `BOUNDARY_ANGLE_VELOCITY` | 0.1 | Rotational speed of node 0 (rad/s) |
| `FRICTION` | True | True = wall fixes x+y; False = wall fixes y only |
| `PLOT_IMAGE` | 1 | 1 = show beam plots during simulation, 0 = suppress |

### Section 3 — Agent Perception & Control
| Flag | Default | Meaning |
|---|---|---|
| `OBSERVATION_LEVEL` | `"positions"` | `"positions"` / `"geometric"` / `"full"` (reserved) |
| `N_PARTIAL` | 5 | Observable nodes shown to orchestrator (set once at init) |
| `K_HISTORY` | 5 | Rolling window of (shape, BC) pairs |
| `ORCHESTRATOR_MODE` | `"plan"` | `"plan"` (grounded) or `"myopic"` (reactive) |
| `ALLOW_REPLAN` | False | If True, stagnation clears the plan for regeneration |
| `MAX_ITERATIONS` | 50 | Hard stop on control cycles |
| `MAX_CONFIGS` | 1000 | Cap on stored configurations (downsampled beyond this) |

CLI overrides: `--target`, `--H`.

---

## Project Structure

```
llm_controller/
├── cli.py                               # Entry point: python3 cli.py --target C
├── requirements.txt                     # Dependencies (to be replaced by pyproject.toml)
├── docs/
│   ├── cases.md                         # Task definitions, losses, metrics, protocol
│   └── paper_outline.md                 # Paper structure notes
│
└── src/
    ├── core/
    │   ├── config.py                    # All tuneable settings
    │   └── paths.py                     # PROJECT_ROOT, DATA_DIR, VISUALIZATION_DIR, LOGS_DIR
    │
    ├── models/
    │   ├── state_llm_control.py         # ControllerState TypedDict + initial_controller_state()
    │   └── physical_system.py           # PhysicalSystem TypedDict + make_physical_system()
    │
    ├── agents/
    │   ├── orchestrator_agent.py        # planner_node + decision_node + build_orchestrator_graph()
    │   ├── programmer_agent.py          # programmer_node — deterministic, zero LLM calls
    │   └── evaluator_agent.py           # evaluator_node — looks up benchmark, runs loss_fn
    │
    ├── workflows/
    │   └── workflow.py                  # build_controller_graph() — master LangGraph
    │
    ├── benchmarks/
    │   ├── __init__.py                  # Benchmark dataclass + register() + get_benchmark()
    │   ├── losses.py                    # Shared loss primitives (profile_loss, etc.)
    │   ├── visualization.py             # plot_shape(), plot_grid()
    │   └── shapes/
    │       ├── beam_C.py                # C-shape: reference_fn, loss_fn, main()
    │       ├── beam_U.py                # U-shape
    │       ├── beam_S.py                # S-shape
    │       └── beam_L.py                # L-shape
    │
    ├── llm/
    │   ├── llm.py                       # call_llm(), call_llm_with_tool(), get_langchain_llm()
    │   └── providers/                   # anthropic, openai, ollama, fake
    │
    ├── tools/
    │   ├── Discrete_Elastic_Beams.py    # discreteBeam() — physics + wall contact + BC interpolation
    │   ├── Discrete_Elastic_Rods.py     # discreteRod() — 3D rod physics (in progress)
    │   ├── shape_checker.py             # compute_shape_error(), build_ideal_from_edge_profile()
    │   └── getConfiguration.py
    │
    └── tests/
        ├── test_beam_chaining.py        # Physics tool integration test
        ├── test_beam_FDM.py             # Finite-difference validation
        ├── test_shape_checker.py        # Shape checker unit test + perturbation tests
        └── test_all_shapes.py           # Registry-wide benchmark validation + grid plot
```

---

## Benchmark System

Each benchmark is a `Benchmark` dataclass registered in `src/benchmarks/__init__.py`:
- `name` — string key matching `state["target_shape"]` (e.g. `"C"`)
- `physics_type` — `"beam"` or `"rod"`
- `target_description` — rich description for the LLM prompt
- `loss_fn(nodes, ps=None) -> dict` — deterministic loss; wraps `shape_checker.py`
- `reference_fn(n, arc_length) -> list` — generates the ideal shape at 100% accuracy
- `success_threshold` — error_percent below which the task is solved

**Adding a new shape:** create one file in `src/benchmarks/shapes/`, register the
`Benchmark` object, add the import to `src/benchmarks/__init__.py`. Nothing else changes.

**Standalone test:** `python -m src.benchmarks.shapes.beam_C` plots the ideal shape
and prints the error (expect ~0%).

**Full registry test:** `python -m src.tests.test_all_shapes` validates all registered
benchmarks and shows a grid plot.

---

## Physics Tool: `discreteBeam`

**Signature:** `discreteBeam(physical_system, q0=None, u0=None, control_position, control_angle, t_start=0.0, dt=0.01, totalTime=None, plotImage=0, plotStep=5) -> dict`

**Key behaviours:**
- `q0=None` / `u0=None` → self-initialises to straight-vertical / zero velocity
- `totalTime=None` → computed as `max(delta_pos / BOUNDARY_VELOCITY, delta_angle / BOUNDARY_ANGLE_VELOCITY)`
- BC interpolates linearly from current position to target over `totalTime`
- Wall contact via predictor/corrector: `is_wall_contact[i]` flag per node; nodes 0 and 1 never touched by wall logic
- `t_start` offsets cumulative display time for chaining runs
- Returns: `{status, message, q_new, u_new, ctime}`

**What NOT to change:** `gradEb`, `hessEb`, `gradEs`, `hessEs`, `getFs`, `getFb`, `objfun` — physics helpers are correct and validated.

---

## LLM Layer

Provider-agnostic. Set `LLM_PROVIDER` and `MODEL` in `src/core/config.py` to switch providers or models.
Thin adapter in `src/llm/llm.py` delegates to `llm_wrapper` (installed via pip).

- **`call_llm(prompt, system, max_tokens)`** — free text. Used for planning only.
- **`call_llm_with_tool(prompt, tool_schema, tool_name, system, max_tokens)`** — forced structured output. Used for BC decisions and convergence assessment.
