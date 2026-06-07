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

### 2. Benchmark case file rules — never violate these.

**A case file in `src/benchmarks/shapes/` is self-contained. It owns everything specific to that case.**

Each case is a class inheriting from `BenchmarkCase` (defined in `src/benchmarks/__init__.py`):

```python
class CShape(BenchmarkCase):
    name               = "C"          # string — registry key
    physics_type       = "beam"       # "beam" or "rod"
    target_description = "..."        # prose for the LLM prompt
    success_threshold  = 0.05         # optional, default 0.05

    def reference_fn(self, x0=0.0, y0=0.0, theta=0.0) -> list:
        # Generate ideal (x, y) nodes anchored at actual node 0's position
        # and slope.  Uses arc-length s ∈ [0,1] — invariant to node count.
        # Return None if this case has no reference geometry.

    def loss_fn(self, nodes, ps=None, q_ref=None) -> dict:
        # Return dict with at least {error_percent, diagnostics}.
        # May include multiple named sub-losses alongside error_percent.

register(CShape())
```

**`reference_fn(x0, y0, theta)` — frame-invariant reference geometry.**
- `x0, y0` — position of node 0 extracted from the actual `q` vector.
- `theta` — tangent direction at node 0 (0 = pointing straight down, −Y).
- The ideal shape integrates a `θ*(s)` profile starting from this exact pose.
- Not all cases need a reference shape — tip-target, contact, and topology cases return `None`.

**`loss_fn(nodes, ps, q_ref=None)` — deterministic error computation.**
- `nodes` — list of `(x, y)` observable positions from the actual `q`.
- `ps` — `PhysicalSystemBase` instance (beam or rod); may be `None` in standalone tests.
- `q_ref` — optional reference from `reference_fn()` for node-wise comparison.
- A case may compute and return multiple named sub-losses alongside `error_percent`.
- Never add case-specific logic to `losses.py`.

**`main()` — standalone visualizer.**
Every case file has a `main()` and a module-level `TEST_CONFIG = None`.
- `TEST_CONFIG = None` → run shows the ideal reference shape only.
- Set `TEST_CONFIG` to a hand-typed list of `(x, y)` tuples → run shows both
  reference and actual overlaid, prints all losses.
- Run: `python -m src.benchmarks.shapes.beam_C`

**`losses.py` is closed.** Contains only: `extract_turning_angles`, `build_ideal_from_edge_profile`, `profile_loss`, and the 7 atomic primitives (`l_turn`, `l_pos`, `l_angle`, `l_dist`, `l_sign`, `l_inflect`, `l_contact`). Never add case-specific logic there.

**`benchmarks/__init__.py` is closed.** Shape files are auto-discovered. Never add an explicit import line there.

**Adding a new case = one new file in `src/benchmarks/shapes/`. Nothing else changes.**

---

### 3. Agent file structure (strict order, separated by dashed comment headers)

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
2. Calls `benchmark.loss_fn(nodes, ps, q_ref)` against the latest observable configuration.
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
physical_system     object                          BeamPhysicalSystem or RodPhysicalSystem (built once, never modified)
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

**Class hierarchy:**
- `PhysicalSystemBase` — empty base class for `isinstance` checks and type hints.
- `BeamPhysicalSystem(PhysicalSystemBase)` — 2D elastic beam. `ndof = 2 * nv`. All fields are **instance attributes accessed via dot notation** (`ps.wall_H`, `ps.nv`, etc.) — never dict-style (`ps["wall_H"]`).
- `RodPhysicalSystem(PhysicalSystemBase)` — 3D elastic rod stub. `ndof = 3 * nv - 1`. Fields TBD when rod support lands.

**BeamPhysicalSystem fields:**
- `type = "beam"`, `nv`, `ndof = 2*nv`
- Raw: `rho`, `Y`, `r0`, `rod_length`, `visc`, `wall_H`, `friction`, `boundary_velocity`, `boundary_angle_velocity`
- Derived (computed once): `EI`, `EA`, `deltaL`, `m`, `W`, `C_diag`
- `observable_nodes` — node indices shown to orchestrator. Computed once from `NV` and `N_PARTIAL`. **Never changed during a run.**

**DOF layout for beams:** `q = [x0, y0, x1, y1, ..., x_{nv-1}, y_{nv-1}]`. Node `i` → `q[2i], q[2i+1]`. Extraction is a reshape — trivial, no helper needed.

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
    │   └── physical_system.py           # PhysicalSystemBase, BeamPhysicalSystem, RodPhysicalSystem + make_physical_system()
    │
    ├── agents/
    │   ├── orchestrator_agent.py        # planner_node + decision_node + build_orchestrator_graph()
    │   ├── programmer_agent.py          # programmer_node — deterministic, zero LLM calls
    │   └── evaluator_agent.py           # evaluator_node — looks up benchmark, runs loss_fn
    │
    ├── workflows/
    │   └── workflow.py                  # build_controller_graph() — master LangGraph
    │
    ├── visualization/
    │   ├── __init__.py
    │   └── run_time_visual.py           # plot_shape(), plot_comparison(), plot_grid()
    │
    ├── benchmarks/
    │   ├── __init__.py                  # re-exports BenchmarkCase; register() + get_benchmark(); auto-discovers shapes/
    │   ├── case.py                      # BenchmarkCase ABC
    │   ├── losses.py                    # Primitives: extract_turning_angles, build_ideal_from_edge_profile,
    │   │                                #   profile_loss, l_turn, l_pos, l_angle, l_dist, l_sign, l_inflect, l_contact — CLOSED
    │   └── shapes/                      # One file per case — self-contained, self-registering
    │       ├── beam_C.py                # class CShape(BenchmarkCase): profile/fit fns, reference_fn, loss_fn, main()
    │       ├── beam_U.py                # class UShape(BenchmarkCase)
    │       ├── beam_S.py                # class SShape(BenchmarkCase)
    │       └── beam_L.py                # class LShape(BenchmarkCase)
    │
    ├── llm/
    │   ├── llm.py                       # call_llm(), call_llm_with_tool(), get_langchain_llm()
    │   └── providers/                   # anthropic, openai, ollama, fake
    │
    ├── tools/
    │   ├── Discrete_Elastic_Beams.py    # discreteBeam() — physics + wall contact + BC interpolation
    │   ├── Discrete_Elastic_Rods.py     # discreteRod() — 3D rod physics (in progress)
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

Each benchmark is a subclass of `BenchmarkCase` (ABC in `src/benchmarks/__init__.py`).

**Required class attributes** (validated at class definition time):
- `name` — string key matching `state["target_shape"]` (e.g. `"C"`)
- `physics_type` — `"beam"` or `"rod"`
- `target_description` — rich description for the LLM prompt

**Optional class attribute:** `success_threshold = 0.05` (error_percent below which task is solved)

**Abstract method:** `loss_fn(self, nodes, ps=None, q_ref=None) -> dict`
- Returns at minimum `{error_percent, diagnostics}`. May include multiple sub-losses.
- `nodes` — observable `(x, y)` positions. `ps` — PhysicalSystemBase. `q_ref` — from reference_fn.

**Optional override:** `reference_fn(self, x0=0.0, y0=0.0, theta=0.0) -> list[(x,y)] | None`
- Anchors ideal shape at `(x0, y0)` with slope `theta` (0 = pointing down, −Y).
- `x0, y0, theta` come from the actual beam's node 0 — frame-invariant by construction.
- Shape is parameterised by arc-length `s ∈ [0, 1]` — invariant to node count.
- Returns `None` (base default) for cases with no reference geometry.

**`losses.py`** — closed. Contains: `extract_turning_angles`, `build_ideal_from_edge_profile`,
and `profile_loss` (θ-profile RMS fitting). Add a new primitive here only when a real case
needs it — never speculatively.

**Adding a new case:** create one file in `src/benchmarks/shapes/`. It self-registers.
Nothing else changes — `__init__.py` auto-discovers all `*.py` files in `shapes/`.

**Standalone visualizer:** `python -m src.benchmarks.shapes.beam_C`
- `TEST_CONFIG = None` → shows reference shape only.
- Set `TEST_CONFIG` to hand-typed `(x, y)` pairs → shows reference + actual overlaid, prints loss.

**Full registry test:** `python -m src.tests.test_all_shapes` — validates all registered benchmarks.

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
