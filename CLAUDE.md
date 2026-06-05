# ESL Agent Framework — CLAUDE.md

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
1. Imports      — stdlib, langgraph, src.llm, src.models, src.tools, config
2. Schemas      — tool schema dicts for call_llm_with_tool; Pydantic models if needed
3. Prompts      — module-level string constants (NEVER inline inside node functions)
4. Helpers      — pure functions called by nodes; one-line docstring each
5. Nodes        — LangGraph node functions: (state: ESLState) -> dict
6. Graph builder— build_X_graph() if the agent has a sub-graph; omit if single node
7. Main         — main() for standalone testing
8. Entry point  — if __name__ == "__main__": main()
```

---

## What This Project Is

**Research question:** How well can LLMs (and later VLMs) perform as controllers
in physical manipulation tasks?

The framework implements a two-agent control loop for a deformable linear object
(DLO / elastic beam). The LLM acts as the controller: it observes the current
configuration, reasons about the physics, and issues boundary condition updates to
steer the beam toward a target shape. The target shapes (C, U, S, L, etc.) are
benchmarks — the real subject of study is the LLM's ability to plan, adapt, and
recognize success in a physical system it cannot directly see or touch.

**Research axes:**
- LLM plan mode vs. myopic mode — does grounding in a prior strategy help?
- LLM vs. VLM — does seeing a rendered image outperform reading geometric descriptors?
- Convergence rate, iteration count, backtracking frequency — quantitative evaluation

**The convergence judgment is intentionally left to the LLM/VLM.** A deterministic
geometric check runs in parallel as ground truth for evaluation, but it does not
control termination. The LLM's ability (or failure) to recognize when it has achieved
the target IS part of what is being measured.

**Physical setup:**
- DLO initially hangs straight down along −Y from the origin.
- X is horizontal, Y is vertical. Gravity acts along −Y.
- A rigid wall sits at y = WALL_H (negative). Beam length L < |WALL_H|.
- The agent controls boundary condition (position + slope angle) at the held end (nodes 0 and 1).
- Wall contact mechanics (predictor/corrector, `is_wall_contact` flag per node) live in the tool.

---

## Architecture

```
[START] → orchestrator sub-graph → programmer_node → (converged or error?) → [END]
                    ↑____________________________________________|
```

### Orchestrator sub-graph (`build_orchestrator_graph()`)
```
planner_node → decision_node → END
```
- **`planner_node`**: In "plan" mode and `state["plan"] == ""` → calls LLM to write a
  multi-step strategy and stores it in `state["plan"]`. No-op in all other cases.
- **`decision_node`**: Always runs. Builds a prompt from the last K (shape, BC) pairs,
  appends `state["plan"]` if non-empty, and calls `call_llm_with_tool` with `_BC_SCHEMA`
  to produce `state["decision"]`. Also calls `assess_convergence` to update `state["converged"]`.

### Programmer node (`programmer_node`)
Pure deterministic Python — zero LLM calls.
1. Reads `state["decision"]` (validated dict `{x0, y0, theta}`).
2. Calls `discreteBeam(physical_system, q0_current, u0_current, t_start=ctime_current, ...)`.
3. On failure: writes `state["error"]`, returns early.
4. Updates: `q0_current`, `u0_current`, `ctime_current`, `bc_current`, `all_configurations`,
   `shape_history` (observable nodes only), `bc_history`.
5. Writes a deterministic text report to `state["result"]`.

### Orchestrator observation levels
The orchestrator can receive the beam configuration at different levels of abstraction,
controlled by `OBSERVATION_LEVEL` in config:

**`"positions"`** (default) — raw (x, y) coordinates of the N_PARTIAL observable nodes.
The LLM infers shape from numbers alone.

**`"geometric"`** — positions + computed tangent angles at each observable node + curvature signs 
(+1 / 0 / -1 for local left/straight/right bends) + total turning angle from first to last node.
The LLM receives the language of mechanics and has an easier time reasoning about shape.
Tangent angles are in radians, counter-clockwise from the −Y axis.

**`"full"`** (reserved) — geometric + wall contact status per node + arc length distribution.
Not yet implemented; for future extension.

**Implementation:** descriptors are computed on the fly when the orchestrator builds its prompt,
dispatched by `_compute_descriptors()` and `_format_descriptors()` in orchestrator_agent.py.
Changing `OBSERVATION_LEVEL` in config requires no changes to state, programmer, or physics tool.

---

## State: `ESLState`

Defined in `src/models/state_llm_control.py`.

```
messages           Annotated[list, add_messages]  LangGraph message history
physical_system    dict                           PhysicalSystem (built once, never modified)
q0_current         list[float]                    Full DOF vector at end of last tool call
u0_current         list[float]                    Velocity vector at end of last tool call
ctime_current      float                          Cumulative simulation time (t_start for next call)
bc_current         list[float]                    [x0, y0, theta] — active BC at end of last run
bc_history         list                           Last K BCs issued
shape_history      list                           Last K observable configurations (N_PARTIAL nodes)
all_configurations dict[int, list[float]]         All q_new snapshots; downsampled at MAX_CONFIGS
target_shape       str                            "C", "U", "S", etc. — from CLI --target
converged          bool                           Set by orchestrator convergence assessment
decision           dict                           {x0, y0, theta} — schema-validated BC decision
plan               str                            High-level strategy (plan mode); "" in myopic
result             str                            Programmer's deterministic text report
error              str | None
messages           Annotated[list, add_messages]
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

## Config (`config.py`)

### Section 1 — LLM
| Flag | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER_FAST` | `"anthropic"` | Provider for fast-tier calls |
| `LLM_PROVIDER_SMART` | `"anthropic"` | Provider for smart-tier calls |
| `FAST_MODEL` | `"claude-haiku-4-5"` | Model for fast calls (BC decision, convergence) |
| `SMART_MODEL` | `"claude-haiku-4-5"` | Model for heavy reasoning (planning) |
| `MAX_TOKENS_FAST` | `512` | Token cap for fast calls |
| `MAX_TOKENS_SMART` | `2048` | Token cap for smart calls (planning) |

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

**Observation level** (controls what information the orchestrator receives):
| Flag | Default | Meaning |
|---|---|---|
| `OBSERVATION_LEVEL` | `"positions"` | `"positions"` = raw (x, y) coordinates only; `"geometric"` = coordinates + tangent angles + curvature signs + total turning angle; `"full"` = geometric + wall contact status (reserved for future) |

**Orchestrator perception & history:**
| Flag | Default | Meaning |
|---|---|---|
| `N_PARTIAL` | 5 | Number of observable nodes shown to orchestrator (computed once at init, never changed) |
| `K_HISTORY` | 5 | Rolling window of (shape, BC) pairs kept in history |

**Control loop:**
| Flag | Default | Meaning |
|---|---|---|
| `ORCHESTRATOR_MODE` | `"plan"` | `"plan"` (grounded) or `"myopic"` (reactive) |
| `ALLOW_REPLAN` | False | If True, stagnation clears the plan for regeneration |
| `MAX_ITERATIONS` | 50 | Hard stop on orchestrator → programmer cycles |
| `MAX_CONFIGS` | 1000 | Cap on stored configurations (downsampled beyond this) |

CLI overrides: `--target`, `--H`.

---

## Project Structure

```
agents_ESL/
├── cli.py                              # Entry point
├── config.py                           # All tuneable settings
├── paths.py                            # PROJECT_ROOT, DATA_DIR, VISUALIZATION_DIR, LOGS_DIR
├── TODO.md                             # Live task list
│
├── workflows/
│   └── workflow.py                     # build_esl_graph() — master LangGraph
│
└── src/
    ├── models/
    │   ├── state_llm_control.py        # ESLState TypedDict + initial_esl_state()
    │   └── physical_system.py          # PhysicalSystem TypedDict + make_physical_system()
    ├── agents/
    │   ├── orchestrator_agent.py       # planner_node + decision_node + build_orchestrator_graph()
    │   └── programmer_agent.py         # programmer_node (deterministic, zero LLM calls)
    ├── llm/
    │   ├── llm.py                      # call_llm(), call_llm_with_tool(), get_langchain_llm()
    │   └── providers/                  # anthropic, openai, ollama, fake
    ├── tools/
    │   └── Discrete_Elastic_Beams.py  # discreteBeam() — physics + wall contact + BC interpolation
    ├── tests/
    │   └── test_beam_chaining.py       # Physics tool integration test
    └── utils/
        └── tool_loader.py
```

---

## Physics Tool: `discreteBeam`

**Signature:** `discreteBeam(physical_system, q0=None, u0=None, control_position, control_angle, t_start=0.0, dt=0.01, totalTime=None, plotImage=0, plotStep=5) -> dict`

**Key behaviours:**
- `q0=None` / `u0=None` → self-initialises to straight-vertical / zero velocity
- `totalTime=None` → computed as `max(delta_pos / BOUNDARY_VELOCITY, delta_angle / BOUNDARY_ANGLE_VELOCITY)`
- BC interpolates linearly from current position to `control_position` / `control_angle` over `totalTime`
- Wall contact via predictor/corrector: `is_wall_contact[i]` flag per node; nodes 0 and 1 never touched by wall logic
- `t_start` offsets the cumulative display time for chaining
- Returns: `{status, message, q_new, u_new, ctime}`

**What NOT to change:** `gradEb`, `hessEb`, `gradEs`, `hessEs`, `getFs`, `getFb`, `objfun` — physics helpers are correct.

---

## LLM Layer

Provider-agnostic. Change two lines in `config.py` to switch providers.

- **`call_llm(prompt, model, system, max_tokens)`** — free text. Used for planning only.
- **`call_llm_with_tool(prompt, tool_schema, tool_name, model, system, max_tokens)`** — forced structured output. Used for BC decisions and convergence assessment.
- **`get_langchain_llm(tier)`** — LangChain-compatible model for future LangGraph tool nodes.
