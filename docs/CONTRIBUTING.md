# Contributing

---

## Setup

```bash
git clone <repo>
cd llm_controller
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # edit .env and add your API key
```

---

## Running things

| What | Command |
|---|---|
| Full control loop | `python3 cli.py --target C` |
| Control loop, custom wall depth | `python3 cli.py --target C --H 0.15` |
| Control loop, custom config | `python3 cli.py --target C --config experiments/my_run.py` |
| Visualize one benchmark case | `python3 -m src.benchmarks.cases.beam_C` |
| Validate all benchmarks (0% error check) | `python3 -m src.tests.test_all_shapes` |
| Profile-fitting round-trip tests | `python3 -m src.tests.test_shape_checker` |
| Physics chaining test | `python3 -m src.tests.test_beam_chaining` |

All commands run from the project root with the venv active.

---

## Adding a new benchmark case

The benchmark system has two layers. **Adding a case = one new file in `cases/`. Nothing else changes.**

```
src/benchmarks/
├── shapes/   — geometric profile definitions (NOT registered, no BenchmarkCase)
│   ├── profile_C.py       profile_c(s_norm, total_turn), fit_c(...)
│   ├── profile_S.py       profile_s(s_norm, amplitude), fit_s(...)
│   ├── profile_U.py       profile_u(s_norm, total_turn), fit_u(...)
│   ├── profile_L.py       profile_l(s_norm, total_turn, turn_at), fit_l(...)
│   ├── profile_tip_reach.py  tip_reach_loss(nodes, target_x, target_y, rod_length)
│   └── profile_custom.py  make_profile(kappa_fn) → (profile_fn, fit_fn)
└── cases/    — fully specified experiments (auto-discovered, self-registering)
    ├── beam_C.py          imports profile_C, registers "C"
    ├── beam_S.py          imports profile_S, registers "S"
    ├── beam_U.py          imports profile_U, registers "U"
    ├── beam_L.py          imports profile_L, registers "L"
    ├── beam_snake_L.py    imports profile_tip_reach, registers "snake_L"
    └── beam_tip_reach.py  imports profile_tip_reach, registers "tip_reach"
```

**`shapes/`** — pure geometry. A profile file exports two functions:
- `profile_fn(s_norm, **params) → ndarray` — ideal turning angle at each arc-length position
- `fit_fn(s_norm, dtheta) → dict` — fit free parameters to observed turning angles

`shapes/` files are **not** imported by `__init__.py` and do **not** call `register()`.

**`cases/`** — full experiments. A case file:
1. Imports profile functions from `shapes/`
2. Defines a `BenchmarkCase` subclass with `name`, `physics_type`, `target_description`, `loss_fn`, `reference_fn`
3. Calls `register(MyCase())` at module level
4. Has a `main()` standalone visualizer

`__init__.py` auto-discovers every `*.py` in `cases/` and imports them. Nothing else needs to change when you add a file.

### Template — profile-based case (C, S, U, L, or a new shape)

**Step 1** — add `src/benchmarks/shapes/profile_myshape.py`:

```python
import numpy as np

def profile_myshape(s_norm: np.ndarray, param: float = 1.0) -> np.ndarray:
    """Ideal turning angle at normalised arc-length s."""
    return param * s_norm   # example: linear ramp = constant curvature

def fit_myshape(s_norm: np.ndarray, dtheta: np.ndarray) -> dict:
    """Fit free params to observed turning angles."""
    denom = float(np.dot(s_norm, s_norm)) + 1e-12
    return {"param": float(np.dot(s_norm, dtheta) / denom)}
```

**Step 2** — add `src/benchmarks/cases/beam_myshape.py`:

```python
import numpy as np
from src.benchmarks import BenchmarkCase, register
from src.benchmarks.profile_fitting import build_ideal_from_edge_profile, profile_loss
from src.benchmarks.shapes.profile_myshape import profile_myshape, fit_myshape
from src.visualization.run_time_visual import plot_shape, plot_comparison

class MyShape(BenchmarkCase):
    name               = "myshape"          # must match --target argument
    physics_type       = "beam"             # "beam" or "rod"
    target_description = "Describe the target shape in plain English for the LLM."
    success_threshold  = 0.05               # error_percent below which task is solved

    def reference_fn(self, x0=0.0, y0=0.0, theta=0.0) -> list:
        return build_ideal_from_edge_profile(
            profile_myshape, n=9, arc_length=0.10,
            x0=x0, y0=y0, theta=theta, param=1.0,
        )

    def loss_fn(self, nodes, ps=None, q_ref=None) -> dict:
        return profile_loss(nodes, profile_myshape, fit_myshape)

register(MyShape())

TEST_CONFIG = None

def main():
    case = MyShape()
    ref  = case.reference_fn()
    if TEST_CONFIG is None:
        plot_shape(ref, name="myshape (reference)")
    else:
        result = case.loss_fn(TEST_CONFIG)
        print(f"error: {result['error_percent']:.2f}%  |  {result['diagnostics']}")
        plot_comparison(ref, TEST_CONFIG, result, name="myshape")

if __name__ == "__main__":
    main()
```

### Template — tip-reach / navigation case (no reference shape)

For cases where success is defined by a goal position rather than a curvature profile, skip `shapes/` entirely and implement `loss_fn` directly in the case file. Use `tip_reach_loss` from `profile_tip_reach.py` if the metric is just Euclidean tip distance:

```python
from src.benchmarks import BenchmarkCase, register
from src.benchmarks.shapes.profile_tip_reach import tip_reach_loss

TARGET_X, TARGET_Y = 0.05, -0.10

class MyNavCase(BenchmarkCase):
    name               = "my_nav"
    physics_type       = "beam"
    target_description = "Navigate the tip to (0.05, -0.10) m."

    # Case-specific walls (optional — omit if no walls)
    walls = [{"P0": [x0, y0], "P1": [x1, y1]}, ...]

    def reference_fn(self, x0=0.0, y0=0.0, theta=0.0):
        return None   # no reference shape for goal-reaching tasks

    def loss_fn(self, nodes, ps=None, q_ref=None) -> dict:
        rod_length = ps.rod_length if ps is not None else 0.1
        return tip_reach_loss(nodes, TARGET_X, TARGET_Y, rod_length)

register(MyNavCase())
```

`test_all_shapes` automatically skips the round-trip check for cases where `reference_fn()` returns `None`.

---

## Custom controller geometry via `controlled_nodes`

`discreteBeam` has a concept of **controlled nodes** — node indices that are always
held fixed and are never subject to wall contact logic.

**Default behaviour** (set automatically by `discreteBeam`):

| Scenario | Controlled nodes |
|---|---|
| `walls` is non-empty | `[0, 1]` — only the driven end |
| `walls == []` | `[0, 1, nv-2, nv-1]` — driven end **and** the tip |

The no-wall case clamps the tip so that moving node 0 produces genuine elastic
bending rather than rigid-body translation under gravity.

**What `controlled_nodes` means internally:**
- Their DOFs are always in `fixed_index` — the Newton solver never adjusts them.
- The wall contact loop (`apply_wall_contacts`) skips them entirely.
- `fixed_index` is derived as `[2*n, 2*n+1 for n in controlled_nodes]` plus any
  dynamically contacted wall nodes.

**If you extend the physics tool** and need a different set of always-fixed nodes,
pass `controlled_nodes` explicitly. Validation is deliberately omitted — if a
controlled node overlaps with a wall, that is treated as a configuration error
and is the caller's responsibility.

---

## Folder structure

```
llm_controller/
│
├── cli.py                        Entry point — builds the graph, runs the loop
├── src/
│   │
│   ├── core/
│   │   ├── config.py             All tunable settings (model, NV, wall depth, etc.)
│   │   └── paths.py              Project-root-relative paths for data/logs
│   │
│   ├── models/
│   │   ├── physical_system.py    BeamPhysicalSystem + RodPhysicalSystem classes
│   │   └── state_llm_control.py  ControllerState TypedDict — shared across all agents
│   │
│   ├── agents/
│   │   ├── orchestrator_agent.py LLM brain: reads current shape, decides next BC
│   │   ├── programmer_agent.py   Deterministic: runs the physics sim, updates state
│   │   └── evaluator_agent.py    Deterministic: looks up benchmark, computes loss
│   │
│   ├── workflows/
│   │   └── workflow.py           Wires agents into a LangGraph control loop
│   │
│   ├── llm/
│   │   └── llm.py                call_llm() and call_llm_with_tool() — provider-agnostic
│   │
│   ├── tools/
│   │   ├── Discrete_Elastic_Beams.py  discreteBeam() — 2D beam physics simulator
│   │   └── Discrete_Elastic_Rods.py   discreteRod()  — 3D rod physics (in progress)
│   │
│   ├── benchmarks/
│   │   ├── case.py               BenchmarkCase ABC — base class for all cases
│   │   ├── __init__.py           Registry: register(), get_benchmark(), auto-discovery
│   │   ├── profile_fitting.py    Shared θ-profile math: build_ideal_from_edge_profile, profile_loss
│   │   │                         build_ideal_from_edge_profile
│   │   ├── shapes/               Geometric profile definitions — NOT registered
│   │   │   ├── profile_C.py      profile_c, fit_c — constant curvature (circular arc)
│   │   │   ├── profile_S.py      profile_s, fit_s — full-sine (S-curve)
│   │   │   ├── profile_U.py      profile_u, fit_u — linear ramp to π (semicircle)
│   │   │   ├── profile_L.py      profile_l, fit_l — step function (straight + sharp bend)
│   │   │   ├── profile_tip_reach.py  tip_reach_loss() — Euclidean tip distance
│   │   │   └── profile_custom.py make_profile(kappa_fn) — wrap any user function
│   │   └── cases/                Fully specified experiments — auto-discovered, registered
│   │       ├── beam_C.py         C-shape case (imports profile_C)
│   │       ├── beam_S.py         S-shape case (imports profile_S)
│   │       ├── beam_U.py         U-shape case (imports profile_U)
│   │       ├── beam_L.py         L-shape case (imports profile_L)
│   │       ├── beam_snake_L.py   L-tube navigation (imports profile_tip_reach)
│   │       └── beam_tip_reach.py Free-space tip reach (imports profile_tip_reach)
│   │
│   ├── visualization/
│   │   └── run_time_visual.py    plot_shape(), plot_comparison(), plot_grid()
│   │
│   └── tests/
│       ├── test_all_shapes.py    Round-trip check: reference_fn → loss_fn ≈ 0%
│       ├── test_shape_checker.py Profile-fitting tests + perturbation checks
│       ├── test_beam_chaining.py Physics tool integration test
│       └── test_beam_FDM.py      Finite-difference validation of beam physics
```

---

## Agent pipeline

```
cli.py
  │
  └─▶ build_controller_graph()         (workflows/workflow.py)
          │
          ▼
    ┌─────────────────────────────────────────────────────┐
    │  ORCHESTRATOR sub-graph                             │
    │    planner_node  — writes a multi-step strategy     │
    │                    (plan mode only, runs once)      │
    │    decision_node — reads current shape + history,   │
    │                    calls LLM, outputs next BC       │
    │                    {x0, y0, theta}                  │
    └─────────────────────┬───────────────────────────────┘
                          │
                          ▼
                  programmer_node          (pure Python, no LLM)
                    Runs discreteBeam() with the new BC.
                    Updates q, shape_history, bc_history.
                          │
                          ▼
                  evaluator_node           (pure Python, no LLM)
                    Calls benchmark.loss_fn(nodes, ps).
                    Writes error_percent to state.
                          │
                    converged?  ──yes──▶  END
                          │no
                          └──────────────▶  ORCHESTRATOR (next cycle)
```

**What the LLM sees each cycle:**
- Last K observable node positions (or geometric descriptors, depending on `OBSERVATION_LEVEL`)
- Last K boundary conditions it issued
- The evaluator's error_percent from the previous cycle
- Its own prior plan (plan mode only)

**What the LLM outputs:**
- `{x0, y0, theta}` — the next boundary condition (position + slope of the held end)
- A convergence judgment (recorded separately as a research measurement)

**Key config knobs** (`src/core/config.py`):
| Flag | What it does |
|---|---|
| `RUN_NAME` | Stem for `logs/<RUN_NAME>.json` + `logs/<RUN_NAME>.log`. Empty = auto `run_<timestamp>_<pid>` |
| `MODEL` | Which LLM to use |
| `ORCHESTRATOR_MODE` | `"plan"` (LLM writes strategy first) or `"myopic"` (reactive only) |
| `OBSERVATION_LEVEL` | `"positions"` (raw x,y) or `"geometric"` (angles + curvature) |
| `N_PARTIAL` | How many nodes the LLM sees |
| `MAX_ITERATIONS` | Hard stop |
| `WALLS` | Wall geometry (floor depth also overridable with `--H`) |

**Config overlay files** let you vary settings per-run without touching `config.py`. The overlay file is plain Python — only define what you want to change:

```python
# experiments/qwen_myopic_U.py
MODEL             = "qwen2.5:14b"
ORCHESTRATOR_MODE = "myopic"
RUN_NAME          = "qwen_myopic_U"
```

```bash
python3 cli.py --target U --config experiments/qwen_myopic_U.py
```

This is the intended pattern for the evaluation harness (TODO §5): spawn one subprocess per `(shape, model, mode)` combination, each with its own overlay file and `RUN_NAME`.
