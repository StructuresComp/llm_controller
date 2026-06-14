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

**Adding a case = one new file in `src/benchmarks/cases/`. Everything is self-contained.**

The directory structure:

```
src/benchmarks/
├── loss_beam/              — reusable loss functions and profiles
│   ├── profile_C.py        profile_c(s) — constant curvature
│   ├── profile_S.py        profile_s(s) — S-curve
│   ├── profile_tip_reach.py  tip_reach_loss(nodes, x_target, y_target, rod_length)
│   └── profile_custom.py   make_profile(kappa_fn) — wrap any curvature function
├── profile_fitting.py      — shared helpers: build_ideal_from_edge_profile, profile_loss
├── case.py                 — BenchmarkCase base class
├── __init__.py             — registry: register(), get_benchmark()
└── cases/                  — your benchmark cases (auto-discovered)
    ├── beam_C.py           ← new case goes here
    ├── beam_S.py
    └── beam_tip_reach.py
```

### Part 1: Define the loss function

A loss function takes observable nodes and returns `{error_percent, diagnostics}`. All cases implement `loss_fn(nodes, ps=None, q_ref=None) -> dict`.

**Three measurement approaches:**

1. **Shape-based loss** (measure how close observed shape is to ideal):

   To compute error, you need a **profile**: a function that returns turning angles (unitless curvature, dimensionless) vs. normalized arc-length s ∈ [0, 1].

   **Way 1: Provide a curvature function (or use existing)**

   Write a simple function that encodes your target shape. Example: constant curvature (circular arc):
   ```python
   def profile_fn(s, kappa=1.0):
       return kappa * s  # unitless turning angle at arc-length s
   ```

   Or import an existing profile:
   ```python
   from src.benchmarks.loss_beam.profile_C import profile_c, fit_c
   ```

   Create new profiles in `src/benchmarks/loss_beam/` following the pattern in `profile_C.py`, `profile_S.py`, etc. (each exports `profile_fn` and `fit_fn`).

   **Way 2: Extract profile from ideal nodes**

   If you have ideal (x, y) nodes defining your target shape, use `make_profile_from_nodes()` to automatically extract the turning-angle profile:

   See **`beam_sine.py`** for a complete worked example. In brief:
   ```python
   from src.benchmarks.profile_fitting import make_profile_from_nodes, profile_loss

   ideal_nodes = [(0, 0), (0.01, 0.015), (0.02, 0.025), ...]  # your target shape
   profile_fn, fit_fn = make_profile_from_nodes(ideal_nodes)

   def loss_fn(self, nodes, ps=None, q_ref=None):
       return profile_loss(nodes, profile_fn, fit_fn)
   ```

   Then register your case and it auto-discovers in the registry.

2. **Tip/nodal loss** (measure distance to target point(s)):

   Define a simple loss function. Example: measure distance from tip to target:
   ```python
   def loss_fn(self, nodes, ps=None, q_ref=None):
       tip_x, tip_y = nodes[-1]
       distance = ((tip_x - 0.05)**2 + (tip_y + 0.11)**2) ** 0.5
       return {"error_percent": min(distance * 100, 100.0), "diagnostics": f"tip at ({tip_x:.3f}, {tip_y:.3f})"}
   ```

   You can extend this to penalize multiple nodes. Or use the pre-built function:
   ```python
   from src.benchmarks.loss_beam.profile_tip_reach import tip_reach_loss

   def loss_fn(self, nodes, ps=None, q_ref=None):
       return tip_reach_loss(nodes, target_x=0.05, target_y=-0.11, rod_length=0.10)
   ```

3. **Combined loss** (blend shape error with tip/nodal error):

   Compute both, blend with a weight:
   ```python
   def loss_fn(self, nodes, ps=None, q_ref=None):
       shape_error = profile_loss(nodes, profile_c, fit_c)["error_percent"]
       tip_error = tip_reach_loss(nodes, target_x=0.05, target_y=-0.11, rod_length=0.10)["error_percent"]
       alpha = 0.7  # shape weight
       combined_error = alpha * shape_error + (1 - alpha) * tip_error
       return {"error_percent": combined_error, "diagnostics": "blended shape + tip loss"}
   ```

### Part 2: Define configs and register

Create one file in `src/benchmarks/cases/beam_mycase.py`. Look at **`beam_C.py`**, **`beam_S.py`**, or **`beam_sine.py`** for complete working examples.

**Step 1: Write target_description**

This description is passed directly to the LLM prompt. Make it clear and actionable:
```python
from src.benchmarks import BenchmarkCase, register

class MyCase(BenchmarkCase):
    name = "mycase"                # registry key; use with: python3 cli.py --target mycase
    physics_type = "beam"          # "beam" or "rod"
    target_description = (
        "Bend the beam into a smooth sine wave: starting straight, "
        "gradually curving up then down, returning to straight orientation."
    )
```

**Step 2: Define case-specific configs (optional)**

You can override any setting from `src/core/config.py` as class attributes:
```python
    # Optional: remove these if using defaults
    walls = [{"P0": [0, -0.12], "P1": [0.1, -0.12]}]  # case-specific walls
    rod_length = 0.12              # material length (m)
    nv = 25                        # number of vertices (mesh resolution)
    rho = 0.001                    # material density (kg/m)
    friction = True                # enable friction on walls
    success_threshold = 0.05       # error_percent below which task is solved
```

See `src/core/config.py` for the complete list of available parameters.

**Step 3: Implement loss_fn and reference_fn**

Choose your loss function from Part 1:
```python
    def loss_fn(self, nodes, ps=None, q_ref=None):
        return profile_loss(nodes, profile_c, fit_c)
    
    def reference_fn(self, x0=0.0, y0=0.0, theta=0.0):
        return build_ideal_from_edge_profile(profile_c, n=9, arc_length=0.10,
                                            x0=x0, y0=y0, theta=theta)
```

**Step 4: Register**

```python
register(MyCase())
```

**Step 5: Test and run**

From the project root:
```bash
# Run from CLI
python3 cli.py --target mycase

# Or test standalone
python -m src.benchmarks.cases.beam_mycase
```

The benchmark system auto-discovers every `*.py` in `cases/` — nothing else needs to change.

---

## Custom controller geometry via `controlled_nodes`

**Controlled nodes** are node indices that are always held fixed and never subject to wall contact logic.

**Default behaviour** (computed automatically in `make_physical_system()`):

| Scenario | Controlled nodes |
|---|---|
| `walls` is non-empty | `[0, 1]` — only the driven end |
| `walls == []` | `[0, 1, nv-2, nv-1]` — driven end **and** the tip |

The no-wall case clamps the tip so that moving node 0 produces genuine elastic bending rather than rigid-body translation under gravity.

**To use custom `controlled_nodes` in a benchmark case:**

1. Add it as a class attribute in your benchmark case:
   ```python
   class MyCase(BenchmarkCase):
       controlled_nodes = [0, 1, 5, 6]  # your custom always-fixed nodes
   ```

2. Modify `src/models/physical_system.py` in the `make_physical_system()` function to read and use the case-specific value instead of computing it automatically.

**What `controlled_nodes` means internally:**
- Their DOFs are always in `fixed_index` — the Newton solver never adjusts them.
- The wall contact loop (`apply_wall_contacts`) skips them entirely.
- `fixed_index` is derived as `[2*n, 2*n+1 for n in controlled_nodes]` plus any dynamically contacted wall nodes.

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
| `RUN_NAME` | Stem for `logs/<RUN_NAME>.jsonl` (streaming trajectory). Empty = auto `run_<timestamp>_<pid>` |
| `MODEL` | Which LLM to use (e.g., `"claude-haiku-4-5"`, `"gpt-4o"`, `"qwen2.5:14b"`) |
| `ORCHESTRATOR_MODE` | `"plan"` (LLM writes strategy first) or `"myopic"` (reactive only) |
| `OBSERVATION_LEVEL` | `"positions"` (raw x,y) or `"geometric"` (angles + curvature) |
| `N_PARTIAL` | How many observable nodes the LLM sees |
| `MAX_ITERATIONS` | Hard stop on control cycles |
| `WALLS` | Wall geometry (list of `{P0, P1}` segments). CLI override: `--H` for wall depth |

Any of these can be overridden at the benchmark case level (see Part 2).
