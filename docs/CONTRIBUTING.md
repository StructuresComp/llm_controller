# Contributing

---

## Setup

```bash
git clone <repo>
cd llm_controller
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add your API key
```

---

## Running things

| What | Command |
|---|---|
| Full control loop | `python3 cli.py --target C` |
| Control loop, custom wall depth | `python3 cli.py --target C --H 0.15` |
| Visualize one benchmark case | `python3 -m src.benchmarks.shapes.beam_C` |
| Validate all benchmarks (0% error check) | `python3 -m src.tests.test_all_shapes` |
| Profile-fitting round-trip tests | `python3 -m src.tests.test_shape_checker` |
| Physics chaining test | `python3 -m src.tests.test_beam_chaining` |

All commands run from the project root with the venv active.

---

## Adding a new benchmark case

**One file. Nothing else changes.**

Create `src/benchmarks/shapes/beam_<name>.py`. The registry picks it up automatically.

### Template

```python
import numpy as np

from src.benchmarks import BenchmarkCase, register
from src.benchmarks.losses import build_ideal_from_edge_profile, profile_loss
from src.visualization.run_time_visual import plot_shape, plot_comparison


# ---------------------------------------------------------------------------
# THETA PROFILE + FIT
# Define the ideal tangent-angle curve θ*(s) for this shape.
# s_norm ∈ [0, 1] is normalised arc-length.  θ*(0) = 0 by convention
# (relative to the start tangent).
# ---------------------------------------------------------------------------

def _profile_myshape(s_norm: np.ndarray, param: float = 1.0) -> np.ndarray:
    # return the ideal relative turning angle at each s
    return param * s_norm   # example: linear ramp = constant curvature


def _fit_myshape(s_norm: np.ndarray, dtheta: np.ndarray) -> dict:
    # fit free params to the observed dtheta; return a dict matching _profile params
    denom = float(np.dot(s_norm, s_norm)) + 1e-12
    return {"param": float(np.dot(s_norm, dtheta) / denom)}


# ---------------------------------------------------------------------------
# CASE
# ---------------------------------------------------------------------------

class MyShape(BenchmarkCase):
    name               = "myshape"          # must match --target argument
    physics_type       = "beam"             # "beam" or "rod"
    target_description = "Describe the target shape in plain English for the LLM."
    success_threshold  = 0.05               # error_percent below which task is solved

    def reference_fn(self, x0: float = 0.0, y0: float = 0.0, theta: float = 0.0) -> list:
        # Generates ideal (x, y) nodes anchored at the actual beam's node 0.
        # x0, y0 = position of node 0 from q.  theta = tangent angle at node 0.
        return build_ideal_from_edge_profile(
            _profile_myshape, n=9, arc_length=0.10,
            x0=x0, y0=y0, theta=theta, param=1.0,
        )

    def loss_fn(self, nodes: list, ps=None, q_ref=None) -> dict:
        # Must return dict with at least {error_percent, diagnostics}.
        return profile_loss(nodes, _profile_myshape, _fit_myshape)


register(MyShape())


# ---------------------------------------------------------------------------
# STANDALONE TEST
# Set TEST_CONFIG to hand-typed (x, y) pairs to visualize loss on real data.
# Leave as None to just show the reference shape.
# ---------------------------------------------------------------------------

TEST_CONFIG = None   # e.g. [(0.0, 0.0), (0.01, -0.02), (0.03, -0.05), ...]


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

### Not using θ-profile fitting?

Some cases don't need a reference shape at all. Skip `reference_fn` and `_profile`/`_fit` entirely — just implement `loss_fn` with whatever logic you need:

```python
class TipReach(BenchmarkCase):
    name               = "tip_reach"
    physics_type       = "beam"
    target_description = "Drive the tip to (x*, y*)."

    def loss_fn(self, nodes, ps=None, q_ref=None) -> dict:
        tip = nodes[-1]
        target = (0.05, -0.10)
        err = ((tip[0] - target[0])**2 + (tip[1] - target[1])**2) ** 0.5
        return {
            "error_percent": min(err / 0.01 * 100, 100.0),
            "diagnostics":   f"tip at ({tip[0]:.4f}, {tip[1]:.4f}), target {target}",
        }
```

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
│   │   ├── losses.py             Shared loss machinery: profile_loss, extract_turning_angles,
│   │   │                         build_ideal_from_edge_profile
│   │   └── shapes/               One file per benchmark case — self-contained
│   │       ├── beam_C.py         C-shape (constant curvature arc)
│   │       ├── beam_U.py         U-shape (semicircle)
│   │       ├── beam_S.py         S-shape (sine curvature profile)
│   │       └── beam_L.py         L-shape (straight + sharp bend)
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
| `MODEL` | Which LLM to use |
| `ORCHESTRATOR_MODE` | `"plan"` (LLM writes strategy first) or `"myopic"` (reactive only) |
| `OBSERVATION_LEVEL` | `"positions"` (raw x,y) or `"geometric"` (angles + curvature) |
| `N_PARTIAL` | How many nodes the LLM sees |
| `MAX_ITERATIONS` | Hard stop |
| `WALL_H` | Wall depth (also overridable with `--H`) |
