# DLO Controller — TODO

Work top-to-bottom within each section. Later sections depend on earlier ones.

---

## 0. Completed

- [x] PhysicalSystem class hierarchy — `BeamPhysicalSystem`, `RodPhysicalSystem` stub, dot access
- [x] `losses.py` atomic primitives — `l_turn`, `l_pos`, `l_angle`, `l_dist`, `l_sign`, `l_inflect`, `l_contact`
- [x] Benchmark registry — auto-discovery in `benchmarks/__init__.py`, `register()`, `get_benchmark()`
- [x] Benchmark cases — C, U, S, L with `reference_fn`, `loss_fn`, standalone visualizer
- [x] `test_all_shapes.py` — registry-wide validation and grid plot
- [x] `Discrete_Elastic_Beams.py` — multi-wall system, `_update_contacts` / `_build_fixed_index`, `cx`/`cy` contact state, `controlled_nodes`, section headers
- [x] `physical_system.py` — `walls` list replaces scalar `wall_H`; `_make_wall` converter
- [x] `mechview_export.py` — one obstacle box per wall, correct surface placement
- [x] `save_state.py` — run JSON written to `logs/`
- [x] Visualization — mechview-based 3D viewer via `visualize.py`

---

## 1. Immediate — needed before `cli.py --target snake_L` works

### 1a. `config.py` ✓
- [x] `WALLS = [...]` list replaces scalar `WALL_H` — done

### 1b. New benchmark: `src/benchmarks/shapes/beam_snake_L.py` ✓
- [x] `SnakeLCase`: L-tube navigation, tip-reach loss to `(0.05, -0.11)`
- [x] `target_description` describes tube geometry, corner, strategy hint
- [x] `loss_fn`: Euclidean tip distance normalised by rod length
- [x] No `reference_fn`
- [x] `success_threshold = 0.05`
- [x] `experiments/snake_L.py` config overlay wires walls + RUN_NAME

### 1c. Observation module ✓
- [x] `src/core/observation.py` — `describe_walls()`, `describe_nodes()`, `describe_history()`
- [x] `describe_walls()` classifies topology (L-tube, channel, etc.) before listing segments
- [x] Orchestrator imports from this module; private helpers removed from `orchestrator_agent.py`
- [x] `python -m src.core.observation` — standalone test with 3 hand-typed configs

### 1d. Run log — complete the record
- [ ] `iteration_log` entries include: BC issued `{x0, y0, theta}`, LLM raw reasoning text,
  LLM convergence judgment, evaluator `error_percent`, wall contact summary
- [ ] Top-level run JSON includes a `physical_system` block:
  walls in user-facing `{P0, P1}` format, `nv`, `rod_length`, `EI`, `EA`, `friction`
- [ ] `save_state.py` serializes the above; `mechview_export.py` already reads `walls`

---

## 2. Near-term benchmarks

- [ ] `beam_snake_T.py` — inverted T navigation: beam enters from top, must exit left or right
  - Harder: the LLM must commit to a side; ambiguous loss until commitment is made
  - Design the four walls, loss (tip-reach on chosen side), `target_description`

---

## 3. Discrete Elastic Rod support

- [ ] `RodPhysicalSystem` fields: `nv`, `rod_length`, `rho`, `r0`, `B`, `beta`,
  `EA`, `voronoi_lengths`, `m`, `W_vec`, `C_diag`, `observable_nodes`
- [ ] `make_rod_system(config)` factory
- [ ] `initial_rod_q0/u0` — packed `[x,y,z × N, θ × (N-1)]`
- [ ] `discreteRod` — wall support via same `cx`/`cy` contact abstraction; needs 3D extension
- [ ] `rod_programmer_node` — calls `discreteRod`, unpacks result, updates state
- [ ] `OBSERVATION_LEVEL` extended to 3D (Bishop frame / Darboux vectors)
- [ ] Rod benchmark shapes — helix, etc.
- [ ] `cli.py --mode beam|rod` flag

---

## 4. Analysis plots (`src/visualization/plot_run.py`)

Reads a `logs/run_<timestamp>.json` and produces static diagnostic figures.
No 3D rendering — that is mechview's job. This module is for paper figures and debugging.

- [ ] `plot_loss(log) -> Figure` — `error_percent` vs. iteration
- [ ] `plot_control(log) -> Figure` — `x0`, `y0`, `theta` each vs. iteration (3-panel)
- [ ] `plot_contact(log) -> Figure` — fraction of nodes in contact vs. iteration (if walls present)
- [ ] `plot_improvement_stability(log) -> Figure` — signed step delta `L_t - L_{t+1}` as a bar chart
- [ ] `main()` — CLI: `python -m src.visualization.plot_run logs/run_<timestamp>.json`
  saves all figures to `viz/` alongside mechview output

---

## 5. Evaluation harness

- [ ] Run N trials per (shape, model, orchestrator_mode)
- [ ] Record: iterations to convergence, final error %, LLM self-assessment accuracy
- [ ] Compare plan vs. myopic across all shapes and iteration budgets

---

## 5. VLM integration (distant — after LLM pipeline is proven)

- [ ] Pass rendered frame to VLM alongside target description
- [ ] New VLM provider in `src/llm/providers/`
- [ ] `call_llm_with_tool` extended to accept image inputs
- [ ] Orchestrator: VLM mode uses image history instead of coordinate text
- [ ] Benchmark: LLM (text) vs. VLM (image) on same shapes and iteration budget

---

## 6. Packaging

- [ ] `pyproject.toml` — `[project]` metadata, dependency list, CLI entry point
- [ ] Remove `requirements.txt` once validated with `pip install -e .`
