# DLO Controller — TODO

Work top-to-bottom within each section. Later sections depend on earlier ones.

## 1. Immediate — needed before `cli.py --target snake_L` works

### 1a. `config.py` ✓
- [x] `WALLS = [...]` list replaces scalar `WALL_H` — done

### 1b. New benchmark: `src/benchmarks/cases/beam_snake_L.py` ✓
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

### 1e. Streaming architecture for long runs ✓
**Status: COMPLETED** (unified single JSONL instead of 3 separate files)

Problem: `all_configurations` grows unbounded in memory during long runs.
Solution: Stream configs incrementally to disk; keep only latest in state.

Implementation:
  - Single `run_<timestamp>_<pid>.jsonl` file (one line per DOF snapshot)
  - Each snapshot tagged with iteration, BC command, error_percent, timestamp
  - `ConfigWriter` class handles append-only I/O with auto-flush
  - Programmer stores q_history in state buffer; evaluator writes to disk
  - Removed all_configurations dict; state is now O(1) memory
  - postrun_visuals.py reads JSONL for plotting
  - Frame sequence export: `trajectory_frames/`, `orchestrator_iterations/`
  - Real-time animations with metadata (BC, error, iteration number)

Memory usage stays constant (O(1)) by design — write-flush-discard pattern proven.

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

## 4. Post-run visualization (`src/visualization/postrun_visuals.py`) ✓
**Status: COMPLETED**

Reads `logs/run_<timestamp>.jsonl` (streaming trajectory file) and produces diagnostic plots.

Implemented:
- [x] `plot_error_curve()` — error_percent vs iteration
- [x] `plot_trajectory_animation()` — animate all snapshots with fading
- [x] `plot_iteration_shapes()` — grid of final shape per iteration with BC label
- [x] `plot_single_iteration()` — detailed view of one iteration's snapshots
- [x] CLI: `python -m src.visualization.postrun_visuals --run run_<timestamp> --all`
  - `--run`: trajectory file stem
  - `--error-curve`, `--animate`, `--shapes`, `--iter N`: individual plots
  - `--output-dir`: save to directory instead of showing

**Note:** mechview (3D viewer) handles 3D rendering; postrun_visuals handles 2D analysis plots

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
