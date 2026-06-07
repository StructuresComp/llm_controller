# DLO Controller — TODO

Work top-to-bottom. Later items depend on earlier ones.

---

## 0. Completed refactoring

- [x] Rename ESL → DLO throughout (state, graph, CLI)
- [x] `src/core/` — moved `config.py` and `paths.py` out of project root
- [x] `src/workflows/` — moved `workflow.py` under `src/`
- [x] `docs/` — paper outline and cases spec moved here
- [x] Removed dead code: `tool_loader.py`, `tool_loader_YAML.py`, `configs/tools.yaml`
- [x] `evaluator_agent.py` — deterministic evaluator node in the graph
- [x] `src/benchmarks/` — Benchmark registry + C, U, S, L shape files with loss, reference, and visualization
- [x] `src/tests/test_all_shapes.py` — registry-wide validation and grid plot

---

## 1. Benchmark / loss refactoring (do first)

### 1a. PhysicalSystem class hierarchy

- [ ] Replace `PhysicalSystem` TypedDict with an empty `PhysicalSystemBase` class
- [ ] `BeamPhysicalSystem(PhysicalSystemBase)` — all current beam fields as instance attributes; add `ndof = 2 * nv`
- [ ] `RodPhysicalSystem(PhysicalSystemBase)` — stub; fields TBD when rod support lands
- [ ] `make_physical_system(config)` returns the right subclass based on `config.physics_type` (currently always beam)
- [ ] Migrate all dict-style `ps["field"]` access to dot access `ps.field` across: `Discrete_Elastic_Beams.py`, `programmer_agent.py`, `orchestrator_agent.py`, `cli.py`, `state_llm_control.py`

### 1b. losses.py — atomic primitives

- [ ] Delete `pose_loss` and `composite_loss` (case-specific, wrong place)
- [ ] Write the 7 atomic primitives from `cases.md`, each returning a normalised float:
  `l_turn`, `l_pos`, `l_angle`, `l_dist`, `l_sign`, `l_inflect`, `l_contact`
- [ ] Keep `profile_loss` as one valid loss option (wraps `shape_checker`)
- [ ] File is then closed — new cases compose these primitives, never add to this file

### 1c. Benchmark case files

- [ ] Delete `beam_pose.py`, `beam_S_impossible.py`, `beam_incompatible.py` (built on wrong foundation)
- [ ] Auto-discover shape files in `benchmarks/__init__.py` (glob `shapes/*.py`) — new case files self-register, no import line needed
- [ ] Remove `skip_reference_check` from `Benchmark` dataclass
- [ ] Revert `test_all_shapes.py` to clean state

### 1d. New case file protocol (one file, nothing else changes)

Each `shapes/beam_X.py` defines:
1. `reference_fn(x0, y0, theta) -> list[(x,y)]` — ideal DOF vector as nodes; visualise standalone
2. `loss_fn(nodes, ps, q_ref=None) -> dict` — compose atomic primitives; inline custom logic if case-specific
3. `Benchmark(name, physics_type, target_description, loss_fn, reference_fn, success_threshold)` + `register()`

---

## 2. Discrete Elastic Rod support

- [ ] `RodPhysicalSystem` TypedDict (or class):
  - `nv`, `rod_length`, `rho`, `r0`
  - `B` (2×2 bending stiffness matrix)
  - `beta` (twist stiffness GJ)
  - `EA`, `voronoi_lengths`, `m`, `W_vec` (3D gravity), `C_diag`
  - `observable_nodes`
- [ ] `make_rod_system(config)` factory
- [ ] `initial_rod_q0(ps)` — packed `[x,y,z × N, θ × (N-1)]`
- [ ] `initial_rod_u0(ps)` — same layout, all zeros
- [ ] `rod_programmer_node` — calls `discreteRod()`, unpacks result, updates state
- [ ] Rod config section: `G` (shear modulus), 3D BCs
- [ ] `cli.py --mode beam|rod` flag
- [ ] `OBSERVATION_LEVEL` extended to 3D (Bishop frame / Darboux vectors)
- [ ] `compute_shape_error` 3D extension (discrete Darboux vector replaces turning angle)
- [ ] Rod benchmark shapes in `src/benchmarks/shapes/` (helix, etc.)

---

## 2. Visualization (`src/visualization/`)

Modular visualization tool with a shared data layer and pluggable backends.
The same saved trial data drives both backends — no re-running the simulation.

### 2a. Shared data layer

What gets saved per trial (by the agent loop, backend-agnostic):

- [ ] **Full shapes** — complete node positions at every physics timestep (`all_configurations`)
- [ ] **Observed shapes** — the `N_PARTIAL` nodes the controller actually sees (`shape_history`)
- [ ] **Controller decisions** — BC issued each iteration: `{x0, y0, theta}` + iteration number + shape error % + LLM convergence call + reasoning string (`iteration_log`)
- [ ] Saved to `VISUALIZATION_DIR/trial_DATETIME_TARGET_MODE/trial_data.json`
- [ ] `trial_data.json` schema: topology once (`edges`) + frames array (full + observed nodes + decision overlay per iteration)

### 2b. Matplotlib backend (quick plots, CI-friendly)

- [ ] `render_matplotlib(trial_data, output_dir)` — reads `trial_data.json`, produces static figures
- [ ] Per-iteration frame: full shape (faded) + observed nodes (highlighted) + BC arrow + shape error % + LLM reasoning quote
- [ ] Optional `.mp4` via `matplotlib.animation`
- [ ] Config flags: `SAVE_VIDEO`, `VIDEO_FPS`, `VIDEO_DPI`

### 2c. Three.js backend (papers and videos)

Python generates one self-contained `viewer.html` — no npm, no server, no Python runtime. Double-click to inspect.

- [ ] `render_threejs(trial_data, output_dir)` — reads `trial_data.json`, writes `viewer.html`
- [ ] Embeds topology + all frames inline; Three.js CDN link only external dependency
- [ ] Shows full shape + observed nodes (distinct color) + BC annotation per iteration
- [ ] Optional scalar field coloring: curvature, contact state, shape error
- [ ] Playback controls: play/pause, frame slider, speed, reset camera, toggle layers
- [ ] Frame granularity: per-iteration by default; full physics timesteps optional via config flag

---

## 3. Shape checker calibration against real beam data

- [ ] Run pipeline on C, U, S, L; collect converged `shape_history` entries
- [ ] Compute `compute_shape_error` on real shapes; verify scores are plausible
- [ ] Tune profile parameters if real beam shapes require adjusted reference profiles

---

## 4. Evaluation harness

Quantitative measurement of controller performance across benchmark shapes.

- [ ] Run N trials per (shape, model, orchestrator_mode) combination
- [ ] Record: iterations to convergence, final error %, LLM self-assessment accuracy
- [ ] Compare plan mode vs. myopic mode across all shapes and iteration budgets
- [ ] Log all decisions, plans, shape histories, and convergence calls to `LOGS_DIR`

---

## 5. Packaging (`pyproject.toml`)

- [ ] Add `[project]` metadata and dependency list (mirrors `requirements.txt`)
- [ ] Define CLI entry point: `llm-controller = "cli:main"`
- [ ] Keep `[build-system]` minimal; do NOT publish to PyPI
- [ ] Remove `requirements.txt` once validated with `pip install -e .`

---

## 6. VLM integration (distant — after LLM pipeline is proven)

- [ ] Pass rendered frame to VLM alongside target shape description
- [ ] New VLM provider in `src/llm/providers/`
- [ ] `call_llm_with_tool` extended to accept image inputs
- [ ] Orchestrator: VLM mode uses image history instead of coordinate text
- [ ] Benchmark: LLM (text) vs. VLM (image) on same shapes and iteration budget
