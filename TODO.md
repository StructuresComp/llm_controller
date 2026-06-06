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

## 1. Discrete Elastic Rod support

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

A standalone module that records and renders the full agentic decision-making process.

- [ ] **Physics frames** — captured inside `discreteBeam` / `discreteRod` at each timestep:
  - Beam/rod shape, wall line, timestep and cumulative time
- [ ] **Decision frames** — generated after each orchestrator cycle:
  - Current shape + target BC annotation
  - Iteration number, BC values, shape error % from evaluator
  - LLM convergence reasoning (one-line quote)
- [ ] **Video composition**: interleave physics + decision frames; one `.mp4` per trial
- [ ] Config flags: `SAVE_VIDEO`, `VIDEO_FPS`, `VIDEO_DPI`
- [ ] File layout:
  ```
  VISUALIZATION_DIR/
    trial_DATETIME_TARGET_MODE/
      trial.mp4
      metadata.json
  ```
- [ ] Works for both beam (2D) and rod (3D or projected 2D)

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
