# ESL Agent Framework — TODO

Checked items are done. Work top-to-bottom — later items depend on earlier ones.

---

## 0. Orchestrator Mode — Design Decision

Two modes, selectable via config flag `ORCHESTRATOR_MODE`:

**`"plan"`** — Grounded approach.
On the first call the orchestrator reasons about the full strategy (given target shape,
physical setup, and beam parameters) and writes a natural-language plan to `state["plan"]`.
Every subsequent decision appends the plan to the prompt, anchoring the LLM to its prior
reasoning. Risk: wrong initial plan commits the orchestrator — `ALLOW_REPLAN` can clear it.

**`"myopic"`** — Reactive approach.
No plan. `state["plan"]` stays empty. The orchestrator sees only the last K (shape, BC)
pairs and issues the next BC. Prone to zigzagging across many iterations.

Both modes implemented. `"plan"` is the default.

---

## 1. config.py
- [x] Add `NV`
- [x] Add `Y`, `RHO`, `R0`, `ROD_LENGTH`, `VISC`
- [x] Add `WALL_H`, `BOUNDARY_VELOCITY`, `BOUNDARY_ANGLE_VELOCITY`, `FRICTION`
- [x] Reorganize into 3 sections: LLM (1), Simulation & Visualization (2), Agent Perception & Control (3)
- [x] Add `OBSERVATION_LEVEL` flag in Section 3a (`"positions"`, `"geometric"`, `"full"`)
- [x] Move orchestrator settings to Section 3: `N_PARTIAL`, `K_HISTORY`, `ORCHESTRATOR_MODE`, `ALLOW_REPLAN`, `MAX_ITERATIONS`, `MAX_CONFIGS`
- [x] Add `MAX_TOKENS_FAST`, `MAX_TOKENS_SMART` — hard caps per LLM call

---

## 2. PhysicalSystem (`src/models/physical_system.py`)
- [x] `PhysicalSystem` TypedDict with raw + derived fields
- [x] `make_physical_system(config)` factory
- [x] `observable_nodes` — computed once from `NV` and `N_PARTIAL`; never changed during run
  - node 0, `N_PARTIAL-2` interior nodes (linspace), node `nv-1`
- [x] `initial_q0(ps)`, `initial_u0(ps)` helpers

---

## 3. ESLState (`src/models/state_llm_control.py`)
- [x] `physical_system`, `q0_current`, `u0_current`
- [x] `bc_current`, `bc_target`, `bc_history`, `shape_history`
- [x] `all_configurations`, `target_shape`, `plan`, `converged`
- [x] `decision: dict` (schema-validated BC from orchestrator)
- [x] Add `ctime_current: float` — cumulative simulation time; passed as `t_start` to next tool call
- [x] Update `initial_esl_state()` to set `ctime_current = 0.0`

---

## 4. `discreteBeam` tool (`src/tools/Discrete_Elastic_Beams.py`)
- [x] Accepts `physical_system`, `q0`, `u0` (self-initialises if None)
- [x] `t_start` for cumulative time chaining
- [x] `totalTime` computed from boundary velocities if not provided
- [x] Returns structured dict: `status`, `message`, `q_new`, `u_new`, `ctime`
- [x] Wall contact: predictor / corrector with `is_wall_contact`
- [x] `getControlNodes` — linear BC interpolation with CCW angle convention

---

## 5. cli.py
- [x] `--target`, `--x0`, `--y0`, `--theta`, `--H`
- [x] Builds `PhysicalSystem`, sets all initial state fields

---

## 6. programmer_agent.py
- [x] Zero LLM calls — entirely deterministic Python
- [x] Helpers: `extract_observable_q`, `format_result`, `append_and_trim`, `downsample_configs`
- [x] Node: reads decision dict, calls tool, updates all state fields, writes text report
- [x] Main: standalone test with hardcoded decision

---

## 7. orchestrator_agent.py
- [x] Two nodes: `planner_node` (no-op or plan generation) → `decision_node` (always runs)
- [x] Schemas: `_BC_SCHEMA` and `_CONVERGED_SCHEMA` (both use `call_llm_with_tool`)
- [x] Prompts: `_PLAN_SYSTEM/PROMPT`, `_DECISION_SYSTEM/PROMPT` (module-level constants)
- [x] Helpers: `_format_plan_section`, `_format_history`, `_format_current`, `_detect_stagnation`
- [x] `build_orchestrator_graph()` → compiled sub-graph
- [x] Main: standalone test of one full orchestrator cycle

---

## 8. workflows/workflow.py
- [x] Convergence edge: loops orchestrator → programmer until converged, error, or recursion_limit
- [x] Orchestrator wired as compiled sub-graph via `build_orchestrator_graph()`
- [x] `recursion_limit = MAX_ITERATIONS * 4` set in cli.py invoke

---

---

## 9. Modular observation levels (`src/agents/orchestrator_agent.py`)
- [x] Add `_compute_descriptors()` — tangent angles, curvature signs, total turning angle
- [x] Add `_format_descriptors()` — human-readable text output of descriptors
- [x] Wire `OBSERVATION_LEVEL` dispatch into `_format_history()` and `_format_current()`
- [x] Descriptors computed on-the-fly when building orchestrator prompt
- [x] No changes to state or programmer when toggling observation level
- [x] Extensible: easy to add more descriptor types (wall contact, arc length, etc.)

---

## 10. CLAUDE.md
- [x] Full rewrite — research framing, updated state table, architecture, config, tool description
- [x] Add Section 3 (Agent Perception & Control) to config table
- [x] Document observation levels and geometric descriptor computation

---

## 11. Project naming and documentation refactor

- [ ] **Rename ESL throughout**: Currently "Energy Sensitivity Learning" but the framework is now generalized for LLM/VLM control of deformable objects. Update in:
  - [ ] README.md — reframe purpose and motivation
  - [ ] CLAUDE.md — remove ESL-specific references, generalize to "agentic control" or "DLO controller" terminology
  - [ ] All docstrings and comments referring to ESL
  - [ ] Config file header and section comments
  - [ ] State model docstrings
  - [ ] Consider keeping internal names (ESLState, esl_state, etc.) for now if refactoring breaks existing code; add a note on deprecation path

---

## 12. Deterministic shape checker — rewrite (`src/tools/shape_checker.py`)

**Design principle (DDG-informed):**
The shape of a discrete curve is fully encoded by the **discrete curvature sequence**
κ_i = Δθ_i = θ_{i+1} − θ_i (turning angle at each interior node, wrapped to [−π, π]).
This is invariant to rigid-body translation and rotation. Everything else — absolute
positions, wall contact, symmetry in XY — is irrelevant to shape identity and must be
removed from the error metric.

**Step 1: Core primitives**
- [ ] `extract_turning_angles(nodes) -> np.ndarray`
  - Compute N−1 edge tangent angles θ_i = arctan2(dx_i, −dy_i) (CCW from −Y)
  - Return κ_i = θ_{i+1} − θ_i wrapped to [−π, π], length N−2
  - This is the only shape descriptor needed

**Step 2: Ideal κ profiles (arc-length parametrised)**

Each target shape is defined by a canonical κ*(s_i) profile over arc-length s ∈ [0,1].
All shapes are invariant to overall scale (normalise by arc-length) and orientation
(try both signs; take minimum error = CW vs CCW invariance).

- [ ] C — constant curvature:
  - κ*(s) = κ_0 (same value at every interior node)
  - Best-fit κ_0 = mean(κ_provided); shape error = how far from constant
  - Error metric: std(κ_provided) / |mean(κ_provided)| (coefficient of variation)

- [ ] U — constant curvature, larger total turn (|Σκ| ≈ π):
  - Same profile as C but |total_turning| ≈ π
  - Error: CV of κ_provided + penalty for |total_turning| far from π

- [ ] S — antisymmetric curvature profile:
  - κ*(s_i) > 0 for i < (N−2)/2, < 0 for i ≥ (N−2)/2; total_turning ≈ 0
  - Best-fit: mirror the κ sequence, compare to its own negated mirror
  - Error metric: antisymmetry residual + penalty for |total_turning| > ε

- [ ] L — concentrated curvature (delta-like):
  - κ*(s_i) ≈ 0 for most i, large at one location
  - Best-fit: find the single interior node with max |κ|; measure how concentrated
  - Error metric: 1 − (max_κ / Σ|κ|) (how much of total turning is in one node)

**Step 3: Orientation invariance**
- [ ] For each target shape, compute error for both κ_provided and −κ_provided
- [ ] Return the minimum (CW and CCW C-shapes are equally valid)

**Step 4: Normalised error output (0–100 scale)**
- [ ] `compute_shape_error(nodes, target_shape) -> dict` returns:
  - `error_percent`: 0 = perfect, 100 = completely wrong
  - `confidence_percent`: 100 − error_percent
  - `best_orientation`: "CW" or "CCW" (which sign gave lower error)
  - `kappa_provided`: the extracted κ sequence
  - `kappa_ideal`: the best-fit ideal κ sequence (for plotting)
  - `diagnostics`: one-line human-readable explanation

**Step 5: Anchored ideal shape for visualisation**
- [ ] `reconstruct_ideal(nodes, target_shape) -> list[tuple]`
  - Start from node 0 of provided shape (position + tangent direction)
  - Integrate κ*(s) forward using unit arc-length steps (same spacing as provided)
  - Returns ideal node positions in the same coordinate frame → overlay on plot

**Step 6: Test file rewrite (`src/tests/test_shape_checker.py`)**
- [ ] Generate ideal shapes using `integrate_kappa(kappa_fn, n_nodes, arc_length, start_pos, start_angle)`
  where kappa_fn is a function of arc-length index i
- [ ] `create_c_shape(n, total_turn)` — constant κ = total_turn / (n−2)
- [ ] `create_u_shape(n)` — same but total_turn = π
- [ ] `create_s_shape(n, amp)` — antisymmetric κ profile, Σκ = 0
- [ ] `create_l_shape(n, turn_at)` — κ concentrated at node turn_at
- [ ] Verify: `compute_shape_error(shape, target)` returns 0% for ideal shapes
- [ ] Plot: two panels — (left) κ_i profile comparison, (right) overlay of provided vs. ideal shape in XY
- [ ] Remove all references to wall_h, contact_fraction, symmetry from tests

**Step 7: Clean up `SHAPE_SPECS`**
- [ ] Replace with `SHAPE_KAPPA_PROFILES` — dict mapping shape name to a function
  `kappa_profile(n_edges) -> np.ndarray` that returns the normalised ideal κ sequence
- [ ] Wall contact, absolute position, symmetry metrics: **delete entirely**

---

## 13. Visualization module (`src/visualization/`)

Core requirement for the final product. All paths go through `paths.VISUALIZATION_DIR`.

- [ ] **Real-time renderer** — plot beam configuration during simulation
- [ ] **Configuration saver** — write each `q_new` snapshot to `DATA_DIR`
- [ ] **Static plot** — beam + wall + contact nodes highlighted
- [ ] **Movie maker** — `.mp4` via `ffmpeg` (prefer over matplotlib.animation for speed)
- [ ] **Decision overlay** — annotate each frame with BC decision (arrow, angle arc, step number)
- [ ] Add `SAVE_CONFIGS` and `SAVE_MOVIE` bool flags to `config.py` section 2e

---

## 14. Evaluation harness (after LLM pipeline is bullet-proof)

Quantitative measurement of LLM controller performance across benchmark shapes.

- [ ] Run N trials per (shape, mode) combination and record: iterations to convergence,
  backtrack count, final geometric error, LLM self-assessment accuracy
- [ ] Geometric ground truth checker — deterministic Python that computes curvature sign
  pattern, total turning angle, and symmetry metrics to label "success" independently
  of the LLM's own convergence judgment (for calibrating LLM self-assessment reliability)
- [ ] Benchmark shapes: C, U, S at minimum; add more as research expands
- [ ] Compare plan mode vs. myopic mode across all shapes and iteration budgets
- [ ] Log all decisions, plans, shape histories, and convergence calls to `LOGS_DIR`
  for post-hoc analysis

---

## 15. VLM integration (distant — after LLM pipeline is proven)

**Prerequisite: items 1–12 complete and LLM agentic loop is stable and reproducible.**

The core research extension: replace (or augment) text-based shape observation with
rendered images, enabling vision-language models to act as the controller.

**Role of VLMs:**
- The programmer renders the current beam configuration as an image (via the visualization
  module, item 11) and passes it to the VLM alongside the target shape description or
  a reference image.
- The VLM issues the next BC decision and convergence judgment from the visual input —
  no geometric descriptors, no (x, y) text.
- This tests whether visual perception of the physical system outperforms text-based
  reasoning over node coordinates.

**What needs to change:**
- [ ] Programmer node: add rendered image output alongside the text report
- [ ] New VLM provider in `src/llm/providers/` — wraps Anthropic Vision or GPT-4V
- [ ] `call_llm_with_tool` extended to accept image inputs (base64 or file path)
- [ ] Orchestrator: VLM mode uses image + target reference image instead of text history
- [ ] Benchmark: compare LLM (text) vs. VLM (image) on same shapes, same iteration budget

**Research questions for VLM phase:**
- Does visual input improve convergence rate and iteration efficiency?
- Can a VLM judge shape convergence more reliably than an LLM reading coordinates?
- Does the plan mode advantage (if any) persist when switching to VLM?
- What is the minimum image resolution / rendering fidelity needed for reliable control?
