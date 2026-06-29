# DLO Controller — TODO

## 1. Near-term benchmarks

- [ ] `beam_snake_T.py` — inverted T navigation: beam enters from top, must exit left or right
  - Harder: the LLM must commit to a side; ambiguous loss until commitment is made
  - Design the four walls, loss (tip-reach on chosen side), `target_description`
  - Very similar in design to `beam_snake_L.py`

---

## 2. Evaluation harness

- [ ] Run N trials per (shape, model, orchestrator_mode)
- [ ] Record: iterations to convergence, final error %, LLM self-assessment accuracy
- [ ] Compare plan vs. myopic vs. replan across all shapes and iteration budgets

---

## 3. VLM integration

Two changes only:
- [ ] `observation.py` — render current beam state to image; return path or bytes
- [ ] `prompt_builder.py` — pass image alongside text prompt when `OBSERVATION_MODE = "image"` or `"both"`
- [ ] Config: add `OBSERVATION_MODE = "text" | "image" | "both"`

No new provider, no new graph mode — same tool schema, same orchestrator.

---

## 4. Discrete Elastic Rod support

- [ ] `RodPhysicalSystem` fields: `nv`, `rod_length`, `rho`, `r0`, `B`, `beta`,
  `EA`, `voronoi_lengths`, `m`, `W_vec`, `C_diag`, `observable_nodes`
- [ ] `make_rod_system(config)` factory
- [ ] `initial_rod_q0/u0` — packed `[x,y,z x N, theta x (N-1)]`
- [ ] `discreteRod` — wall support via same `cx`/`cy` contact abstraction; needs 3D extension
- [ ] `rod_programmer_node` — calls `discreteRod`, unpacks result, updates state
- [ ] `OBSERVATION_LEVEL` extended to 3D (Bishop frame / Darboux vectors)
- [ ] Rod benchmark shapes — helix, etc.

---

## 5. Packaging

- [ ] `pyproject.toml` — `[project]` metadata, dependency list, CLI entry point
- [ ] Remove `requirements.txt` once validated with `pip install -e .`
