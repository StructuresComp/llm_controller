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

## 5. Robust tool-call fallback (permanent, not TOOL_CALL_ATTEMPTS band-aid)

Implemented in `decision_node` (`orchestrator_agent.py`): when `TOOL_CALL_ATTEMPTS` is
exhausted, falls back to a plain `call_llm` (free text, no schema) to the same model,
then a separate extraction-only pass via `config.EXTRACTION_PROVIDER`/`EXTRACTION_MODEL`
(default `anthropic`/`claude-haiku-4-5`) using `call_llm_with_tool` to parse the free
text into `_BC_SCHEMA`. Extraction failure is terminal, same as `TOOL_CALL_ATTEMPTS`
exhaustion — no separate retry budget. `src/llm/llm.py`'s `call_llm`/`call_llm_with_tool`
now accept optional `provider`/`model` overrides to support this.

- [ ] Track which path produced each decision (native / JSON-fallback / extraction) — feeds
  into `analyze_sweep.py`'s tool-calling-reliability metric. Not yet implemented.

---

## 6. Packaging

- [ ] `pyproject.toml` — `[project]` metadata, dependency list, CLI entry point
- [ ] Remove `requirements.txt` once validated with `pip install -e .`
