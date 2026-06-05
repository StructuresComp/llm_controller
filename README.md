# ESL Agent Framework

A two-agent framework built on LangGraph for running physics simulations and learning reduced-order energy models.

## AGENT STRUCTURE
`ESLState` → Orchestrator node → Programmer node → `physical_state`

## Overview

The framework runs discrete elastic beam/rod simulations through a two-agent LangGraph pipeline. The **Orchestrator** acts as a controller: it reads the current deformed configuration, computes a boundary condition update, and issues the next simulation instruction. The **Programmer** calls the physics tool, captures the output (DOF vector + sensitivity matrix), and reports back.

The loop repeats until a target shape is reached. Example objectives:

- **C-shape** — press one end of the rod against a wall and curve it to one side
- **U-shape** — symmetric deep bend
- **S-shape** — double-curve with inflection point

At each equilibrium the physics tool returns the sensitivity operator `S = -J_ff⁻¹ J_fb`, which maps a perturbation of the boundary conditions to the resulting change in the free configuration. The orchestrator uses `S` to estimate the right update — making this a sensitivity-guided controller rather than blind trial and error.

This is the foundation for Equilibrium Sensitivity Learning (ESL): once enough shape data is collected, a reduced-order energy model is learned from the accumulated simulation history.

## Quickstart

### 1. Clone and set up environment

```bash
git clone <repo-url>
cd agents_ESL
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Set up your LLM provider

Create a `.env` file in the project root:

```bash
ANTHROPIC_API_KEY=sk-ant-...
```

Then open `config.py` and confirm the provider:

```python
LLM_PROVIDER_FAST  = "anthropic"
FAST_MODEL         = "claude-haiku-4-5"
```

To switch to OpenAI or a local Ollama model, change those two lines (and add the corresponding API key to `.env`).

### 3. Test the physics tool (optional but recommended)

Before running the full agent pipeline, verify the beam simulator works correctly — two chained runs where the second starts from the first's final position and velocity:

```bash
python -m src.tests.test_beam_chaining
```

Expected output:
```
Controller decision:  position = (0.0, -0.03),  angle = 0.000 rad
Simulation result:    Simulation completed 500 steps to t = 5.0000 s.  |  tip: x=... m, y=... m

Controller decision:  position = (0.0, -0.05),  angle = 0.400 rad
Simulation result:    Simulation completed 500 steps to t = 10.0000 s.  |  tip: x=... m, y=... m

All checks passed.
```

### 4. Run

```bash
source venv/bin/activate
python cli.py --target C
```

**CLI flags:**
- `--target SHAPE` — target shape label (`"C"`, `"U"`, `"S"`, etc.)
- `--H DEPTH` — wall depth override (meters; default from config.py)

Example with wall override:
```bash
python cli.py --target U --H 0.15
```

Expected output:
```
--- Target ---
  Shape : C

--- Decision ---
{...orchestrator's BC decision...}

--- Result ---
Moved node 0 from (0.0000, 0.0000) θ=0.0000 rad to ...
```

## Configuration

All LLM settings live in `config.py`. No other file needs to change to switch providers or models.

| Setting | Default | Description |
|---|---|---|
| `LLM_PROVIDER_FAST` | `"anthropic"` | Provider for both agents (`"anthropic"`, `"openai"`, `"ollama"`) |
| `FAST_MODEL` | `"claude-haiku-4-5"` | Model for fast/cheap calls |
| `SMART_MODEL` | `"claude-haiku-4-5"` | Model for heavier reasoning (swap to `claude-sonnet-4-6`) |
| `LOG_LLM_CALLS` | `True` | If True, log all LLM prompts + responses to `logs/llm_calls.jsonl` |

## Debugging & Logs

All LLM calls (prompts and responses) are logged to `logs/llm_calls.jsonl` in JSON-lines format (one JSON object per line). This is useful for:
- Debugging agent failures: see what the orchestrator was told and what it decided
- Analyzing LLM reasoning: read the system prompt and full prompt context
- Reproducing runs: log contains timestamps, models, and exact input/output

**To disable logging** (reduces disk writes), set `LOG_LLM_CALLS = False` in `config.py`.

**To inspect logs:**
```bash
# Pretty-print the last 5 LLM calls
tail -5 logs/llm_calls.jsonl | jq

# Find all "tool" type calls (BC decisions, convergence checks)
grep '"call_type": "tool"' logs/llm_calls.jsonl | jq
```

## Project Structure

```
agents_ESL/
├── cli.py                          # Entry point
├── config.py                       # LLM provider + model settings
├── workflows/
│   └── workflow.py                 # LangGraph graph: orchestrator → programmer
└── src/
    ├── models/esl_state.py         # ESLState TypedDict
    ├── agents/
    │   ├── orchestrator_agent.py   # Decides what simulation to run
    │   └── programmer_agent.py     # Runs the tool, updates physical_state
    ├── llm/
    │   ├── llm.py                  # call_llm(), call_llm_with_tool(), get_langchain_llm()
    │   └── providers/              # Anthropic, OpenAI, Ollama, Fake
    └── tools/
        ├── Discrete_Elastic_Beams.py
        └── Discrete_Elastic_Rods.py
```

## Adding a New Provider

Implement `BaseProvider` in `src/llm/providers/` (two methods: `call_text`, `call_tool`), then register it in `src/llm/llm.py` and set the provider name in `config.py`.
