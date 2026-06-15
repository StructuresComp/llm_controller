# TODO: Orchestrator Prompting

## Status: ACTIVE

**Done:**
- ✅ All Tier 1-4 prompt architecture implemented
- ✅ Observation helpers (physics, walls, BC, trajectory, history, contact, feedback)
- ✅ Prompt builders for plan and decision (with semantic labels, narrative framing)
- ✅ K_HISTORY properly truncated in prompt
- ✅ Initial beam state shows concrete coordinates instead of vague text
- ✅ BC constraints show actual fixed position values
- ✅ Coordinate system explicitly defined (X right, Y up, Z out)
- ✅ Recent trajectory and history have narrative context
- ✅ Geometric observation level (tangent angles, curvature, total turning) explained in plain English

---

## TODO: Add Error/Loss to History Display

**Issue:** History currently shows control decisions and resulting shapes, but no error progression. The LLM can't see whether recent moves improved or worsened the error.

**Solution:** Each history step should include:
```
Step 1: moved node 0 → (x0, y0) θ=...
         Point 1: x=..., y=...
         ...
         Error: 65.3% (up from 62.1% — this move made it worse!)

Step 2: moved node 0 → (x0, y0) θ=...
         ...
         Error: 58.7% (down from 65.3% — progress!)
```

### Implementation

**In `src/core/observation.py::describe_history()`:**
- Accept `error_history` parameter (list of error_percent values, one per step)
- For each step, append the error value after the shape description
- Show delta from previous step (↑ up/worse, ↓ down/better, → flat/stagnant)

**In `src/core/prompt_builder.py`:**
- Build an `error_history` list from `state["iteration_log"]` (if available)
- Pass it to `describe_history()`
- Ensure truncation respects K_HISTORY

**Output format example:**
```
Step 1: moved node 0 → (0.0100, -0.0300) θ=0.1000 rad
         Point 1 (0% along): x=+0.0100 m, y=-0.0300 m
         ...
         Error: 65.3%

Step 2: moved node 0 → (0.0150, -0.0350) θ=0.1500 rad
         Point 1 (0% along): x=+0.0150 m, y=-0.0350 m
         ...
         Error: 58.7% (↓ improved from 65.3%)

Step 3: moved node 0 → (0.0100, -0.0400) θ=0.0500 rad
         Point 1 (0% along): x=+0.0100 m, y=-0.0400 m
         ...
         Error: 58.6% (→ flat, almost no change)
```

---

## TODO: Add Target Shape Comparison to Prompt

**Issue:** LLM sees current tangent angles/curvatures but has no reference. "What are these values *supposed* to be?"

**Solution:** Show target vs. current side-by-side so LLM can directly compare.

### Implementation

**In `src/core/observation.py`:**
- New helper: `describe_target_shape(benchmark, target_nodes) -> str`
  - Input: BenchmarkCase instance + reference nodes from `benchmark.reference_fn()`
  - For shape cases (C, U, S, L): compute target's tangent angles + curvatures, format as prose
  - For position cases (tip_reach, snake_L): compute target position, current tip position, distance gap
  - Return formatted comparison string

**In `src/core/prompt_builder.py`:**
- After "Current beam state" section, add target section:
  ```python
  # Target shape (for comparison)
  if benchmark.reference_fn is not None:
      try:
          target_nodes = benchmark.reference_fn(
              state["bc_current"][0],
              state["bc_current"][1],
              state["bc_current"][2],
          )
          if target_nodes is not None:  # Shape-based case
              target_section = describe_target_shape(benchmark, target_nodes)
              parts.extend([
                  "TARGET SHAPE (what you should aim for):",
                  target_section,
                  "",
              ])
          else:  # Position-based case (tip_reach, snake_L)
              # Show target position from benchmark if available
              target_section = describe_target_position(benchmark, current_nodes[-1])
              if target_section:
                  parts.extend([
                      "TARGET POSITION (for tip-reach):",
                      target_section,
                      "",
                  ])
      except:
          pass  # Case doesn't have reference_fn or it failed
  ```

### Output Examples

**Shape-based (C-shape):**
```
CURRENT BEAM STATE:
  Beam direction at each point (radians, counter-clockwise from straight-down):
    Point 1: 1.5708 rad = +90.0°
    Point 2: 0.4412 rad = +25.3°
    Point 3: -0.9200 rad = -52.7°
    Point 4: 0.0000 rad = +0.0°

  Curvature pattern:
    1→2: left turn (convex left)
    2→3: left turn (convex left)
    3→4: right turn (convex right)

TARGET SHAPE (what you should aim for):
  Beam direction at each point:
    Point 1: 0.0000 rad = +0.0°
    Point 2: 0.4909 rad = +28.1°
    Point 3: 0.9817 rad = +56.3°
    Point 4: 1.5708 rad = +90.0°

  Curvature pattern:
    1→2: right turn (convex right)
    2→3: right turn (convex right)
    3→4: right turn (convex right)

→ Gap: current is backwards! It curves left instead of right. Start tangent should be straight down (0°), not right (90°).
```

**Position-based (tip_reach):**
```
CURRENT TIP POSITION: (0.015, -0.095) m

TARGET TIP POSITION: (0.050, -0.100) m

→ Gap: 0.036 m away (need to reach +0.035 m right and -0.005 m down)
```

---

## TODO: Deterministic Convergence Report

**Issue:** Convergence assessment is currently left to the LLM via a minimal prompt. This is weak because:
1. The prompt only shows error % + diagnostics string
2. The LLM doesn't see the actual beam shape
3. The evaluator already computes `solved` deterministically

**Solution:** Use deterministic convergence judgment instead of LLM assessment.

### Tasks

**1. Modify `src/agents/orchestrator_agent.py::decision_node()`**
   - Remove the LLM convergence call (lines 193-223)
   - Replace with: `converged = state["eval_result"].get("solved", False)`
   - Keep the print statement for real-time feedback

**2. Document in CLAUDE.md**
   - Add section: "Convergence Assessment"
   - Explain: evaluator computes `solved` by comparing error_percent to benchmark.success_threshold
   - This is ground truth; LLM convergence assessment was research-only and added noise

### Code Change

**Before:**
```python
convergence = call_llm_with_tool(
    prompt = convergence_prompt,
    tool_schema = _CONVERGED_SCHEMA,
    tool_name   = "assess_convergence",
    system      = "You are a structural mechanics expert. Judge whether the current beam deformation matches the target shape.",
)
return {
    "decision":  decision,
    "converged": convergence["converged"],
}
```

**After:**
```python
# Use deterministic ground truth from evaluator
eval_result = state.get("eval_result", {})
converged = eval_result.get("solved", False)

if converged:
    print(f"  ✓ CONVERGED: error={eval_result.get('error_percent', 100):.2f}% ≤ threshold={...}%")
else:
    print(f"  → Not converged: error={eval_result.get('error_percent', 100):.2f}%")

return {
    "decision":  decision,
    "converged": converged,
}
```

---

## Files Modified This Session

1. `src/core/observation.py`
   - Updated `describe_nodes()` to show semantic labels ("Point 1 (0% along)" instead of "obs[0]")
   - Updated `describe_history()` to use same semantic labels
   - Rewrote `_format_descriptors()` to explain tangent angles, curvature patterns, total rotation in prose + degrees

2. `src/core/prompt_builder.py`
   - Added coordinate system section (X right, Y up, Z out)
   - Added narrative framing to "Recent progress" section
   - Added narrative framing to "Your recent control decisions and results" section
   - Added explicit K_HISTORY truncation (defensive, though programmer_agent already maintains rolling buffers)

3. `src/core/config.py` (for reference, not modified)
   - OBSERVATION_LEVEL, N_PARTIAL, K_HISTORY — all properly used in prompt_builder now

---

## Next Session: Do the Convergence Task

Once approved, implement the deterministic convergence change:
1. Edit `src/agents/orchestrator_agent.py` — remove LLM convergence call
2. Update CLAUDE.md — document convergence assessment design
3. Test with CLI: `python3 cli.py --target C --max-iterations 5`
4. Verify prompt still works without convergence prompt
