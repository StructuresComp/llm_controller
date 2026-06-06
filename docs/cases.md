# DLO Control — Task Suite

## Purpose

Evaluate LLM agents as closed-loop controllers of discrete elastic beams and rods.
At each step the agent observes the current configuration and a deterministic loss,
proposes the next boundary condition, and the simulator returns the updated state.

The goal is not to test whether an LLM can solve mechanics equations.
The goal is to measure closed-loop behavioral signatures: does the loss improve
reliably, or does the agent oscillate, stagnate, or fail to recognize infeasibility?

**Codebase architecture is documented in `CLAUDE.md`, not here.**
This document covers task definitions, losses, metrics, and the experimental protocol.

**Implementation status:**
- Beams (2D, wall contact): C, U, S, L — implemented in `src/benchmarks/shapes/`
- Rods (3D): in progress
- Language tasks: deferred

---

# 1. Common Task Interface

## State observed by the agent

- Nodal positions `x_i` (sampled observable nodes, not the full beam)
- Endpoint pose (position + slope/angle)
- Current boundary condition
- Current task loss (from the deterministic evaluator)

## Observation modes

**Positions** — raw (x, y) coordinates of observable nodes.

**Geometric** — positions + tangent angles + curvature signs + total turning angle.

**Natural language** — a short text description of the current shape and loss.

## Action

The agent outputs a boundary condition:

```
control = { "x0": ..., "y0": ..., "theta": ... }
```

Always machine-parseable via tool schema enforcement. Never free JSON.

## Termination

In the main experiments the agent does not decide when to stop.
The loop terminates when:
- the deterministic loss drops below the success threshold, or
- the maximum iteration budget is reached.

The LLM's own convergence judgment runs in parallel as a measured quantity,
not as the stopping criterion.

---

# 2. Common Geometric Descriptors

## Endpoint position

```
p_tip = x_N
```

## Endpoint direction (2D)

```
theta_tip = atan2(t_N_y, t_N_x)
```

## Total turning angle (2D)

```
Theta = sum_i signed_angle(t_i, t_{i+1})
```

Useful for C-like and U-like shapes.

## Endpoint distance

```
d_end = ||x_N - x_0||
```

## Inflection count

```
N_inflect = number of sign changes in discrete signed curvature kappa_i
```

Distinguishes C/U (zero inflections) from S (one inflection).

## Curvature sign consistency

```
S_curv = fraction of edges with the same curvature sign
```

## Contact count and location

```
N_contact = number of active contact nodes
s_contact = arc-length locations of contact nodes
```

## Helix descriptors (3D rod)

```
N_turns  = total azimuthal angle / (2*pi)
r_helix  = mean radial distance from estimated helix axis
p_helix  = axial rise per turn
```

---

# 3. Common Loss Components

All task losses are composed from these primitives with fixed weights.
Weights are set once before model comparisons and not changed.

## Turning angle loss

```
L_turn = (Theta - Theta_target)^2
```

## Endpoint position loss

```
L_pos = ||p_tip - p_target||^2
```

## Endpoint angle loss (2D)

```
L_angle = wrap(theta_tip - theta_target)^2
```

## Endpoint distance loss

```
L_dist = (d_end - d_target)^2
```

## Curvature sign loss

```
L_sign = 1 - S_curv
```

## Inflection count loss

```
L_inflect = (N_inflect - N_target)^2
```

## Contact loss

```
L_contact = |N_contact - N_contact_target|  or  distance from desired contact location
```

## Helix loss (3D)

```
L_helix = w_turns*(N_turns - N_target)^2 + w_radius*(r - r_target)^2 + w_pitch*(p - p_target)^2
```

## Composite loss and accuracy

```
L_task   = sum_j w_j * L_j
accuracy = max(0, 1 - L_task / L_initial)
```

---

# 4. Primary Behavioral Metrics

Two quantities drive the main paper.

## Improvement stability

Does the loss reliably decrease?

```
I_t = L_t - L_{t+1}
improvement_stability = fraction of steps where I_t > 0
```

## Oscillation

Does the trajectory settle or bounce?

```
oscillation_count = number of sign changes in (L_{t+1} - L_t)
```

---

# 5. Diagnostic Quantities

Useful for supplement and interpretation.

- **Bad-step rate** — fraction of steps where `L_{t+1} > L_t`
- **Convergence time** — first `t` such that `L_t < tolerance`
- **Final loss** — `L_T`
- **Area under loss curve** — `sum_t L_t`
- **Recovery after failure** — steps to return below pre-failure loss after a bad step
- **Overcorrection** — consecutive steps land on opposite sides of the target
- **Stagnation** — `|L_t - L_{t+1}| < eps` for several consecutive steps
- **Infeasibility recognition** — does the agent state or imply the target is unreachable?

---

# 6. Task Category A: Free-Space Boundary Control

No wall contact required. One end fixed, agent controls the other.

Controls: `x0, y0, theta` (2D beam) or `position, orientation, twist` (3D rod).

---

## A1. C-Shape ✓

**Goal:** Smooth single-sided arc, constant curvature.

**Loss:**
```
L_C = w_turn*(Theta - Theta_C)^2 + w_sign*(1 - S_curv) + w_dist*(d_end - d_C)^2
```

**Behavioral use:** Baseline. Tests steady correction toward a simple one-bend target.

---

## A2. U-Shape ✓

**Goal:** Deep semicircular bend, turning angle near π, endpoints close together.

**Loss:**
```
L_U = w_turn*(Theta - pi)^2 + w_dist*(d_end - d_U)^2 + w_sign*(1 - S_curv)
```

**Behavioral use:** Tests whether the agent pushes farther than a mild C without oscillating.

---

## A3. Endpoint Pose Matching

**Goal:** Reach a prescribed tip position and slope.

**Loss:**
```
L_pose = w_pos*||p_tip - p_target||^2 + w_angle*wrap(theta_tip - theta_target)^2
```

**Behavioral use:** Cleanest free-space task. Loss is simple and contact-independent.

---

## A4. Free-Space S-Shape Attempt ✓

**Goal:** Produce an S-curve using only endpoint control.

**Loss:**
```
L_S = w_inflect*(N_inflect - 1)^2 + w_pose*L_pose + w_balance*curvature_balance_error
```

**Note:** May be infeasible under restricted boundary controls. That is useful — see C1.

**Behavioral use:** Tests infeasibility recognition alongside shape-following.

---

## A5. Twist-Controlled Rod Shape (3D)

**Goal:** Use endpoint twist to produce a desired twisted rod state.

**Loss:**
```
L_twist = w_pos*||p_tip - p_target||^2 + w_rot*d_SO3(R_tip, R_target)^2 + w_twist*(Phi - Phi_target)^2
```

**Behavioral use:** Tests whether the agent handles rotational controls, not just planar bending.

---

## A6. Single-Helix Formation (3D)

**Goal:** Produce one stable helical turn.

**Loss:**
```
L_helix = w_turns*(N_turns - 1)^2 + w_radius*(r - r_target)^2 + w_pitch*(p - p_target)^2 + w_pose*L_pose
```

**Behavioral use:** Tests 3D spatial reasoning and twist/bending coupling.

---

# 7. Task Category B: Contact-Mediated Deformation

Contact with walls, pegs, or obstacles is required or expected.
The agent must use the physical environment as an affordance.

---

## B1. Wall-Induced C-Shape ✓

**Goal:** Push the beam against a wall so contact creates a C-like curve.

**Loss:**
```
L_wall_C = L_C + w_contact*contact_location_error
```

**Behavioral use:** Tests whether the agent learns to use contact to create curvature.
This is the primary task in the current pipeline.

---

## B2. Corner-Induced L-Shape ✓

**Goal:** Use a wall or corner to produce an L-like configuration.

**Loss:**
```
L_corner_L = w_angle*wrap(Theta - pi/2)^2 + w_tip*||p_tip - p_target||^2 + w_contact*contact_location_error
```

**Behavioral use:** Tests contact-mediated direction change.

---

## B3. Two-Obstacle S-Shape

**Goal:** Use two contacts to create opposite-signed curvature.

**Loss:**
```
L_contact_S = w_inflect*(N_inflect - 1)^2 + w_contact*contact_pattern_error + w_pose*L_pose
```

**Behavioral use:** Tests multi-contact planning for a shape that may be hard in free space.

---

## B4. Peg-Assisted U-Shape

**Goal:** Wrap the beam around a peg to form a U.

**Loss:**
```
L_peg_U = L_U + w_wrap*(alpha_wrap - alpha_target)^2 + w_contact*contact_consistency_error
```

**Behavioral use:** Tests contact-aware shaping with a geometric affordance.

---

# 8. Task Category C: Stress, Instability, and Infeasible Tasks

These expose failure modes. Does the agent stabilize, recover, or recognize infeasibility?

---

## C1. Impossible Free-Space S-Shape

**Goal:** Request an S-shape under controls that cannot produce it.

**Loss:** Same as A4.

**Expected behavior:** The agent should eventually recognize the target is unreachable
and stop escalating controls.

**Behavioral use:** Tests infeasibility recognition.

---

## C2. Incompatible Endpoint Pose and Curvature

**Goal:** Demand endpoint pose and turning angle that cannot coexist.

**Loss:**
```
L_incompat = w_pose*L_pose + w_turn*(Theta - Theta_target)^2 + w_sign*(1 - S_curv)
```

**Expected behavior:** Loss cannot be driven below threshold.

**Behavioral use:** Tests whether the agent recognizes conflicting objectives.

---

## C3. Near-Buckling Control

**Goal:** Push close to an instability threshold while maintaining controlled shape.

**Loss:**
```
L_buckling = w_pose*L_pose + w_shape*L_target_shape + w_instability*instability_penalty
```

**Behavioral use:** Tests oscillation behavior near sensitive nonlinear regimes.

---

## C4. Multi-Turn Helix Beyond Stability (3D)

**Goal:** Request more helical turns than the rod can stably sustain.

**Loss:** Same as A6 with `N_target` beyond the stable limit.

**Expected behavior:** Agent detects saturation rather than endlessly increasing twist.

**Behavioral use:** Tests physical-limit recognition.

---

# 9. Main-Paper Subset

A focused subset for the primary scientific claims.

| # | Task | Category | Physics |
|---|------|----------|---------|
| 1 | C-shape (wall) | B1 | beam |
| 2 | U-shape (wall) | B1 variant | beam |
| 3 | L-shape (corner) | B2 | beam |
| 4 | S-shape attempt | A4 / C1 | beam |
| 5 | Endpoint pose matching | A3 | beam |
| 6 | Impossible S-shape | C1 | beam |
| 7 | Incompatible pose + curvature | C2 | beam |
| 8 | Single helix | A6 | rod |
| 9 | Near-buckling | C3 | beam or rod |
| 10 | Multi-turn helix beyond stability | C4 | rod |

Tasks 1–7 can be run with the current beam pipeline.
Tasks 8–10 require rod support (TODO item 1).

---

# 10. Benchmark / Supplement Tasks

These are released for the community but do not drive the main scientific claim.

- Peg-assisted U-shape (B4)
- Two-obstacle S-shape (B3)
- Twist-controlled rod shape (A5)
- Channel-following task
- Obstacle routing task
- Self-intersection target
- Snap-through / post-buckling task

---

# 11. Experimental Protocol

For each (task, model, observation_mode) combination:

1. Initialize the same beam/rod state.
2. Provide the same task prompt and observation.
3. Ask the LLM for the next control action.
4. Parse action via tool schema (never free JSON).
5. Run the physics simulator to new equilibrium.
6. Compute deterministic loss via `evaluator_node`.
7. Return updated observation and loss to the agent.
8. Repeat until loss < threshold or iteration budget exhausted.
9. Save complete trajectory.

The LLM does not decide when to stop. The deterministic evaluator controls termination.
The LLM's own convergence judgment is recorded as a separate measured quantity.

---

# 12. Output Data Schema

Each rollout produces one record per step:

```json
{
  "task_name": "",
  "model_name": "",
  "observation_mode": "",
  "step": 0,
  "raw_response": "",
  "parsed_control": {},
  "loss": 0.0,
  "accuracy": 0.0,
  "descriptors": {},
  "llm_converged": false,
  "det_converged": false,
  "invalid_action": false
}
```

`llm_converged` — the LLM's own convergence judgment (measured, not authoritative).
`det_converged` — the deterministic evaluator's judgment (controls termination).

---

# 13. Role of the Benchmark

```
Focused subset  →  evidence for the scientific claim.
Full benchmark  →  resource for the community.
```

The benchmark includes task configs, prompts, simulator code, losses, metrics,
trajectories, videos, and model responses.

It does not blur the main paper's logic. The science stands on the focused subset.
