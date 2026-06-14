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
- Beams (2D, wall contact): C, U, S, L — implemented in `src/benchmarks/cases/`
- Navigation cases (snake, T): designed, not yet implemented
- Rods (3D): in progress

---

## Infeasibility

Infeasibility is **not a separate task class**. Any task from any class may be feasible or infeasible depending on geometry, beam length, wall spacing, or control limits. The same S-shape benchmark is feasible with sufficient freedom and infeasible under tight endpoint constraints. The same tube navigation is feasible with a thin beam and infeasible if the tube is narrower than the beam diameter.

Infeasibility is a property of a (task, physical_system) pair, not of a task type. Cases that are known to be infeasible are flagged explicitly and are valuable for testing whether the LLM recognizes the limit rather than spinning out.

---

# 1. Common Task Interface

## State observed by the agent

- Nodal positions `x_i` (sampled observable nodes, not the full beam)
- Endpoint pose (position + slope/angle)
- Current boundary condition
- Current task loss (from the deterministic evaluator)
- Wall geometry (prose description via `convert_wall_to_language`)

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

# 6. Task Class 1: Free-Space Shape Formation

No wall contact. One end fixed, agent controls the other. Shape loss only.

Controls: `x0, y0, theta` (2D beam) or `position, orientation, twist` (3D rod).

---

## 1.1. C-Shape ✓

**Goal:** Smooth single-sided arc, constant curvature.

**Loss:**
```
L_C = w_turn*(Theta - Theta_C)^2 + w_sign*(1 - S_curv) + w_dist*(d_end - d_C)^2
```

**Behavioral use:** Baseline. Tests steady correction toward a simple one-bend target.

---

## 1.2. U-Shape ✓

**Goal:** Deep semicircular bend, turning angle near π, endpoints close together.

**Loss:**
```
L_U = w_turn*(Theta - pi)^2 + w_dist*(d_end - d_U)^2 + w_sign*(1 - S_curv)
```

**Behavioral use:** Tests whether the agent pushes farther than a mild C without oscillating.

---

## 1.3. Endpoint Pose Matching

**Goal:** Reach a prescribed tip position and slope.

**Loss:**
```
L_pose = w_pos*||p_tip - p_target||^2 + w_angle*wrap(theta_tip - theta_target)^2
```

**Behavioral use:** Cleanest free-space task. Loss is simple and contact-independent.

---

## 1.4. Free-Space S-Shape ✓

**Goal:** Produce an S-curve using only endpoint control.

**Loss:**
```
L_S = w_inflect*(N_inflect - 1)^2 + w_pose*L_pose + w_balance*curvature_balance_error
```

**Infeasible variant:** Tighten endpoint constraints so the S cannot be achieved.
Expected behavior: loss plateaus, agent should recognize stagnation.

**Behavioral use:** Tests shape following with a single inflection; infeasible variant tests recognition.

---

## 1.5. Twist-Controlled Rod Shape (3D)

**Goal:** Use endpoint twist to produce a desired twisted rod state.

**Loss:**
```
L_twist = w_pos*||p_tip - p_target||^2 + w_rot*d_SO3(R_tip, R_target)^2 + w_twist*(Phi - Phi_target)^2
```

---

## 1.6. Single-Helix Formation (3D)

**Goal:** Produce one stable helical turn.

**Loss:**
```
L_helix = w_turns*(N_turns - 1)^2 + w_radius*(r - r_target)^2 + w_pitch*(p - p_target)^2 + w_pose*L_pose
```

**Infeasible variant:** Request N_turns beyond the physically stable limit.

---

# 7. Task Class 2: Contact-Mediated Shape Formation

One or a small number of walls. Contact is an affordance — the agent must use the
wall to create curvature it could not produce alone, or avoid it while shaping.

---

## 2.1. Wall-Induced C-Shape ✓

**Goal:** Push the beam against a horizontal wall so contact creates a C-like curve.

**Loss:**
```
L_wall_C = L_C + w_contact*contact_location_error
```

**Behavioral use:** Tests whether the agent learns to use contact to create curvature.
Primary task in the current pipeline.

---

## 2.2. Corner-Induced L-Shape ✓

**Goal:** Use a wall corner to produce an L-like configuration.

**Loss:**
```
L_corner_L = w_angle*wrap(Theta - pi/2)^2 + w_tip*||p_tip - p_target||^2 + w_contact*contact_location_error
```

---

## 2.3. Two-Obstacle S-Shape

**Goal:** Use two contacts with opposite-facing walls to create opposite-signed curvature.

**Loss:**
```
L_contact_S = w_inflect*(N_inflect - 1)^2 + w_contact*contact_pattern_error + w_pose*L_pose
```

**Behavioral use:** Tests multi-contact planning for a shape that is hard in free space.

---

## 2.4. Narrow-Slot Insertion

**Goal:** Insert the beam through a finite-width horizontal slot as far as possible.

**Loss:** Tip horizontal penetration depth beyond the slot entrance.

**Infeasible variant:** Slot height < beam diameter. Agent cannot enter; should recognize
the geometric constraint rather than repeatedly attempting insertion.

**Behavioral use:** Tests contact-aware pushing and infeasibility recognition via geometry.

---

## 2.5. Peg-Assisted U-Shape

**Goal:** Wrap the beam around a peg to form a U.

**Loss:**
```
L_peg_U = L_U + w_wrap*(alpha_wrap - alpha_target)^2 + w_contact*contact_consistency_error
```

---

# 8. Task Class 3: Multi-Wall Navigation

Multiple walls form a channel or obstacle field. The beam must thread through
a constrained path. Shape along the way is determined by the walls; the task
goal is tip placement at a target unreachable without navigating the geometry.

These tasks require spatial planning — the LLM must commit to a path, not just
correct a shape signal step by step.

---

## 3.1. L-Tube Navigation (snake)

**Geometry:** An L-shaped tube — vertical entry channel, 90° bend, horizontal exit.
Four walls (right end cap removed so tube is open). Dimensions roughly:
vertical channel x ∈ [−0.01, 0.01], horizontal channel y ∈ [−0.12, −0.10].

**Goal:** Thread the tip through the L-bend and reach target `(x*, y*)` near the
open end of the horizontal channel.

**Loss:**
```
L_snake = ||p_tip - p_target||^2 / rod_length^2
```

**Infeasible variant:** Reduce the channel width below the beam's natural bending radius.
The beam cannot negotiate the bend.

**Behavioral use:** Simplest navigation task. One decision point (commit to the bend),
one target. Tests whether the LLM plans globally rather than correcting locally.

**Implementation:** `src/benchmarks/cases/beam_snake.py` — Class 3, no `reference_fn`.

---

## 3.2. Inverted-T Navigation

**Geometry:** A vertical entry stem connects to a horizontal bar at the bottom.
The horizontal bar is open at both ends (left and right exit). Five walls total.

```
         entry
           |
    -------+-------
    (left)     (right)
```

**Goal:** Thread the beam through the stem and reach a specified target at one
of the two exits. Which exit is specified in the task; the other is a distractor.

**Loss:** Tip distance to the designated exit target.

**Infeasible variant:** Target on one side with walls that make that side unreachable
given the beam's bending stiffness.

**Behavioral use:** Harder than L-tube. Two possible paths create ambiguity. The LLM
must commit early and cannot recover cheaply if it chooses the wrong direction.
Tests forward planning and irreversible decision-making.

---

# 9. Main-Paper Subset

A focused subset for the primary scientific claims.

| # | Task | Class | Physics | Status |
|---|------|-------|---------|--------|
| 1 | C-shape (wall-assisted) | 2.1 | beam | ✓ |
| 2 | U-shape (wall-assisted) | 2.1 variant | beam | ✓ |
| 3 | L-shape (corner) | 2.2 | beam | ✓ |
| 4 | S-shape (free space) | 1.4 | beam | ✓ |
| 5 | S-shape (infeasible) | 1.4 infeasible | beam | planned |
| 6 | Endpoint pose matching | 1.3 | beam | planned |
| 7 | L-tube navigation | 3.1 | beam | planned |
| 8 | Inverted-T navigation | 3.2 | beam | planned |
| 9 | Single helix | 1.6 | rod | pending rod support |
| 10 | Near-buckling control | 1.x | beam or rod | planned |

Tasks 1–8 use the beam pipeline. Tasks 9–10 require rod support.

---

# 10. Benchmark / Supplement Tasks

These are released for the community but do not drive the main scientific claim.

- Peg-assisted U-shape (2.5)
- Two-obstacle S-shape (2.3)
- Narrow-slot insertion (2.4)
- Twist-controlled rod shape (1.5)
- Multi-turn helix beyond stability (1.6 infeasible)

---

# 11. Experimental Protocol

For each (task, model, observation_mode) combination:

1. Initialize the same beam/rod state.
2. Provide the same task prompt and observation (including wall description).
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

Each rollout produces one JSON file. Top-level fields:

```json
{
  "task_name": "",
  "model_name": "",
  "observation_mode": "",
  "physics_type": "beam",
  "nv": 0,
  "rod_length": 0.0,
  "EI": 0.0,
  "EA": 0.0,
  "friction": true,
  "walls": [{"P0": [], "P1": []}],
  "all_configurations": {},
  "iteration_log": [
    {
      "step": 0,
      "bc_issued": {"x0": 0.0, "y0": 0.0, "theta": 0.0},
      "llm_reasoning": "",
      "llm_converged": false,
      "error_percent": 0.0,
      "contact_summary": ""
    }
  ]
}
```

`llm_converged` — the LLM's own convergence judgment (measured, not authoritative).
`error_percent` — the deterministic evaluator's loss for this step.
`walls` — user-facing `{P0, P1}` format only; derived fields (`n`, `d`) are not stored.

---

# 13. Role of the Benchmark

```
Focused subset  →  evidence for the scientific claim.
Full benchmark  →  resource for the community.
```

The benchmark includes task configs, prompts, simulator code, losses, metrics,
trajectories, videos, and model responses.

It does not blur the main paper's logic. The science stands on the focused subset.
