# Paper Outline: Closed-Loop Behavioral Signatures of LLM Agents Across Language and Physical Control

## Abstract
As large language models move from answering questions to acting as agents, the central question is no longer only whether they produce the right final answer, but how they update decisions while interacting with the world. We built a closed-loop experimental system in which an LLM orchestrator observes the state of nonlinear deformable-body simulations, calls tools through a multi-agent framework, receives deterministic physical feedback, and revises boundary-control decisions over time. The physical systems are not decorative demos: discrete elastic rods and beams expose bending, stretching, twisting, contact, instability, and infeasible objectives, creating a controlled physical arena where agent behavior can be measured rather than merely described.

This work shows that LLM agents exhibit stable, model-specific decision dynamics that persist from language to physical control. Models that revise, oscillate, overcorrect, persist, or fail to recognize impossible constraints in language show corresponding closed-loop behaviors when controlling deformable physical systems in simulation and hardware. The result reframes physical AI as an experimental instrument for studying agent intelligence: by forcing LLMs to act, fail, recover, and confront physical impossibility, we reveal behavioral fingerprints that are invisible in static benchmarks but essential for deploying agents in the real world.

## Core Idea

Large language models are increasingly used as agents: they observe a state, make a decision, receive feedback, and revise their next decision. Most evaluations ask whether the final answer or final task outcome is correct. This misses a more fundamental question: how does the agent behave while it is updating its decisions?

The central claim of this paper is that LLM agents exhibit measurable closed-loop behavioral signatures. These signatures include overcorrection, undercorrection, oscillation, persistence, recovery from bad decisions, and recognition of infeasible objectives. We show that these signatures are not limited to one task domain. The same model-level behaviors that appear in language revision tasks also appear in physical AI control tasks involving discrete elastic rods and beams.

The physical system is not merely a robotics benchmark. It is a controlled scientific instrument for measuring LLM decision dynamics. A rod or beam simulator provides a state, a controllable action space, deterministic task feedback, nonlinear physical consequences, and hard feasibility limits. This allows us to quantify how an LLM agent updates its decisions over time.

## DLO-ControlBench
We release a standardized closed-loop benchmark for evaluating LLM agents on deformable-object control tasks involving free-space boundary control, contact-mediated shaping, and infeasible/stress objectives.

---

## Unified Closed-Loop View

We describe both language tasks and physical AI tasks using the same closed-loop structure:

1. The agent observes the current state.
2. The agent makes a decision.
3. The environment or evaluator returns feedback or loss.
4. The agent updates its next decision.

In language tasks, the state may be a paper, proposal, technical claim, or previous answer. The decision may be a scalar score, ranking, or judgment. The feedback may be a fixed weak social perturbation, evaluator score, or reference error.

In physical AI tasks, the state is the current configuration of a discrete elastic rod or beam. The decision is a control input such as endpoint position, endpoint slope, endpoint rotation, imposed twist, displacement, or contact action. The feedback is a deterministic shape loss or target accuracy computed from the simulator or robot sensing pipeline.

The paper is built around this shared structure. Language and physical control are not treated as separate unrelated domains. They are paired instantiations of the same decision-update process.

---

## Main Scientific Hypothesis

The main hypothesis is:

**LLM agents have model-specific closed-loop stability signatures that transfer across language and physical control.**

In plain terms:

* A model that over-revises in language tends to overcorrect in physical control.
* A model that oscillates in language tends to oscillate in physical control.
* A model that is highly persistent in language tends to produce smoother but slower physical correction.
* A model that fails to recognize inconsistent language constraints tends to struggle with physically infeasible control objectives.

The goal is not to prove that language and mechanics are the same. The goal is to show that the decision-update behavior of the agent has stable quantitative properties that can be measured in both domains.

---

## Shared Behavioral Metrics

The paper defines a small family of domain-independent behavioral metrics before running the full experiments. These metrics are then instantiated in both language and physical AI.

### 1. Improvement Stability

This measures whether the agent steadily improves after feedback.

In language, improvement means that the revised score, ranking, or answer moves closer to a reference judgment or evaluator score.

In physical AI, improvement means that the rod or beam shape loss decreases after the agent applies a new control action.

This metric answers:

**Does the agent make steady progress, or does it improve and degrade unpredictably?**

---

### 2. Bad-Step Rate

This measures the fraction of updates that make the situation worse.

In language, a bad step occurs when a revision moves farther from the reference score or produces a less consistent judgment.

In physical AI, a bad step occurs when the next rod or beam configuration has higher shape loss than the previous one.

This metric answers:

**How often does the agent revise in the wrong direction?**

---

### 3. Overcorrection and Undercorrection

This measures whether the agent responds too aggressively or too weakly to feedback.

In language, overcorrection occurs when a weak or limited feedback signal causes an excessive score change, judgment flip, or unjustified explanation drift. Undercorrection occurs when meaningful feedback produces almost no update.

In physical AI, overcorrection occurs when the agent pushes the rod or beam past the desired target descriptor or repeatedly overshoots the target shape. Undercorrection occurs when it keeps making tiny updates that barely reduce the loss.

This metric answers:

**Is the agent calibrated, overly reactive, or too timid?**

---

### 4. Oscillation

This measures whether the agent repeatedly reverses direction instead of stabilizing.

In language, oscillation appears when the score or judgment moves up, down, and up again across repeated feedback rounds, or when the agent repeatedly flips its conclusion.

In physical AI, oscillation appears when the shape loss repeatedly decreases and increases, or when the rod or beam repeatedly crosses the desired target descriptor without settling.

This metric answers:

**Does the agent stabilize near a solution, or does it bounce around?**

---

### 5. Persistence

This measures whether the agent maintains a consistent strategy over time.

In language, persistence appears when the agent maintains its original judgment under weak feedback unless stronger evidence is provided.

In physical AI, persistence appears when the agent continues a coherent control strategy instead of changing direction at every step.

Persistence is not automatically good or bad. Excessive persistence can become stubbornness. Too little persistence can become instability.

This metric answers:

**Does the agent maintain a coherent plan, and does it know when to revise it?**

---

### 6. Recovery from Failure

This measures what happens after the agent makes a bad decision.

In language, recovery means that after a poor revision, the model can return to a better judgment when given feedback.

In physical AI, recovery means that after an action increases shape loss or creates an unfavorable configuration, the model can bring the rod or beam back toward the target.

This metric answers:

**Does the agent recover from errors or spiral into worse behavior?**

---

### 7. Infeasibility Recognition

This measures whether the agent recognizes that the requested objective cannot be achieved under the current constraints.

In language, infeasibility may involve inconsistent reviewer demands, contradictory instructions, impossible scoring requirements, or mutually incompatible constraints.

In physical AI, infeasibility may involve asking a rod or beam to reach a shape that is impossible under the available boundary conditions, such as an S-shape under restricted free-space actuation, a multi-turn helical configuration beyond stability limits, or an incompatible endpoint pose and curvature profile.

This metric answers:

**Does the agent recognize limits, or does it keep optimizing nonsense?**

---

## Paired Behavioral Assays

The experiments are organized by behavioral property, not by domain. Each assay has a language version and a physical AI version.

### Assay 1: Correction Sensitivity

The language task asks the LLM to score or judge a paper, proposal, or technical claim. The model receives feedback and is asked to revise. We measure how strongly the score or judgment changes.

The physical task asks the same LLM to control a discrete elastic rod or beam toward a target shape. After each action, the simulator or robot returns a shape loss or accuracy. We measure how strongly the model updates its control behavior.

The question is whether models that strongly revise in language also strongly correct or overcorrect in physical control.

---

### Assay 2: Oscillation

The language task applies repeated rounds of feedback and measures whether the model’s judgment stabilizes or keeps reversing.

The physical task applies repeated closed-loop control steps and measures whether the rod or beam loss stabilizes or repeatedly rises and falls.

The question is whether models that oscillate in reasoning also oscillate in physical control.

---

### Assay 3: Persistence and Stability

The language task measures whether the model holds a judgment under weak feedback and revises only when feedback is meaningful.

The physical task measures whether the model maintains a coherent control strategy across steps, rather than changing direction erratically.

The question is whether persistence in language corresponds to smoother physical control, and whether excessive persistence causes slow adaptation.

---

### Assay 4: Failure Recovery

The language task introduces cases where the model initially makes a poor judgment or receives corrective feedback. We measure whether it can recover.

The physical task places the rod or beam in configurations where a bad action increases the loss. We measure whether the LLM can recover and continue toward the target.

The question is whether models that recover well in language also recover well in physical control.

---

### Assay 5: Infeasibility Recognition

The language task gives the agent inconsistent or impossible constraints.

The physical task gives the agent an impossible or near-impossible rod or beam objective.

The question is whether the agent recognizes infeasibility across both domains, or whether it keeps issuing increasingly unreasonable decisions.

---

## Physical AI Testbed

The physical AI side uses discrete elastic rod and beam simulations. Here, “bending” refers broadly to elastic deformation, including bending, stretching, twisting, contact, and geometric nonlinearities.

The LLM does not directly solve mechanics equations. Instead, it acts as a closed-loop controller. At each step, it receives an observation and a deterministic loss or accuracy score, then proposes the next control input.

Possible controls include:

* endpoint position,
* endpoint slope or orientation,
* imposed rotation,
* imposed twist,
* clamp angle,
* displacement magnitude,
* contact action,
* push direction.

The simulator computes the resulting equilibrium shape and reports the updated loss. Representative tasks are transferred to robotic hardware to test whether the same behavioral signatures appear outside simulation.

---

## Physical Task Families

The physical tasks are grouped into three main families to avoid overwhelming the story.

### 1. Free-Space Boundary Control

One end of the rod or beam is fixed. The LLM controls the other end through position, slope, rotation, or twist. There is no contact with walls or obstacles.

This setting is clean and minimal. It tests whether the LLM can update boundary conditions to reduce a shape loss under a constrained actuation model.

Example tasks include:

* forming a C-like shape,
* forming a U-like shape,
* reaching a target endpoint pose,
* approximating a curvature profile,
* attempting an S-like shape that may be infeasible under the available controls.

This family tests correction, convergence, undercorrection, overcorrection, and infeasibility under limited actuation.

---

### 2. Contact-Mediated Control

The rod or beam interacts with walls, pegs, obstacles, channels, or surfaces. The LLM must use contact as a physical affordance.

Example tasks include:

* pushing a beam against a wall to create a curved shape,
* using obstacles to form C, U, S, or L shapes,
* routing a rod or cable through a constrained path,
* wrapping around pegs,
* using contact to realize shapes that are difficult or impossible in free space.

This family tests whether the LLM can exploit the environment rather than treating contact as a failure.

---

### 3. Stress and Infeasible Tasks

The LLM is asked to achieve targets that are physically difficult, near instability, or impossible under the available controls.

Example tasks include:

* producing more helical turns than the rod can stably sustain,
* forcing incompatible endpoint pose and curvature,
* reaching a target shape that requires self-intersection,
* producing an S-shape in a free-space setting where the actuation cannot realize it,
* continuing deformation beyond a buckling or instability threshold.

This family tests whether the LLM can detect physical limits, avoid meaningless escalation, and recover from failed attempts.

---
## Agent Architectures

We consider three agent architectures for deformable object control. **Reactive Control** is purely myopic: at each iteration, the LLM receives the current shape, target shape, and interaction history, and directly selects the next boundary condition. **Plan-Then-Control** first asks the LLM to generate a high-level strategy for reaching the target shape; this plan is then included in the prompt at every subsequent control step but remains fixed throughout the episode. **Adaptive Plan-and-Control** combines planning and control, where at each iteration the LLM selects the next boundary condition and simultaneously decides whether to retain, modify, or completely revise its current plan based on the observed outcome. Together, these architectures span increasing levels of temporal abstraction, from reactive decision-making to persistent strategic reasoning and adaptive replanning.

---

## Language Task Families

The language side is designed to mirror the same closed-loop behavioral structure.

### 1. Scalar Judgment Revision

The model scores papers, proposals, or technical claims. It receives feedback and revises the score. The score trajectory is analyzed for overcorrection, undercorrection, oscillation, and convergence.

### 2. Fixed Weak Social Perturbation

The model receives a fixed phrase such as:

“Are you sure? I heard something different from another reviewer.”

This feedback contains no task-specific evidence but does contain weak social pressure. The experiment measures whether the model changes its judgment too easily, resists appropriately, or oscillates.

### 3. Evaluator-Guided Revision

The model receives deterministic feedback from an evaluator or reference score. This creates a language analogue of physical shape loss. The model is repeatedly asked to revise until it reaches a tolerance or exhausts a fixed budget.

### 4. Inconsistent or Impossible Constraints

The model is given impossible reviewer demands, contradictory criteria, or infeasible scoring instructions. This mirrors impossible physical targets in rods and beams.

---

## Main Result to Demonstrate

The main result is a cross-domain relationship between language and physical AI behavior.

The paper should show that:

* models that over-revise in language also tend to overcorrect in physical control,
* models that oscillate in language also tend to oscillate in rod and beam control,
* models that show stable language revision tend to show stable physical correction,
* models that fail to recognize impossible language constraints also fail to recognize impossible physical targets,
* these relationships persist across task families, prompts, observation formats, simulations, and representative hardware experiments.

The strongest version of the result is not that one particular metric correlates once. The strongest version is that a pre-defined family of stability metrics gives the same model-level ordering across language and physical control.

---

## Controls and Robustness

The paper must address the concern that the metrics were chosen after seeing the results.

To do this, we define the behavioral metric family before the full experiments. We then report the primary metrics, secondary metrics, and negative results.

Important controls include:

* multiple prompts,
* multiple LLMs,
* multiple physical tasks,
* multiple language tasks,
* different observation formats,
* different feedback styles,
* fixed iteration budgets,
* deterministic loss functions,
* simulation-to-hardware comparison.

The paper should explicitly report when correlations weaken or fail. This makes the central claim more credible.

---

## Hardware Experiments

The hardware experiments are included to show that the measured behavior is not merely a simulation artifact.

The real robot performs representative deformable-object manipulation tasks using the same closed-loop structure. The LLM receives observations and loss feedback, proposes controls, and the robot executes them.

The hardware experiments should include:

* at least one free-space boundary-control task,
* at least one contact-mediated task,
* at least one difficult or infeasible task.

The videos and images are scientifically useful because they make the behavioral signatures visible. However, every hardware demo must also include quantitative traces such as loss over time, bad-step rate, oscillation, convergence, and infeasibility behavior.

---

## Contribution

This paper is not primarily about making LLMs bend rods or beams. It is about using controlled physical systems to quantify the closed-loop behavior of LLM agents.

The contributions are:

1. A unified closed-loop framework for comparing LLM behavior across language and physical AI.

2. A pre-defined family of behavioral metrics including improvement stability, bad-step rate, overcorrection, undercorrection, oscillation, persistence, recovery, and infeasibility recognition.

3. A discrete-elastic-rod and beam testbed that exposes nonlinear deformation, contact, instability, and physical infeasibility.

4. A paired experimental design showing that language revision behavior and physical control behavior are connected at the model level.

5. Hardware-in-the-loop demonstrations showing that these behavioral signatures appear in real physical manipulation, not only in simulation.

The central message is:

**LLM agents have measurable closed-loop behavioral signatures, and these signatures transfer across language, simulation, and embodied physical control.**
