# TODO — DLO LLM Control Benchmark

## 1. Run sweep

- [ ] Define final model list:
  - `qwen2.5:14b`
  - `qwen2.5-coder:14b`
  - `phi4`
  - `mistral-nemo`
  - `gemma3:12b`
  - `olmo2:13b`
  - optional API subset: Haiku, Sonnet, GPT, Gemini
- [ ] Define final benchmark list:
  - C-shape
  - U-shape
  - S-shape
  - circular arc
  - endpoint pose
  - wall-induced C-shape
  - wall/contact variants
  - infeasible or stress cases
- [ ] Run multiple stochastic replicates per `(model, task)`.
- [ ] Save every rollout with the same iteration budget and same stopping rule.

## 2. Save per iteration

For every iteration `t`, save:

```json
{
  "iteration": 0,
  "bc": {"x0": 0.0, "y0": 0.0, "theta": 0.0},
  "q": [],
  "error_percent": 0.0,
  "timestamp": 0.0
}
```

Minimum required trajectory data:

- [ ] Error `e_t`
- [ ] Boundary condition / control `u_t`
- [ ] DLO shape `x_t`

## 3. Normalize control variables

Before computing gain metrics, normalize boundary controls.

For 2D beam control:

```math
\tilde u_t = \left[\frac{x_t}{L},\frac{y_t}{L},\frac{\theta_t}{\pi}\right]
```

Then define control step size:

```math
a_t = \|\tilde u_{t+1} - \tilde u_t\|
```

## 4. Basic performance metrics

Compute per rollout:

```math
E_T = e_T
```

Final error:

```math
E_{final}^{norm} = \frac{e_T}{e_0 + \epsilon}
```

Best error:

```math
E_{best}^{norm} = \frac{\min_t e_t}{e_0 + \epsilon}
```

Area under error curve:

```math
AUC_E = \frac{1}{T+1}\sum_{t=0}^{T}\frac{e_t}{e_0 + \epsilon}
```

Best-to-final degradation:

```math
D_{regress} = \frac{e_T - \min_t e_t}{e_0 + \epsilon}
```

Success flag:

```math
success = \mathbf{1}[e_T < \tau]
```

## 5. Error stability metrics

Bad-step rate:

```math
R_{bad} = \frac{1}{T}\sum_{t=0}^{T-1}\mathbf{1}[e_{t+1} > e_t]
```

Improvement stability:

```math
R_{improve} = \frac{1}{T}\sum_{t=0}^{T-1}\mathbf{1}[e_{t+1} < e_t]
```

Oscillation count:

```math
N_{osc} = \#\text{ sign changes in }(e_{t+1}-e_t)
```

Largest error spike:

```math
S_{max} = \max_t(e_{t+1}-e_t)
```

## 6. Control-gain metrics

Mean control step:

```math
\bar a = \frac{1}{T}\sum_{t=0}^{T-1} a_t
```

Effective gain:

```math
K_{eff} = \frac{1}{T}\sum_{t=0}^{T-1}\frac{a_t}{e_t + \epsilon}
```

Gain variability:

```math
\sigma_K = \operatorname{std}_t\left(\frac{a_t}{e_t + \epsilon}\right)
```

Useful gain:

```math
G_{useful} = \frac{\sum_{t=0}^{T-1}\max(e_t-e_{t+1},0)}{\sum_{t=0}^{T-1}a_t + \epsilon}
```

Harmful gain:

```math
G_{harmful} = \frac{\sum_{t=0}^{T-1}\max(e_{t+1}-e_t,0)}{\sum_{t=0}^{T-1}a_t + \epsilon}
```

Control flip rate:

```math
R_{flip} = \frac{1}{T-1}\sum_{t=1}^{T-1}\mathbf{1}\left[(\tilde u_{t+1}-\tilde u_t)\cdot(\tilde u_t-\tilde u_{t-1}) < 0\right]
```

Control jerk / control entropy:

```math
H_u = \frac{1}{T-1}\sum_{t=1}^{T-1}\|(\tilde u_{t+1}-\tilde u_t) - (\tilde u_t-\tilde u_{t-1})\|
```

## 7. Main summary table

Create one table aggregated over rollouts:

| Model | Task | Success % | Final error | Best error | AUC | Bad-step % | Osc. count | Gain | Gain SD | Useful gain | Harmful gain | Flip % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

Also create an overall model phenotype table aggregated across tasks:

| Model | Success % | Final error | AUC | Bad-step % | Gain | Gain SD | Useful gain | Flip % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

## 8. Figures

- [ ] Error vs iteration for representative rollouts.
- [ ] Model × task heatmap using success rate or final error.
- [ ] Controller gain vs final error scatter.
- [ ] Bad-step rate vs useful gain scatter.
- [ ] Failure-mode distribution per model.
- [ ] Shape snapshots for representative successes and failures.

## 9. Core interpretation

Main claim to test:

> LLMs exhibit model-specific closed-loop controller phenotypes. Some models are aggressive and unstable, some are conservative and stagnant, and some reduce error efficiently. General model capability does not necessarily imply closed-loop physical-control reliability.

Important distinctions:

- High success does not always mean stable control.
- Smooth error does not always mean good control.
- High gain can produce fast progress or nonlinear failure.
- Low gain can avoid catastrophic loops but stagnate.
- Best error and final error should both be reported because some models find a good state and then destroy it.
