"""
experiments/behavioral_metrics.py
---------------------------------
The pre-registered closed-loop behavioral metrics from docs/paper_outline.md,
computed over one controller run's per-iteration error/control trajectory.

These are domain-independent quantities (the paper instantiates the same family
on language-revision loops) measured here on physical control:

  improvement_stability   net error reduction relative to total variation
                          (1 = monotone improvement, ~0 = thrash, <0 = net worse)
  bad_step_rate           fraction of steps that increased the error
  oscillation_count       sign changes in the step-to-step error change
  overcorrection_index    mean control step on worsening steps / mean control step
                          (>1 = large moves tend to make things worse)
  persistence             mean cosine between consecutive control updates
                          (1 = coherent direction, ~0 = erratic, <0 = reversing)
  recovery                fraction of the worst-case error recovered by run end
  false_convergence_rate  fraction of iterations the LLM self-reported "converged"
                          while the deterministic evaluator said "not solved"
                          (a proxy for poor infeasibility recognition / overconfidence)

Usage:
    python experiments/behavioral_metrics.py logs/run_xxx.jsonl
    python experiments/behavioral_metrics.py --self-test
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stability_analysis import load_trajectory  # noqa: E402

_EPS = 1e-9


def compute_metrics(records: list) -> dict:
    """Compute the behavioral-metric family from per-iteration records."""
    T = len(records)
    e = np.array([r["error_percent"] / 100.0 for r in records])
    bc = np.array([r["bc"] for r in records], dtype=float)
    solved = np.array([bool(r.get("solved", False)) for r in records])
    converged = np.array([bool(r.get("converged", False)) for r in records])

    m = {"n_iterations": T, "final_error_percent": float(e[-1] * 100) if T else float("nan")}
    if T < 2:
        return {**m, "note": "trajectory too short for step-based metrics"}

    de = np.diff(e)                       # >0 = worsened, <0 = improved
    m["net_improvement"] = float(e[0] - e[-1])
    m["improvement_stability"] = float((e[0] - e[-1]) / (np.sum(np.abs(de)) + _EPS))
    m["bad_step_rate"] = float(np.mean(de > 0))

    # Oscillation: sign changes in the sequence of step-to-step error changes.
    sd = np.sign(de)
    sd = sd[sd != 0]
    m["oscillation_count"] = int(np.sum(sd[1:] != sd[:-1])) if sd.size > 1 else 0

    # Control effort and overcorrection.
    dbc = np.diff(bc, axis=0)
    step = np.linalg.norm(dbc, axis=1)
    m["mean_control_step"] = float(np.mean(step))
    worsened = de > 0
    if np.any(worsened) and np.mean(step) > _EPS:
        m["overcorrection_index"] = float(np.mean(step[worsened]) / (np.mean(step) + _EPS))
    else:
        m["overcorrection_index"] = 0.0

    # Persistence: mean cosine between consecutive control updates.
    coss = []
    for k in range(1, len(dbc)):
        a, b = dbc[k], dbc[k - 1]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > _EPS and nb > _EPS:
            coss.append(float(a @ b / (na * nb)))
    m["persistence"] = float(np.mean(coss)) if coss else 0.0

    # Recovery: how much of the worst error was recovered by the end.
    worst = int(np.argmax(e))
    after_min = float(np.min(e[worst:]))
    span = float(e[worst] - np.min(e))
    m["recovery"] = float((e[worst] - after_min) / (span + _EPS)) if span > _EPS else 1.0

    # Outcome + infeasibility recognition.
    m["solved"] = bool(np.any(solved))
    m["solve_iteration"] = int(np.argmax(solved) + 1) if np.any(solved) else None
    m["false_convergence_rate"] = float(np.mean(converged & ~solved))

    return m


def report(jsonl_path: str):
    records = load_trajectory(jsonl_path)
    m = compute_metrics(records)
    print(f"Run: {jsonl_path}   (iterations: {m['n_iterations']})")
    order = [
        "solved", "solve_iteration", "final_error_percent", "net_improvement",
        "improvement_stability", "bad_step_rate", "oscillation_count",
        "overcorrection_index", "persistence", "recovery",
        "mean_control_step", "false_convergence_rate",
    ]
    for k in order:
        if k in m:
            v = m[k]
            print(f"   {k:<24} {v}")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _records_from_errors(errs):
    return [{"error_percent": e * 100, "bc": [0.0, 0.0, 0.0], "solved": False, "converged": False}
            for e in errs]


def self_test() -> bool:
    ok = True

    mono = compute_metrics(_records_from_errors([0.5, 0.4, 0.3, 0.2, 0.1]))
    checks = [
        ("monotone improvement_stability==1", abs(mono["improvement_stability"] - 1.0) < 1e-6),
        ("monotone bad_step_rate==0", mono["bad_step_rate"] == 0.0),
        ("monotone oscillation_count==0", mono["oscillation_count"] == 0),
        ("monotone recovery==1", abs(mono["recovery"] - 1.0) < 1e-6),
    ]

    osc = compute_metrics(_records_from_errors([0.5, 0.3, 0.5, 0.3, 0.5]))
    checks += [
        ("oscillating bad_step_rate==0.5", abs(osc["bad_step_rate"] - 0.5) < 1e-9),
        ("oscillating oscillation_count==3", osc["oscillation_count"] == 3),
        ("oscillating improvement_stability==0", abs(osc["improvement_stability"]) < 1e-9),
    ]

    # False convergence: LLM says converged but evaluator never solved.
    recs = _records_from_errors([0.5, 0.4, 0.4])
    for r in recs:
        r["converged"] = True
    fc = compute_metrics(recs)
    checks.append(("false_convergence_rate==1", abs(fc["false_convergence_rate"] - 1.0) < 1e-9))

    print("Behavioral-metrics self-test:")
    for label, passed in checks:
        print(f"   [{'PASS' if passed else 'FAIL'}] {label}")
        ok = ok and passed
    print("RESULT:", "ALL PASS" if ok else "SOME FAILED")
    return ok


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return
    if args[0] == "--self-test":
        sys.exit(0 if self_test() else 1)
    report(args[0])


if __name__ == "__main__":
    main()
