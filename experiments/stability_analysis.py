"""
experiments/stability_analysis.py
---------------------------------
Bridge from the LLM controller loop to the LLM_Stability / HSKD framework.

Treats one controller run as a trajectory of a stochastic dynamical system and
estimates its Koopman spectral radius rho(A):

    rho  < 1  ->  the loop contracts toward a fixed point (stable correction)
    rho >= 1  ->  it does not settle (persistent oscillation / divergence)

Pipeline (mirrors the LLM_Stability EDMD/Koopman method):
  1. Per-iteration diagnostic observables, read from the run JSONL:
       err      = error_percent / 100      external physics criterion (artifact error)
       d_err    = err_{k-1} - err_k         improvement / proposal utility
       bc_step  = ||bc_k - bc_{k-1}||       control effort (over/undercorrection)
       bc_cos   = cos(dbc_k, dbc_{k-1})     direction reversal (oscillation)
  2. Hankel delay-embedding (depth d) to expose the linear dynamics.
  3. EDMD: ridge least-squares affine fit  z_{t+1} ~= A z_t + b ; rho = max|eig(A)|.
  4. Online rho_k over growing windows, for early-warning lead-time studies.

Why this testbed matters: unlike the language-only drafting loops, `err` here is
an EXTERNAL deterministic physics criterion, not an LLM self-report. A rho that
predicts the run outcome is therefore NOT tautological.

Usage:
    python experiments/stability_analysis.py logs/run_xxx.jsonl
    python experiments/stability_analysis.py --self-test
"""

import json
import sys

import numpy as np


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_trajectory(jsonl_path: str) -> list:
    """Aggregate JSONL DOF-snapshots into one record per orchestrator iteration.

    Each iteration writes several DOF snapshots (q_history frames) that share the
    iteration's error_percent and BC. We keep the LAST snapshot of each iteration
    (the settled configuration). Returns records ordered by iteration:
        [{"iteration": k, "error_percent": e, "bc": [x0, y0, theta]}, ...]
    """
    by_iter = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            it = s.get("iteration")
            if it is None:
                continue
            rec = by_iter.setdefault(int(it), {"iteration": int(it)})
            rec["error_percent"] = float(s.get("error_percent", rec.get("error_percent", 100.0)))
            bc = s.get("bc") or {}
            if bc:
                rec["bc"] = [float(bc.get("x0", 0.0)), float(bc.get("y0", 0.0)), float(bc.get("theta", 0.0))]
            # Diagnostic signals (present in runs instrumented after 2026-06; older logs lack them).
            if "solved" in s:
                rec["solved"] = bool(s["solved"])
            if "converged" in s:
                rec["converged"] = bool(s["converged"])
            if "safety_rejections" in s:
                rec["safety_rejections"] = int(s["safety_rejections"])
            if s.get("convergence_reason"):
                rec["convergence_reason"] = s["convergence_reason"]
    out = [by_iter[k] for k in sorted(by_iter)]
    for r in out:
        r.setdefault("bc", [0.0, 0.0, 0.0])
        r.setdefault("solved", False)
        r.setdefault("converged", False)
        r.setdefault("safety_rejections", 0)
    return out


# ---------------------------------------------------------------------------
# Observables
# ---------------------------------------------------------------------------

def build_observables(records: list):
    """Per-iteration diagnostic vector. Returns (O [T, 4], names)."""
    errs = np.array([r["error_percent"] / 100.0 for r in records])
    bcs = np.array([r["bc"] for r in records])
    T = len(records)

    d_err = np.zeros(T)
    d_err[1:] = errs[:-1] - errs[1:]            # positive = improving

    dbc = np.zeros((T, 3))
    dbc[1:] = bcs[1:] - bcs[:-1]
    bc_step = np.linalg.norm(dbc, axis=1)

    bc_cos = np.zeros(T)
    for k in range(2, T):
        a, b = dbc[k], dbc[k - 1]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        bc_cos[k] = float(a @ b / (na * nb)) if na > 1e-9 and nb > 1e-9 else 0.0

    O = np.column_stack([errs, d_err, bc_step, bc_cos])
    return O, ["err", "d_err", "bc_step", "bc_cos"]


# ---------------------------------------------------------------------------
# EDMD / Koopman
# ---------------------------------------------------------------------------

def _scale(O: np.ndarray) -> np.ndarray:
    """Per-observable std scaling. A diagonal change of coordinates leaves the
    spectral radius invariant, so this only improves conditioning."""
    sd = O.std(axis=0)
    sd[sd < 1e-9] = 1.0
    return O / sd


def hankel_embed(O: np.ndarray, depth: int) -> np.ndarray:
    """Delay-embed: row t = [o_{t-depth+1}, ..., o_t]. Returns [T-depth+1, m*depth]."""
    T = O.shape[0]
    if T < depth:
        return np.empty((0, 0))
    return np.array([O[t - depth + 1:t + 1].reshape(-1) for t in range(depth - 1, T)])


def edmd_spectral_radius(O: np.ndarray, depth: int = 2, ridge: float = 1e-6):
    """Ridge-regularised affine EDMD. Returns (rho, eigenvalues).

    Fits z_{t+1} ~= A z_t + b (constant column absorbs the equilibrium offset, so
    we do not mean-subtract and bias short series). rho = max |eig(A)|.
    """
    Z = hankel_embed(_scale(O), depth)
    n = Z.shape[0]
    if n < depth + 2:
        return float("nan"), np.array([])

    X = Z[:-1].T                                  # [f, n-1]
    Y = Z[1:].T                                   # [f, n-1]
    f = X.shape[0]
    Xa = np.vstack([X, np.ones((1, X.shape[1]))]) # affine augmentation
    G = Xa @ Xa.T
    M = Y @ Xa.T @ np.linalg.inv(G + ridge * np.eye(f + 1))
    A = M[:, :f]                                  # dynamics block (drop intercept col)
    ev = np.linalg.eigvals(A)
    return float(np.max(np.abs(ev))), ev


def online_rho(O: np.ndarray, depth: int = 2, min_points: int = None, ridge: float = 1e-6):
    """rho_k fitted on the first k iterations, for k = min_points .. T.

    min_points defaults to the smallest window for which the embedded
    least-squares fit is over-determined (more snapshot pairs than embedded
    dimensions); starting earlier yields ill-conditioned, meaningless rho_k.
    """
    if min_points is None:
        m = O.shape[1]
        min_points = depth * (m + 1) + 3      # > embedded state dim => well-posed
    out = []
    for k in range(min_points, O.shape[0] + 1):
        rho, _ = edmd_spectral_radius(O[:k], depth, ridge)
        out.append((k, rho))
    return out


# ---------------------------------------------------------------------------
# Self-test: recover a known spectral radius from a partial observation
# ---------------------------------------------------------------------------

def _synthetic_linear(rho_true: float, T: int = 120, theta: float = 0.35) -> np.ndarray:
    """2D rotation-scaling with eigenvalue magnitude rho_true; observe coord 0 only.
    Hankel embedding must reconstruct the hidden 2D dynamics from this scalar."""
    R = rho_true * np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]])
    x = np.array([1.0, 0.2])
    xs = []
    for _ in range(T):
        xs.append(x.copy())
        x = R @ x
    return np.array(xs)[:, 0:1]


def self_test() -> bool:
    """Validate the EDMD/Hankel pipeline against systems with known rho."""
    print("EDMD self-test — recover rho from a scalar observation of a 2D linear system:")
    ok = True
    for rho_true in [0.70, 0.90, 1.05, 1.20]:
        obs = _synthetic_linear(rho_true, T=120)
        rho_est, _ = edmd_spectral_radius(obs, depth=4, ridge=1e-10)
        err = abs(rho_est - rho_true)
        status = "PASS" if err < 2e-2 else "FAIL"
        ok = ok and err < 2e-2
        print(f"   true rho = {rho_true:.3f}   estimated = {rho_est:.3f}   |err| = {err:.4f}   [{status}]")
    print("RESULT:", "ALL PASS" if ok else "SOME FAILED")
    return ok


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def analyse_run(jsonl_path: str, depth: int = 2):
    records = load_trajectory(jsonl_path)
    T = len(records)
    print(f"Run: {jsonl_path}")
    print(f"Iterations (trajectory length): {T}")
    if T < 3:
        print("Too short to fit a Koopman surrogate (need >= 3 iterations).")
        return

    O, names = build_observables(records)
    errs = O[:, 0]
    print(f"error_percent: first={errs[0]*100:.1f}%  last={errs[-1]*100:.1f}%  "
          f"min={errs.min()*100:.1f}%")

    # Auto-reduce embedding depth if the trajectory is short.
    eff_depth = depth
    while eff_depth > 1 and hankel_embed(O, eff_depth).shape[0] < eff_depth * O.shape[1] + 2:
        eff_depth -= 1
    underdetermined = hankel_embed(O, eff_depth).shape[0] < eff_depth * O.shape[1] + 2

    rho, ev = edmd_spectral_radius(O, depth=eff_depth)
    print(f"\nKoopman EDMD (observables={names}, depth={eff_depth}):")
    print(f"   spectral radius rho = {rho:.4f}   ->  "
          f"{'STABLE (contracts)' if rho < 1 else 'NOT STABLE (rho >= 1)'}")
    if ev.size:
        top = sorted(ev, key=lambda z: -abs(z))[:3]
        print("   dominant eigenvalues: " + ", ".join(f"{z.real:+.3f}{z.imag:+.3f}j" for z in top))
    if underdetermined:
        print("   [caveat] trajectory shorter than the embedded state dimension — "
              "rho is ridge-regularised and should be read as indicative only.")

    onl = online_rho(O, depth=eff_depth)
    if onl:
        print("\nOnline rho_k (growing window):")
        print("   " + "  ".join(f"k={k}:{r:.3f}" for k, r in onl))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return
    if args[0] == "--self-test":
        sys.exit(0 if self_test() else 1)
    analyse_run(args[0])


if __name__ == "__main__":
    main()
