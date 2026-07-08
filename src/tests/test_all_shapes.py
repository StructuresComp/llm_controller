"""
Registry-wide benchmark validation.

For every registered benchmark:
  1. Call reference_fn() to get the ideal shape (no config needed).
  2. Feed it into loss_fn() — expected error ≈ 0%.
  3. Assert error < PASS_THRESHOLD (round-trip consistency check).
  4. Plot all shapes in a grid.

Run from project root:
    python -m src.tests.test_all_shapes
"""

import sys
from types import SimpleNamespace
import numpy as np
from src.benchmarks import _REGISTRY, list_benchmarks
from src.visualization.run_time_visual import plot_grid

PASS_THRESHOLD = 1.0


def run() -> bool:
    names = list_benchmarks()

    print("=" * 60)
    print(f"BENCHMARK VALIDATION  ({len(names)} registered)")
    print("=" * 60)

    all_pass = True
    for name in names:
        bm  = _REGISTRY[name]
        ref = bm.reference_fn()
        if ref is None:
            print(f"  [SKIP]  {name:6s}  (no reference shape — tip-reach case)")
            continue
        ref        = np.asarray(ref)
        nv         = len(ref)
        arc_length = float(np.sum(np.linalg.norm(np.diff(ref, axis=0), axis=1)))
        ps         = SimpleNamespace(nv=nv, deltaL=arc_length / (nv - 1), rod_length=arc_length)
        q          = ref.flatten()
        result = bm.loss_fn(q, ps)
        err    = result.get("error_percent", 100.0)
        status = "PASS" if err < PASS_THRESHOLD else "FAIL"
        if err >= PASS_THRESHOLD:
            all_pass = False
        print(f"  [{status}]  {name:6s}  {err:.4f}%")

    print("=" * 60)
    print("All passed." if all_pass else "SOME TESTS FAILED.")
    return all_pass


def main():
    ok = run()
    benchmarks = [_REGISTRY[n] for n in list_benchmarks()]
    plot_grid(benchmarks)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
