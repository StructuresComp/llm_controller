# ---------------------------------------------------------------------------
# src/tests/test_all_shapes.py
# ---------------------------------------------------------------------------
# Registry-wide benchmark validation.
#
# For every registered benchmark:
#   1. Generate the ideal shape via reference_fn().
#   2. Feed it into loss_fn() — expected error ≈ 0%.
#   3. Assert error < 1% (round-trip consistency check).
#   4. Plot all shapes in a grid.
#
# Run from project root:
#   python -m src.tests.test_all_shapes
# ---------------------------------------------------------------------------

import sys
import matplotlib.pyplot as plt

from src.benchmarks import _REGISTRY, list_benchmarks
from src.benchmarks.visualization import plot_grid


PASS_THRESHOLD = 1.0   # % — reference shape must score below this


def run() -> bool:
    names = list_benchmarks()

    print("=" * 60)
    print(f"BENCHMARK VALIDATION  ({len(names)} registered)")
    print("=" * 60)

    all_pass = True
    for name in names:
        bm     = _REGISTRY[name]
        nodes  = bm.reference_fn()
        result = bm.loss_fn(nodes)
        err    = result.get("error_percent", 100.0)
        status = "PASS" if err < PASS_THRESHOLD else "FAIL"
        if err >= PASS_THRESHOLD:
            all_pass = False
        diag = result.get("diagnostics", "")
        print(f"  [{status}]  {name:6s}  {err:.4f}%   {diag}")

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
