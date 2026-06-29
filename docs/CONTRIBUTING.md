# Contributing

We will add a benchmark called `"myshape"` that runs as:
```bash
python3 cli.py --target myshape
```

---

## Adding a new benchmark case

**One new file in `src/benchmarks/cases/beam_myshape.py`. Nothing else changes.**

The registry auto-discovers all `*.py` files in `cases/` at import time.

```
src/benchmarks/
├── beam_losses.py          — loss functions + LLM descriptions (curvature, node, tip)
├── loss_beam/              — reusable curvature profile functions
│   ├── profile_C.py        — C-shape (uniform curvature + terminal curls)
│   ├── profile_S.py        — S-curve
│   ├── profile_U.py        — Semicircle
│   └── profile_L.py        — L-shape
├── case.py                 — BenchmarkCase base class
├── __init__.py             — registry: register(), get_benchmark()
└── cases/                  — your file goes here (auto-discovered)
    ├── beam_C.py
    ├── beam_arc.py
    └── beam_myshape.py     ← new file
```

### Required class attributes

| Attribute | Type | Description |
|---|---|---|
| `name` | str | Registry key — matches the `--target` argument |
| `physics_type` | str | `"beam"` or `"rod"` |
| `target_description` | str | Plain-English goal sent verbatim to the LLM prompt |
| `loss_type` | str | `"curvature"`, `"node"`, or `"tip"` — see below |

### Choosing loss_type

**`"curvature"`** — target is a curvature profile along the beam. Requires `reference_kappa_fn`:
a function `kappa(s_norm, rod_length)` returning curvature (1/m) at normalised arc-length `s ∈ [0, 1]`.

For simple shapes, define inline:
```python
def kappa_myshape(s_norm, rod_length=1.0):
    return np.full_like(np.asarray(s_norm, dtype=float), 2.0 / rod_length)

class MyShape(BenchmarkCase):
    loss_type          = "curvature"
    reference_kappa_fn = staticmethod(kappa_myshape)
```

For more complex profiles, add a file to `loss_beam/` and import from there:
```python
from src.benchmarks.loss_beam.profile_C import kappa_c

class MyShape(BenchmarkCase):
    loss_type          = "curvature"
    reference_kappa_fn = staticmethod(kappa_c)
```

See `beam_arc.py` (inline) and `beam_C.py` (imported) for complete examples.

---

**`"node"`** — same as `"curvature"` but error is measured as RMS positional deviation of nodes
rather than curvature deviation. Requires `reference_kappa_fn` (used to generate reference node positions).

---

**`"tip"`** — target is a single tip position. Requires `reference_tip_fn`: a `(target_x, target_y)` tuple.

```python
class MyShape(BenchmarkCase):
    loss_type        = "tip"
    reference_tip_fn = (0.05, -0.11)
```

See `beam_tip_reach.py` and `beam_snake_L.py` for complete examples.

### Optional class attributes

| Attribute | Default | Description |
|---|---|---|
| `success_threshold` | `0.05` | `error_percent` below which the task is solved |
| `walls` | `[]` | Wall segments defining the benchmark environment (merged with `config.WALLS`) |

### Walls

**In the case file** — for walls that are part of the benchmark definition (permanent geometry, e.g. a tube or channel):

```python
class MyShape(BenchmarkCase):
    walls = [
        {"P0": [-0.01, -0.12], "P1": [ 0.12, -0.12]},   # floor
        {"P0": [-0.01, -0.02], "P1": [-0.01, -0.12]},   # left wall
    ]
```

Each wall is `{"P0": [x, y], "P1": [x, y]}` — axis-aligned only (horizontal or vertical).
The outward normal is `(P1 - P0)` rotated 90° CCW, which encodes which side is the allowed region.

**From the CLI** — for quick experiments with a single horizontal floor, without editing any file:

```bash
python3 cli.py --target myshape --H 0.12   # places floor at y = -0.12
```

### Complete example

```python
import numpy as np
from src.benchmarks import BenchmarkCase, register
from src.visualization.run_time_visual import plot_shape


def kappa_myshape(s_norm, rod_length=1.0):
    """Constant curvature: radius = rod_length / 2."""
    return np.full_like(np.asarray(s_norm, dtype=float), 2.0 / rod_length)


class MyShape(BenchmarkCase):
    name               = "myshape"
    physics_type       = "beam"
    loss_type          = "curvature"
    reference_kappa_fn = staticmethod(kappa_myshape)
    target_description = (
        "A uniform circular arc with constant positive curvature. "
        "Positive curvature bends toward +X when the clamped end points downward. "
        "The radius equals half the beam length, giving a total turning angle of 2 radians."
    )
    success_threshold  = 0.05


register(MyShape())


def main():
    case = MyShape()
    ref  = case.reference_fn()
    plot_shape(ref, name="myshape (reference)")


if __name__ == "__main__":
    main()
```

Inspect the reference geometry:
```bash
python3 -m src.benchmarks.cases.beam_myshape
```

Validate all registered benchmarks:
```bash
python3 -m src.tests.test_all_shapes
```
