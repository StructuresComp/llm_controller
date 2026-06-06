# ---------------------------------------------------------------------------
# src/benchmarks/__init__.py
# ---------------------------------------------------------------------------
# Benchmark registry.
#
# A Benchmark bundles everything needed to define one evaluation task:
#   - a rich target description for the LLM prompt
#   - a deterministic loss function:  loss_fn(nodes, ps=None) -> dict
#   - a reference generator:          reference_fn(n, arc_length) -> list[(x,y)]
#   - a success threshold (error_percent below which the task is solved)
#
# Shape files in shapes/ call register() at import time.
# This module imports all shape modules at the bottom to trigger registration.
# ---------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict


@dataclass
class Benchmark:
    name:               str
    physics_type:       str      # "beam" or "rod"
    target_description: str      # rich description passed to the LLM prompt
    loss_fn:            Callable  # (nodes, ps=None) -> dict with error_percent etc.
    reference_fn:       Callable  # (n, arc_length) -> list of (x, y) ideal nodes
    success_threshold:  float = 0.05  # fraction: 0.05 = 5% error


_REGISTRY: Dict[str, Benchmark] = {}


def register(benchmark: Benchmark) -> None:
    """Register a benchmark. Called at module level in each shape file."""
    _REGISTRY[benchmark.name] = benchmark


def get_benchmark(name: str) -> Benchmark:
    """Look up a benchmark by name. Raises KeyError with helpful message."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown benchmark '{name}'. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_benchmarks() -> list:
    """Return all registered benchmark names, sorted."""
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# Import shape modules to trigger registration.
# Add new shape modules here when created.
# ---------------------------------------------------------------------------
from src.benchmarks.shapes import beam_C, beam_U, beam_S, beam_L  # noqa: E402, F401
