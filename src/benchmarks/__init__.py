from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Dict

from src.benchmarks.case import BenchmarkCase


__all__ = ["BenchmarkCase", "register", "get_benchmark", "list_benchmarks"]


_REGISTRY: Dict[str, BenchmarkCase] = {}
_discovered = False


def register(case: BenchmarkCase) -> None:
    """Register a case instance. Called at module level in each shape file."""
    _REGISTRY[case.name] = case


def _discover() -> None:
    """
    Import every *.py file in shapes/ to trigger their register() calls.
    Runs once; skips modules already in sys.modules (handles running a shape
    file directly as __main__ via python -m).
    """
    global _discovered
    if _discovered:
        return
    _discovered = True
    shapes_dir = Path(__file__).parent / "shapes"
    for path in sorted(shapes_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        module_name = f"src.benchmarks.shapes.{path.stem}"
        if module_name not in sys.modules:
            importlib.import_module(module_name)


def get_benchmark(name: str) -> BenchmarkCase:
    """Look up a benchmark by name. Raises KeyError with helpful message."""
    _discover()
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown benchmark '{name}'. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_benchmarks() -> list:
    """Return all registered benchmark names, sorted."""
    _discover()
    return sorted(_REGISTRY)
