"""
src/benchmarks/case.py
-----------------------
BenchmarkCase — the abstract base class every benchmark must subclass.

CONTRACT
--------
Each case file in cases/ defines one subclass and calls register(instance)
at module level.  The subclass must set three class attributes as strings
and implement loss_fn.  Everything else has a sensible default.

Required class attributes:
    name               str   registry key; must match the --target CLI argument
    physics_type       str   "beam" or "rod"
    target_description str   plain-English goal sent verbatim to the LLM prompt

Optional class attributes:
    success_threshold  float  error_percent below which the task is solved (default 0.05)
    walls              list   case-specific wall geometry; None = use config.WALLS

Required method:
    loss_fn(nodes, ps, q_ref) -> dict
        Must return at minimum {"error_percent": float, "diagnostics": str}.

Optional override:
    reference_fn(x0, y0, theta) -> list[(x,y)] | None
        Ideal shape anchored at node 0's actual position and slope.
        Default returns None (correct for tip-reach / navigation cases).

WHY THIS FILE EXISTS
--------------------
__init_subclass__ catches missing required attributes at import time — you
get a clear TypeError the moment the module loads, not a cryptic AttributeError
three iterations into a run.  The walls = None default lets cli.py resolve
case-specific vs. config-level walls with a single `bm.walls or config.WALLS`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BenchmarkCase(ABC):

    name:               str
    physics_type:       str
    target_description: str
    success_threshold:  float = 0.05
    walls:              list  = None   # None → cli.py falls back to config.WALLS

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Abstract intermediate classes haven't resolved their abstract methods yet;
        # skip validation so they can exist without all attributes set.
        if getattr(cls, '__abstractmethods__', None):
            return
        for attr in ('name', 'physics_type', 'target_description'):
            if not isinstance(getattr(cls, attr, None), str):
                raise TypeError(
                    f"{cls.__name__} must define class attribute '{attr}' as a string"
                )

    @abstractmethod
    def loss_fn(self, nodes: list, ps=None, q_ref=None) -> dict:
        """
        Evaluate how close nodes is to the target shape.

        nodes  — list of (x, y) observable positions from q
        ps     — PhysicalSystemBase instance; may be None in standalone tests
        q_ref  — optional reference from reference_fn() for node-wise comparison

        Must return a dict with at least:
            error_percent  — 0.0 = perfect match, 100.0 = maximally wrong
            diagnostics    — human-readable string for logs and LLM feedback
        """
        ...

    def reference_fn(
        self,
        x0: float = 0.0,
        y0: float = 0.0,
        theta: float = 0.0,
    ) -> list | None:
        """
        Ideal reference shape anchored at node 0's actual position and slope.

        x0, y0 — position of node 0 extracted from q
        theta   — tangent angle at node 0 (0 = pointing straight down, −Y)

        Parameterised by arc-length s ∈ [0, 1] — invariant to node count.
        Returns None for cases that have no geometric reference (tip-reach,
        navigation, contact-count, etc.).  Override in subclasses that need it.
        """
        return None
