from __future__ import annotations

from abc import ABC, abstractmethod


class BenchmarkCase(ABC):
    """
    Abstract base for every benchmark case.

    Subclasses must define three class attributes as strings:
        name               — registry key, e.g. "C"
        physics_type       — "beam" or "rod"
        target_description — rich prose for the LLM prompt

    And implement:
        loss_fn(self, nodes, ps=None, q_ref=None) -> dict

    Optionally override:
        reference_fn(self, x0, y0, theta) -> list[(x, y)] | None
            Anchors the ideal shape at node 0's actual position and slope.
            Returns None (default) for cases that have no reference geometry.

    success_threshold (float, default 0.05) can be overridden as a class
    attribute or left at the default.
    """

    name:               str
    physics_type:       str
    target_description: str
    success_threshold:  float = 0.05

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Skip validation for abstract intermediate classes (they still have
        # unresolved abstract methods).
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
        Evaluate how close nodes is to the target.

        Args:
            nodes  — list of (x, y) positions (observable nodes from q)
            ps     — PhysicalSystemBase instance (beam or rod); may be None
                     in standalone tests
            q_ref  — optional reference from reference_fn(); cases that do
                     node-wise comparison receive it here

        Returns dict with at least:
            error_percent  — 0.0 = perfect, 100.0 = maximally wrong
            diagnostics    — human-readable string
        Multiple named sub-losses may appear alongside error_percent.
        """
        ...

    def reference_fn(
        self,
        x0: float = 0.0,
        y0: float = 0.0,
        theta: float = 0.0,
    ) -> list | None:
        """
        Generate the ideal reference shape anchored at (x0, y0) with slope theta.

        x0, y0 — position of node 0 (extracted from actual q)
        theta   — tangent angle at node 0 (0 = pointing straight down, −Y)

        Parameterised by arc-length s ∈ [0, 1] — invariant to node count.
        Returns None for cases with no reference geometry (tip-target,
        contact-count, etc.).  Override in subclasses that need it.
        """
        return None
