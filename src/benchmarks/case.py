"""
src/benchmarks/case.py
-----------------------
BenchmarkCase — the abstract base class every benchmark must subclass.

CONTRACT
--------
Each case file in cases/ defines one subclass and calls register(instance)
at module level.  The subclass sets the class attributes declared below.
loss_fn and reference_fn have default implementations that dispatch on
loss_type — no override needed for standard cases.

Required class attributes:
    name               str    registry key; must match the --target CLI argument
    physics_type       str    "beam" or "rod"
    target_description str    plain-English goal sent verbatim to the LLM prompt
    loss_description   str    explanation of the loss metric for the LLM
    loss_type          str    "curvature", "node", or "tip"

Optional class attributes:
    success_threshold  float  error_percent below which the task is solved (default 0.05)
    walls              list   extra walls added on top of config.WALLS at startup; [] = none
    reference_kappa_fn        staticmethod(kappa_fn) or pre-computed kappa array;
                              required when loss_type is "curvature" or "node".
                              Must be wrapped in staticmethod() when set to a function —
                              Python binds bare class-attribute functions to the instance.
    reference_tip_fn          (target_x, target_y) tuple; required when loss_type is "tip"

WHY THIS FILE EXISTS
--------------------
__init_subclass__ catches missing required attributes at import time — you
get a clear TypeError the moment the module loads, not a cryptic AttributeError
three iterations into a run.
"""

from __future__ import annotations

import numpy as np
from abc import ABC

from src.benchmarks.beam_losses import (
    build_ideal_from_edge_profile,
    compute_curvature_loss,
    compute_node_loss,
    compute_tip_loss,
    describe_curvature_loss,
    describe_node_loss,
    describe_tip_loss,
)


class BenchmarkCase(ABC):

    name:               str
    physics_type:       str
    target_description: str
    loss_type:          str   # "curvature", "node", or "tip"
    success_threshold:  float = 0.05
    walls:              list  = []     # extra walls merged with config.WALLS at startup
    reference_kappa_fn         = None  # staticmethod(kappa_fn) or pre-computed kappa array
    reference_tip_fn           = None  # (target_x, target_y) for tip-reach cases

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Skip abstract intermediate classes that haven't resolved all attributes yet.
        if getattr(cls, '__abstractmethods__', None):
            return
        for attr in ('name', 'physics_type', 'target_description', 'loss_type'):
            if not isinstance(getattr(cls, attr, None), str):
                raise TypeError(
                    f"{cls.__name__} must define class attribute '{attr}' as a string"
                )
        valid = ("curvature", "node", "tip")
        if cls.loss_type not in valid:
            raise TypeError(
                f"{cls.__name__}.loss_type must be one of {valid!r}, got {cls.loss_type!r}"
            )

    def loss_fn(self, q: np.ndarray, ps) -> dict:
        """
        Evaluate how close the beam is to the target shape.

        q  — full DOF vector (np.ndarray, length 2*nv)
        ps — PhysicalSystemBase instance

        Returns dict with at least {"error_percent": float}.
        Dispatches on loss_type using reference_kappa_fn or reference_tip_fn.
        """
        if self.loss_type == "curvature":
            return {"error_percent": compute_curvature_loss(q, self.reference_kappa_fn, ps)}
        if self.loss_type == "node":
            return {"error_percent": compute_node_loss(q, self.reference_kappa_fn, ps)}
        if self.loss_type == "tip":
            target_x, target_y = self.reference_tip_fn
            nv    = ps.nv if ps else len(q) // 2
            nodes = [(float(q[2*i]), float(q[2*i+1])) for i in range(nv)]
            return {"error_percent": compute_tip_loss(nodes, target_x, target_y, ps.rod_length)}
        raise ValueError(f"Unknown loss_type: {self.loss_type!r}")

    def describe_fn(self, q, ps) -> str:
        """Natural language description of target vs current state for the LLM prompt."""
        if self.loss_type == "curvature":
            return describe_curvature_loss(q, ps, self)
        if self.loss_type == "node":
            return describe_node_loss(q, ps, self)
        if self.loss_type == "tip":
            return describe_tip_loss(q, ps, self)
        return ""

    def reference_fn(
        self,
        x0: float = 0.0,
        y0: float = 0.0,
        theta: float = -np.pi / 2,
        ps=None,
    ) -> list | None:
        """
        Ideal reference shape anchored at node 0's actual position and slope.

        x0, y0 — position of node 0 extracted from q
        theta   — tangent angle at node 0, CCW from +X (standard). -pi/2 = pointing straight down.

        Derived from reference_kappa_fn via build_ideal_from_edge_profile.
        Returns None when reference_kappa_fn is not set (tip-reach cases).
        """
        kappa = self.reference_kappa_fn
        if kappa is None:
            return None
        rod_length = ps.rod_length if ps else 1.0
        nv         = ps.nv         if ps else 21
        return build_ideal_from_edge_profile(kappa, nv, rod_length, x0=x0, y0=y0, theta=theta)
