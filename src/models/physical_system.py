# ---------------------------------------------------------------------------
# src/models/physical_system.py
# ---------------------------------------------------------------------------
# Physical system class hierarchy and factory.
#
# PhysicalSystemBase  — empty base class; use for isinstance checks / type hints
# BeamPhysicalSystem  — 2D elastic beam; ndof = 2 * nv
# RodPhysicalSystem   — 3D elastic rod stub; ndof = 3 * nv - 1 (fields TBD)
#
# make_physical_system(config) reads config.py, computes derived quantities,
# and returns the appropriate subclass.  Tools never import config — they
# read from the PhysicalSystem instance only.
# ---------------------------------------------------------------------------

import numpy as np


# ---------------------------------------------------------------------------
# BASE CLASS
# ---------------------------------------------------------------------------

class PhysicalSystemBase:
    """Empty base class for all physical system types."""
    pass


# ---------------------------------------------------------------------------
# BEAM
# ---------------------------------------------------------------------------

class BeamPhysicalSystem(PhysicalSystemBase):
    """
    Pre-computed physical and derived quantities for a 2D elastic beam.
    Built once at startup by make_physical_system(); never modified during a run.

    DOF layout: q = [x0, y0, x1, y1, ..., x_{nv-1}, y_{nv-1}]  (length = ndof = 2*nv)
    """

    def __init__(
        self,
        nv:         int,
        rho:        float,
        Y:          float,
        r0:         float,
        rod_length: float,
        visc:       float,
        wall_H:                  float,
        friction:                bool,
        boundary_velocity:       float,
        boundary_angle_velocity: float,
        EI:               float,
        EA:               float,
        deltaL:           float,
        m:                list,
        W:                list,
        C_diag:           list,
        observable_nodes: list,
    ):
        self.type       = "beam"
        self.nv         = nv
        self.ndof       = 2 * nv

        self.rho        = rho
        self.Y          = Y
        self.r0         = r0
        self.rod_length = rod_length
        self.visc       = visc

        self.wall_H                  = wall_H
        self.friction                = friction
        self.boundary_velocity       = boundary_velocity
        self.boundary_angle_velocity = boundary_angle_velocity

        self.EI               = EI
        self.EA               = EA
        self.deltaL           = deltaL
        self.m                = m
        self.W                = W
        self.C_diag           = C_diag
        self.observable_nodes = observable_nodes


# ---------------------------------------------------------------------------
# ROD  (stub — fields TBD when rod support lands)
# ---------------------------------------------------------------------------

class RodPhysicalSystem(PhysicalSystemBase):
    """
    Pre-computed physical and derived quantities for a 3D elastic rod.
    DOF layout: q = [x0,y0,z0, x1,y1,z1, ..., theta0, ..., theta_{nv-2}]
    ndof = 3*nv + (nv-1) = 3*nv - 1   (positions + twist angles)
    """

    def __init__(self, nv: int, rod_length: float, **kwargs):
        self.type       = "rod"
        self.nv         = nv
        self.ndof       = 3 * nv - 1
        self.rod_length = rod_length
        # remaining fields set from kwargs; proper __init__ TBD with rod support
        for k, v in kwargs.items():
            setattr(self, k, v)


# ---------------------------------------------------------------------------
# FACTORY
# ---------------------------------------------------------------------------

def make_physical_system(cfg=None) -> BeamPhysicalSystem:
    """
    Build a BeamPhysicalSystem from config.py (or a config-like object for testing).
    Returns BeamPhysicalSystem.  Extend to dispatch on cfg.physics_type when rod
    support lands.
    """
    if cfg is None:
        import src.core.config as cfg

    nv         = cfg.NV
    rho        = cfg.RHO
    Y          = cfg.Y
    r0         = cfg.R0
    rod_length = cfg.ROD_LENGTH
    visc       = cfg.VISC
    wall_H                  = cfg.WALL_H
    friction                = cfg.FRICTION
    boundary_velocity       = cfg.BOUNDARY_VELOCITY
    boundary_angle_velocity = cfg.BOUNDARY_ANGLE_VELOCITY

    if boundary_velocity <= 0:
        raise ValueError(f"BOUNDARY_VELOCITY must be > 0, got {boundary_velocity}")
    if boundary_angle_velocity <= 0:
        raise ValueError(f"BOUNDARY_ANGLE_VELOCITY must be > 0, got {boundary_angle_velocity}")

    n_partial = cfg.N_PARTIAL
    if n_partial < 2:
        raise ValueError(f"N_PARTIAL must be >= 2, got {n_partial}")
    if n_partial > nv:
        raise ValueError(f"N_PARTIAL ({n_partial}) cannot exceed NV ({nv})")

    interior = np.round(np.linspace(1, nv - 2, n_partial - 2)).astype(int).tolist()
    observable_nodes = [0] + interior + [nv - 1]

    deltaL = rod_length / (nv - 1)
    EI     = Y * np.pi * r0**4 / 4
    EA     = Y * np.pi * r0**2

    total_mass = rho * np.pi * r0**2 * rod_length
    m = np.zeros(2 * nv)
    for k in range(nv):
        dm = total_mass / (nv - 1) / 2 if (k == 0 or k == nv - 1) else total_mass / (nv - 1)
        m[2*k]     = dm
        m[2*k + 1] = dm

    g = np.array([0.001, -9.8])
    W = np.zeros(2 * nv)
    for k in range(nv):
        W[2*k]     = m[2*k]     * g[0]
        W[2*k + 1] = m[2*k + 1] * g[1]

    C_diag = np.zeros(2 * nv)
    for k in range(nv):
        C_diag[2*k]     = visc * deltaL
        C_diag[2*k + 1] = visc * deltaL

    return BeamPhysicalSystem(
        nv         = nv,
        rho        = rho,
        Y          = Y,
        r0         = r0,
        rod_length = rod_length,
        visc       = visc,
        wall_H                  = wall_H,
        friction                = friction,
        boundary_velocity       = boundary_velocity,
        boundary_angle_velocity = boundary_angle_velocity,
        EI         = float(EI),
        EA         = float(EA),
        deltaL     = float(deltaL),
        m                = m.tolist(),
        W                = W.tolist(),
        C_diag           = C_diag.tolist(),
        observable_nodes = observable_nodes,
    )


# ---------------------------------------------------------------------------
# INITIAL CONDITIONS
# ---------------------------------------------------------------------------

def initial_q0(ps: BeamPhysicalSystem) -> list:
    """Straight vertical initial configuration: nodes at (0,0), (0,-deltaL), ..."""
    q = np.zeros(ps.ndof)
    for i in range(ps.nv):
        q[2*i]     = 0.0
        q[2*i + 1] = -i * ps.deltaL
    return q.tolist()


def initial_u0(ps: BeamPhysicalSystem) -> list:
    """Zero velocity initial condition."""
    return [0.0] * ps.ndof
