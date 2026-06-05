# ---------------------------------------------------------------------------
# src/models/physical_system.py
# ---------------------------------------------------------------------------
# PhysicalSystem TypedDict and factory function.
#
# make_physical_system(config) reads raw parameters from config.py,
# computes all derived quantities once, and returns a plain dict that
# can be stored in ESLState and passed directly to physics tools.
#
# Tools never import config — they read from PhysicalSystem only.
# ---------------------------------------------------------------------------

import numpy as np
from typing_extensions import TypedDict


class PhysicalSystem(TypedDict):
    # Identity
    type:       str          # "beam" (future: "rod", "shell", ...)

    # Raw parameters (from config)
    nv:         int
    rho:        float        # density (kg/m³)
    Y:          float        # Young's modulus (Pa)
    r0:         float        # cross-sectional radius (m)
    rod_length: float        # total beam length (m)
    visc:       float        # viscous damping (Pa·s)
    wall_H:                  float   # wall position: y = wall_H (m)
    friction:                bool    # True = fix x and y at contact; False = fix y only
    boundary_velocity:       float   # translational speed of node 0 (m/s)
    boundary_angle_velocity: float   # rotational speed of node 0 (rad/s)

    # Derived (computed once by make_physical_system)
    EI:               float  # bending stiffness = Y π r0⁴ / 4
    EA:               float  # axial stiffness   = Y π r0²
    deltaL:           float  # edge length = rod_length / (nv - 1)
    m:                list   # mass vector, length 2*nv (diagonal of mMat)
    W:                list   # gravity force vector, length 2*nv
    C_diag:           list   # diagonal of damping matrix C, length 2*nv
    observable_nodes: list   # node indices shown to orchestrator — set once, never changed


def make_physical_system(cfg=None) -> PhysicalSystem:
    """
    Build a PhysicalSystem from config.py (or a config-like object for testing).

    Parameters
    ----------
    cfg : module or object, optional
        Must expose NV, RHO, Y, R0, ROD_LENGTH, VISC, WALL_H, FRICTION.
        Defaults to the project config module.

    Returns
    -------
    PhysicalSystem
    """
    if cfg is None:
        import config as cfg

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

    # Observable nodes: node 0, N_PARTIAL-2 interior nodes, node nv-1.
    # Computed once here — never recomputed during a run.
    interior = np.round(np.linspace(1, nv - 2, n_partial - 2)).astype(int).tolist()
    observable_nodes = [0] + interior + [nv - 1]

    deltaL = rod_length / (nv - 1)
    EI     = Y * np.pi * r0**4 / 4
    EA     = Y * np.pi * r0**2

    # Mass vector (diagonal of mMat)
    total_mass = rho * np.pi * r0**2 * rod_length
    m = np.zeros(2 * nv)
    for k in range(nv):
        dm = total_mass / (nv - 1) / 2 if (k == 0 or k == nv - 1) else total_mass / (nv - 1)
        m[2*k]     = dm
        m[2*k + 1] = dm

    # Gravity force vector
    # Small x-perturbation (0.001 g) breaks left-right symmetry for the solver.
    g = np.array([0.001, -9.8])
    W = np.zeros(2 * nv)
    for k in range(nv):
        W[2*k]     = m[2*k]     * g[0]
        W[2*k + 1] = m[2*k + 1] * g[1]

    # Diagonal of damping matrix C
    C_diag = np.zeros(2 * nv)
    for k in range(nv):
        C_diag[2*k]     = visc * deltaL
        C_diag[2*k + 1] = visc * deltaL

    return PhysicalSystem(
        type       = "beam",
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


def initial_q0(ps: PhysicalSystem) -> list:
    """Straight vertical initial configuration: nodes at (0,0), (0,-deltaL), ..."""
    nv     = ps["nv"]
    deltaL = ps["deltaL"]
    q = np.zeros(2 * nv)
    for i in range(nv):
        q[2*i]     = 0.0
        q[2*i + 1] = -i * deltaL
    return q.tolist()


def initial_u0(ps: PhysicalSystem) -> list:
    """Zero velocity initial condition."""
    return [0.0] * (2 * ps["nv"])
