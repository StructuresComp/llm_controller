import numpy as np
import matplotlib.pyplot as plt


# Helper function
def crossMat(a):
    """
    Returns the cross product matrix of vector 'a'.

    Parameters:
    a : np.ndarray
        A 3-element array representing a vector.

    Returns:
    A : np.ndarray
        The cross product matrix corresponding to vector 'a'.
    """
    A = np.array([[0, -a[2], a[1]],
                  [a[2], 0, -a[0]],
                  [-a[1], a[0], 0]])

    return A

# Gradient and Hessian of elastic energies
def gradEb(xkm1, ykm1, xk, yk, xkp1, ykp1, curvature0, l_k, EI):
    """
    Returns the derivative of bending energy E_k^b with respect to
    x_{k-1}, y_{k-1}, x_k, y_k, x_{k+1}, and y_{k+1}.

    Parameters:
    xkm1, ykm1 : float
        Coordinates of the previous node (x_{k-1}, y_{k-1}).
    xk, yk : float
        Coordinates of the current node (x_k, y_k).
    xkp1, ykp1 : float
        Coordinates of the next node (x_{k+1}, y_{k+1}).
    curvature0 : float
        Discrete natural curvature at node (xk, yk).
    l_k : float
        Voronoi length of node (xk, yk).
    EI : float
        Bending stiffness.

    Returns:
    dF : np.ndarray
        Derivative of bending energy.
    """

    # Nodes in 3D
    node0 = np.array([xkm1, ykm1, 0.0])
    node1 = np.array([xk, yk, 0])
    node2 = np.array([xkp1, ykp1, 0])

    # Unit vectors along z-axis
    m2e = np.array([0, 0, 1])
    m2f = np.array([0, 0, 1])

    kappaBar = curvature0

    # Initialize gradient of curvature
    gradKappa = np.zeros(6)

    # Edge vectors
    ee = node1 - node0
    ef = node2 - node1

    # Norms of edge vectors
    norm_e = np.linalg.norm(ee)
    norm_f = np.linalg.norm(ef)

    # Unit tangents
    te = ee / norm_e
    tf = ef / norm_f

    # Curvature binormal
    kb = 2.0 * np.cross(te, tf) / (1.0 + np.dot(te, tf))

    chi = 1.0 + np.dot(te, tf)
    tilde_t = (te + tf) / chi
    tilde_d2 = (m2e + m2f) / chi

    # Curvature
    kappa1 = kb[2]

    # Gradient of kappa1 with respect to edge vectors
    Dkappa1De = 1.0 / norm_e * (-kappa1 * tilde_t + np.cross(tf, tilde_d2))
    Dkappa1Df = 1.0 / norm_f * (-kappa1 * tilde_t - np.cross(te, tilde_d2))

    # Populate the gradient of kappa
    gradKappa[0:2] = -Dkappa1De[0:2]
    gradKappa[2:4] = Dkappa1De[0:2] - Dkappa1Df[0:2]
    gradKappa[4:6] = Dkappa1Df[0:2]

    # Gradient of bending energy
    dkappa = kappa1 - kappaBar
    dF = gradKappa * EI * dkappa / l_k

    return dF

def hessEb(xkm1, ykm1, xk, yk, xkp1, ykp1, curvature0, l_k, EI):
    """
    Returns the Hessian (second derivative) of bending energy E_k^b
    with respect to x_{k-1}, y_{k-1}, x_k, y_k, x_{k+1}, and y_{k+1}.

    Parameters:
    xkm1, ykm1 : float
        Coordinates of the previous node (x_{k-1}, y_{k-1}).
    xk, yk : float
        Coordinates of the current node (x_k, y_k).
    xkp1, ykp1 : float
        Coordinates of the next node (x_{k+1}, y_{k+1}).
    curvature0 : float
        Discrete natural curvature at node (xk, yk).
    l_k : float
        Voronoi length of node (xk, yk).
    EI : float
        Bending stiffness.

    Returns:
    dJ : np.ndarray
        Hessian of bending energy.
    """

    # Nodes in 3D
    node0 = np.array([xkm1, ykm1, 0])
    node1 = np.array([xk, yk, 0])
    node2 = np.array([xkp1, ykp1, 0])

    # Unit vectors along z-axis
    m2e = np.array([0, 0, 1])
    m2f = np.array([0, 0, 1])

    kappaBar = curvature0

    # Initialize gradient of curvature
    gradKappa = np.zeros(6)

    # Edge vectors
    ee = node1 - node0
    ef = node2 - node1

    # Norms of edge vectors
    norm_e = np.linalg.norm(ee)
    norm_f = np.linalg.norm(ef)

    # Unit tangents
    te = ee / norm_e
    tf = ef / norm_f

    # Curvature binormal
    kb = 2.0 * np.cross(te, tf) / (1.0 + np.dot(te, tf))

    chi = 1.0 + np.dot(te, tf)
    tilde_t = (te + tf) / chi
    tilde_d2 = (m2e + m2f) / chi

    # Curvature
    kappa1 = kb[2]

    # Gradient of kappa1 with respect to edge vectors
    Dkappa1De = 1.0 / norm_e * (-kappa1 * tilde_t + np.cross(tf, tilde_d2))
    Dkappa1Df = 1.0 / norm_f * (-kappa1 * tilde_t - np.cross(te, tilde_d2))

    # Populate the gradient of kappa
    gradKappa[0:2] = -Dkappa1De[0:2]
    gradKappa[2:4] = Dkappa1De[0:2] - Dkappa1Df[0:2]
    gradKappa[4:6] = Dkappa1Df[0:2]

    # Compute the Hessian (second derivative of kappa)
    DDkappa1 = np.zeros((6, 6))

    norm2_e = norm_e**2
    norm2_f = norm_f**2

    Id3 = np.eye(3)

    # Helper matrices for second derivatives
    tt_o_tt = np.outer(tilde_t, tilde_t)
    tmp = np.cross(tf, tilde_d2)
    tf_c_d2t_o_tt = np.outer(tmp, tilde_t)
    kb_o_d2e = np.outer(kb, m2e)

    D2kappa1De2 = (2 * kappa1 * tt_o_tt - tf_c_d2t_o_tt - tf_c_d2t_o_tt.T) / norm2_e - \
                  kappa1 / (chi * norm2_e) * (Id3 - np.outer(te, te)) + \
                  (kb_o_d2e + kb_o_d2e.T) / (4 * norm2_e)

    tmp = np.cross(te, tilde_d2)
    te_c_d2t_o_tt = np.outer(tmp, tilde_t)
    tt_o_te_c_d2t = te_c_d2t_o_tt.T
    kb_o_d2f = np.outer(kb, m2f)

    D2kappa1Df2 = (2 * kappa1 * tt_o_tt + te_c_d2t_o_tt + te_c_d2t_o_tt.T) / norm2_f - \
                  kappa1 / (chi * norm2_f) * (Id3 - np.outer(tf, tf)) + \
                  (kb_o_d2f + kb_o_d2f.T) / (4 * norm2_f)
    D2kappa1DeDf = -kappa1 / (chi * norm_e * norm_f) * (Id3 + np.outer(te, tf)) \
                  + 1.0 / (norm_e * norm_f) * (2 * kappa1 * tt_o_tt - tf_c_d2t_o_tt + \
                  tt_o_te_c_d2t - crossMat(tilde_d2))
    D2kappa1DfDe = D2kappa1DeDf.T

    # Populate the Hessian of kappa
    DDkappa1[0:2, 0:2] = D2kappa1De2[0:2, 0:2]
    DDkappa1[0:2, 2:4] = -D2kappa1De2[0:2, 0:2] + D2kappa1DeDf[0:2, 0:2]
    DDkappa1[0:2, 4:6] = -D2kappa1DeDf[0:2, 0:2]
    DDkappa1[2:4, 0:2] = -D2kappa1De2[0:2, 0:2] + D2kappa1DfDe[0:2, 0:2]
    DDkappa1[2:4, 2:4] = D2kappa1De2[0:2, 0:2] - D2kappa1DeDf[0:2, 0:2] - \
                         D2kappa1DfDe[0:2, 0:2] + D2kappa1Df2[0:2, 0:2]
    DDkappa1[2:4, 4:6] = D2kappa1DeDf[0:2, 0:2] - D2kappa1Df2[0:2, 0:2]
    DDkappa1[4:6, 0:2] = -D2kappa1DfDe[0:2, 0:2]
    DDkappa1[4:6, 2:4] = D2kappa1DfDe[0:2, 0:2] - D2kappa1Df2[0:2, 0:2]
    DDkappa1[4:6, 4:6] = D2kappa1Df2[0:2, 0:2]

    # Hessian of bending energy
    dkappa = kappa1 - kappaBar
    dJ = 1.0 / l_k * EI * np.outer(gradKappa, gradKappa)
    dJ += 1.0 / l_k * dkappa * EI * DDkappa1

    return dJ

def gradEs(xk, yk, xkp1, ykp1, l_k, EA):
    """
    Calculate the gradient of the stretching energy with respect to the coordinates.

    Args:
    - xk (float): x coordinate of the current point
    - yk (float): y coordinate of the current point
    - xkp1 (float): x coordinate of the next point
    - ykp1 (float): y coordinate of the next point
    - l_k (float): reference length
    - EA (float): elastic modulus

    Returns:
    - F (np.array): Gradient array
    """
    F = np.zeros(4)
    F[0] = -(1.0 - np.sqrt((xkp1 - xk)**2.0 + (ykp1 - yk)**2.0) / l_k) * ((xkp1 - xk)**2.0 + (ykp1 - yk)**2.0)**(-0.5) / l_k * (-2.0 * xkp1 + 2.0 * xk)
    F[1] = -(0.1e1 - np.sqrt((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k) * ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1) / l_k * (-0.2e1 * ykp1 + 0.2e1 * yk)
    F[2] = -(0.1e1 - np.sqrt((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k) * ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1) / l_k * (0.2e1 * xkp1 - 0.2e1 * xk)
    F[3] = -(0.1e1 - np.sqrt((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k) * ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1) / l_k * (0.2e1 * ykp1 - 0.2e1 * yk)

    F = 0.5 * EA * l_k * F  # Scale by EA and l_k

    return F

def hessEs(xk, yk, xkp1, ykp1, l_k, EA):
    """
    This function returns the 4x4 Hessian of the stretching energy E_k^s with
    respect to x_k, y_k, x_{k+1}, and y_{k+1}.
    """
    J = np.zeros((4, 4))  # Initialize the Hessian matrix
    J11 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (-2 * xkp1 + 2 * xk) ** 2) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * ((-2 * xkp1 + 2 * xk) ** 2) / 0.2e1 - 0.2e1 * (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1)) / l_k
    J12 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (-2 * ykp1 + 2 * yk) * (-2 * xkp1 + 2 * xk)) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (-2 * xkp1 + 2 * xk) * (-2 * ykp1 + 2 * yk) / 0.2e1
    J13 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * xkp1 - 2 * xk) * (-2 * xkp1 + 2 * xk)) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (-2 * xkp1 + 2 * xk) * (2 * xkp1 - 2 * xk) / 0.2e1 + 0.2e1 * (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1)) / l_k
    J14 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * ykp1 - 2 * yk) * (-2 * xkp1 + 2 * xk)) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (-2 * xkp1 + 2 * xk) * (2 * ykp1 - 2 * yk) / 0.2e1
    J22 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (-2 * ykp1 + 2 * yk) ** 2) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * ((-2 * ykp1 + 2 * yk) ** 2) / 0.2e1 - 0.2e1 * (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1)) / l_k
    J23 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * xkp1 - 2 * xk) * (-2 * ykp1 + 2 * yk)) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (-2 * ykp1 + 2 * yk) * (2 * xkp1 - 2 * xk) / 0.2e1
    J24 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * ykp1 - 2 * yk) * (-2 * ykp1 + 2 * yk)) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (-2 * ykp1 + 2 * yk) * (2 * ykp1 - 2 * yk) / 0.2e1 + 0.2e1 * (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1)) / l_k
    J33 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * xkp1 - 2 * xk) ** 2) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * ((2 * xkp1 - 2 * xk) ** 2) / 0.2e1 - 0.2e1 * (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1)) / l_k
    J34 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * ykp1 - 2 * yk) * (2 * xkp1 - 2 * xk)) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (2 * xkp1 - 2 * xk) * (2 * ykp1 - 2 * yk) / 0.2e1
    J44 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * ykp1 - 2 * yk) ** 2) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * ((2 * ykp1 - 2 * yk) ** 2) / 0.2e1 - 0.2e1 * (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1)) / l_k

    J = np.array([[J11, J12, J13, J14],
                   [J12, J22, J23, J24],
                   [J13, J23, J33, J34],
                   [J14, J24, J34, J44]])

    J *= 0.5 * EA * l_k

    return J

# Functions to create elastic force vector and its Hessian
def getFs(q, EA, deltaL):
  # q - DOF vector of size N
  # EA - stretching stiffness
  # deltaL - undeformed reference length (assume to be a scalar for this simple example)
  # Output:
  # Fs - a vector (negative gradient of elastic stretching force)
  # Js - a matrix (negative hessian of elastic stretching force)

  ndof = q.size # Number of DOFs
  N = ndof // 2 # Number of nodes

  Fs = np.zeros(ndof) # stretching force
  Js = np.zeros((ndof, ndof))

  for k in range(0, N-1):
      # May need to modify if network of beams
      # k-th stretching spring (USE A LOOP for the general case
      xkm1 = q[2*k] # x coordinate of the first node
      ykm1 = q[2*k+1] # y coordinate of the first node
      xk = q[2*k+2] # x coordinate of the second node
      yk = q[2*k+3] # y coordinate of the second node
      ind = np.arange(2*k, 2*k+4) # 0, 1, 2, 3 for k = 0
      gradEnergy = gradEs(xkm1, ykm1, xk, yk, deltaL, EA)
      hessEnergy = hessEs(xkm1, ykm1, xk, yk, deltaL, EA)

      Fs[ind] -= gradEnergy # force = - gradient of energy. Fs is the stretching force
      Js[np.ix_(ind, ind)] -= hessEnergy # index vector: 0:4

  return Fs, Js

def getFb(q, EI, deltaL):
  # q - DOF vector of size N
  # EI - bending stiffness
  # deltaL - undeformed Voronoi length (assume to be a scalar for this simple example)
  # Output:
  # Fb - a vector (negative gradient of elastic stretching force)
  # Jb - a matrix (negative hessian of elastic stretching force)

  ndof = q.size # Number of DOFs
  N = ndof // 2 # Number of nodes

  Fb = np.zeros(ndof) # bending force
  Jb = np.zeros((ndof, ndof))

  # First bending spring (USE A LOOP for the general case)
  for k in range(1, N-1):
    xkm1 = q[2*k-2] # x coordinate of the first node
    ykm1 = q[2*k-1] # y coordinate of the first node
    xk = q[2*k] # x coordinate of the second node
    yk = q[2*k+1] # y coordinate of the second node
    xkp1 = q[2*k+2] # x coordinate of the third node
    ykp1 = q[2*k+3] # y coordinate of the third node
    ind = np.arange(2*k-2, 2*k+4)
    gradEnergy = gradEb(xkm1, ykm1, xk, yk, xkp1, ykp1, 0, deltaL, EI)
    hessEnergy = hessEb(xkm1, ykm1, xk, yk, xkp1, ykp1, 0, deltaL, EI)

    Fb[ind] -= gradEnergy # force = - gradient of energy. Fb is the stretching force
    Jb[np.ix_(ind, ind)] -= hessEnergy # index vector: 0:6

  return Fb, Jb

def objfun(q_old, u_old, dt, tol, maximum_iter,
           m, mMat, # inertia
           EI, EA, # elastic stiffness
           W, C, # external force
           deltaL,
           free_index):

  """
  Objective Function or Integrator
  Given the old position and the old velocity, find out th new position and the new velocity
  """

  q_new = q_old.copy() # Guess solution

  # Newton Raphson
  iter_count = 0 # number of iterations
  error = tol * 10 # error
  flag = 1 # if flag = 1, it is a good solution

  while error > tol:
    # Inertia
    F_inertia = m/dt * ((q_new - q_old) / dt - u_old)
    J_inertia = mMat / dt ** 2

    # Elastic forces: Stretching and Bending
    Fs, Js = getFs(q_new, EA, deltaL)
    Fb, Jb = getFb(q_new, EI, deltaL)
    F_elastic = Fs + Fb
    J_elastic = Js + Jb

    # External forces
    # Viscous force
    Fv = - C @ ( q_new - q_old ) / dt
    Jv = - C / dt

    # Equations of motion
    f = F_inertia - F_elastic - Fv - W
    J = J_inertia - J_elastic - Jv

    f_free = f[free_index]
    J_free = J[np.ix_(free_index, free_index)]

    # Newton's update (all DOFs are FREE)
    dq_free = np.linalg.solve(J_free, f_free)
    q_new[free_index] = q_new[free_index] - dq_free

    # Get the error
    error = np.linalg.norm(f_free)

    # Update the iteration number
    iter_count += 1
    if iter_count > maximum_iter:
      flag = -1
      print("Maximum number of iterations reached.")
      return q_new, f, flag

    # u_new = (q_new - q_old) / dt # Velocity

  reactionForce = f
  return q_new, reactionForce, flag

def getControlNodes(
  ctime: float,
  control_position: tuple,
  control_angle: float,
  initial_controlled_nodes: np.ndarray,
  totalTime: float,
  deltaL: float) -> np.ndarray:
  """
  Return positions of the two controlled (fixed) nodes at the current time.

  Linearly interpolates position and angle from the initial BC to the target BC
  over totalTime, then places node 1 from node 0 using the inextensible
  beam assumption.

  Angle convention: theta is measured counter-clockwise from the -Y axis.
    theta =  0      -> beam points straight down  (tangent = [ 0, -1])
    theta =  pi/2   -> beam points right           (tangent = [ 1,  0])
    theta = -pi/2   -> beam points left            (tangent = [-1,  0])
    theta =  pi     -> beam points straight up     (tangent = [ 0,  1])
  Tangent vector: (sin(theta), -cos(theta))
  Node 1 = Node 0 + deltaL * (sin(theta), -cos(theta))

  Parameters
  ----------
  ctime : float
      Current simulation time.
  control_position : tuple[float, float]
      Target position of node 0: (x0_target, y0_target).
  control_angle : float
      Target angle at node 0 (radians, from -Y axis, clockwise positive).
  initial_controlled_nodes : np.ndarray
      [x0, y0, x1, y1] of nodes 0 and 1 at t = 0.
  totalTime : float
      Total simulation duration; alpha = 1 at this time.
  deltaL : float
      Edge length (m); used to place node 1 from node 0 + tangent.

  Returns
  -------
  np.ndarray
      [x0, y0, x1, y1] at the current time.
  """
  x0_init, y0_init = initial_controlled_nodes[0], initial_controlled_nodes[1]
  x1_init, y1_init = initial_controlled_nodes[2], initial_controlled_nodes[3]

  # Recover initial angle from the initial node positions
  theta_init = np.arctan2(x1_init - x0_init, -(y1_init - y0_init))

  x0_target, y0_target = control_position

  # Linear interpolation fraction (clamped to avoid overshoot)
  alpha = min(ctime / totalTime, 1.0)

  x0    = x0_init + alpha * (x0_target - x0_init)
  y0    = y0_init + alpha * (y0_target - y0_init)
  theta = theta_init + alpha * (control_angle - theta_init)

  # Node 1 from node 0 + tangent (inextensible beam)
  x1 = x0 + deltaL * np.sin(theta)
  y1 = y0 - deltaL * np.cos(theta)

  return np.array([x0, y0, x1, y1])


def discreteBeam(
  physical_system: dict,
  q0: list = None,
  u0: list = None,
  control_position: tuple = (0.0, 0.0),
  control_angle: float = 0.0,
  t_start: float = 0.0,
  dt: float = 0.01,
  totalTime: float = None,
  plotImage: int = 0,
  plotStep: int = 5) -> dict:

  """
  Discrete elastic beam simulator.

  Simulates a planar discrete elastic beam using stretching and bending
  energies. Physical parameters are read from physical_system; the caller
  supplies q0 and u0 so runs can be chained without re-initialising.

  Parameters
  ----------
  physical_system : PhysicalSystem dict
      Pre-computed physical and derived quantities (from make_physical_system).

  q0 : list[float], optional
      Initial DOF vector, length 2*nv. Defaults to straight-vertical
      configuration. Pass the previous call's q_new to chain runs.

  u0 : list[float], optional
      Initial velocity vector, length 2*nv. Defaults to zero.
      Pass the previous call's u_new to chain runs.

  control_position : tuple[float, float]
      Target position of node 0: (x0_target, y0_target).

  control_angle : float
      Target angle at node 0 (radians, counter-clockwise from -Y axis).
      theta = 0 -> straight down;  theta = pi/2 -> pointing right.

  t_start : float
      Cumulative simulation time at the start of this call (seconds).
      Pass the previous call's ctime to keep displayed time continuous.
      BC interpolation uses relative time internally so physics is unaffected.

  dt : float
      Time step for implicit integration (seconds).

  totalTime : float, optional
      Duration of this simulation run (seconds). If None (default), computed
      automatically as max(delta_pos / BOUNDARY_VELOCITY,
      delta_angle / BOUNDARY_ANGLE_VELOCITY), ensuring the BC moves at the
      prescribed quasi-static rate. Override for testing.

  plotImage : int
      If 1, show beam configuration at every plotStep time steps.

  plotStep : int
      Plot every this many time steps (only used when plotImage = 1).

  Returns
  -------
  dict
      status  : "success" or "failed_convergence"
      message : human-readable outcome string
      q_new   : list[float] — final DOF vector, length 2*nv
      u_new   : list[float] — final velocity vector, length 2*nv
      ctime   : float       — cumulative simulation time reached
  """

  # Unpack physical system
  nv         = physical_system.nv
  EI         = physical_system.EI
  EA         = physical_system.EA
  deltaL     = physical_system.deltaL
  rod_length = physical_system.rod_length
  wall_H     = physical_system.wall_H
  friction   = physical_system.friction

  m    = np.array(physical_system.m)
  mMat = np.diag(m)
  W    = np.array(physical_system.W)
  C    = np.diag(np.array(physical_system.C_diag))

  ndof         = 2 * nv
  tol          = EI / rod_length**2 * 1e-3
  maximum_iter = 50

  # Initialise q0/u0 on first call (straight vertical, zero velocity)
  from src.models.physical_system import initial_q0, initial_u0
  if q0 is None:
    q0 = np.array(initial_q0(physical_system), dtype=float)
  else:
    q0 = np.array(q0, dtype=float)

  if u0 is None:
    u0 = np.array(initial_u0(physical_system), dtype=float)
  else:
    u0 = np.array(u0, dtype=float)

  # Compute totalTime from boundary velocities if not explicitly provided
  if totalTime is None:
    bv  = physical_system.boundary_velocity
    bav = physical_system.boundary_angle_velocity

    x0_curr, y0_curr = q0[0], q0[1]
    x1_curr, y1_curr = q0[2], q0[3]
    theta_curr = np.arctan2(x1_curr - x0_curr, -(y1_curr - y0_curr))

    x0_tgt, y0_tgt = control_position
    delta_pos   = np.sqrt((x0_tgt - x0_curr)**2 + (y0_tgt - y0_curr)**2)
    delta_angle = abs(control_angle - theta_curr)

    t_pos   = delta_pos   / bv
    t_angle = delta_angle / bav

    totalTime = max(t_pos, t_angle, dt)   # floor at one time step

  all_DOFs    = np.arange(ndof)
  fixed_index = np.array([0, 1, 2, 3])           # controlled nodes 0 and 1 are always fixed
  free_index  = np.setdiff1d(all_DOFs, fixed_index)

  initial_controlled_nodes = q0[0:4].copy()

  # Wall contact state: one flag per node; nodes 0 and 1 are never set by wall logic
  is_wall_contact = np.zeros(nv, dtype=bool)

  Nsteps = round(totalTime / dt)
  ctime  = t_start          # cumulative time (for display / chaining)
  q_new  = q0.copy()        # safe default if Nsteps < 2

  # Loop over the time steps
  for timeStep in range(1, Nsteps + 1):

    # Relative time within this run (0 → totalTime) drives BC interpolation
    rel_time = ctime - t_start

    # Impose moving boundary condition on nodes 0 and 1
    BC_DOFs = getControlNodes(rel_time, control_position, control_angle,
                              initial_controlled_nodes, totalTime, deltaL)
    q0[0:4] = BC_DOFs

    # ------------------------------------------------------------------
    # Predictor step — solve with current fixed_index
    # ------------------------------------------------------------------
    q_pred, reactionForce, flag = objfun(q0, u0, dt, tol, maximum_iter,
                                         m, mMat, EI, EA, W, C,
                                         deltaL, free_index)
    if flag < 0:
      return {
        "status":  "failed_convergence",
        "message": f"Predictor did not converge at step {timeStep} (t = {ctime:.4f} s).",
        "q_new":   q0.tolist(),
        "u_new":   u0.tolist(),
        "ctime":   ctime,
      }

    # ------------------------------------------------------------------
    # Update wall contact flags (checks 1 and 2 independently on predictor)
    # ------------------------------------------------------------------
    for i in range(2, nv):  # nodes 0 and 1 are never subject to wall logic
      # Check 1: free node penetrates wall -> add contact
      if not is_wall_contact[i] and q_pred[2*i+1] <= wall_H:
        is_wall_contact[i] = True
      # Check 2: already-contact node has wrong-sign reaction -> release.
      # Must be elif: reactionForce at a newly-contacted node is ~0 (it was
      # free in the predictor), which would incorrectly trigger release.
      elif is_wall_contact[i] and reactionForce[2*i+1] <= 0:
        is_wall_contact[i] = False

    # ------------------------------------------------------------------
    # Rebuild fixed_index / free_index and enforce wall constraints on q0
    # ------------------------------------------------------------------
    fixed_index = [0, 1, 2, 3]  # nodes 0 and 1 always fixed
    for i in range(2, nv):
      if is_wall_contact[i]:
        q0[2*i+1] = wall_H          # y always pinned at wall surface
        fixed_index.append(2*i+1)
        if friction:
          q0[2*i] = q_pred[2*i]     # x frozen only with friction
          fixed_index.append(2*i)
    fixed_index = np.array(fixed_index)
    free_index  = np.setdiff1d(all_DOFs, fixed_index)

    # ------------------------------------------------------------------
    # Corrector step — solve with updated fixed_index
    # ------------------------------------------------------------------
    q_new, reactionForce, flag = objfun(q0, u0, dt, tol, maximum_iter,
                                        m, mMat, EI, EA, W, C,
                                        deltaL, free_index)
    if flag < 0:
      return {
        "status":  "failed_convergence",
        "message": f"Corrector did not converge at step {timeStep} (t = {ctime:.4f} s).",
        "q_new":   q0.tolist(),
        "u_new":   u0.tolist(),
        "ctime":   ctime,
      }

    u_new = (q_new - q0) / dt
    ctime += dt

    q0 = q_new.copy()
    u0 = u_new.copy()

    # Plot
    if plotImage and timeStep % plotStep == 0:
      x_arr = q_new[::2] # q[0], q[2], q[4]
      y_arr = q_new[1::2] # q[1], q[3], q[5]

      h1 = plt.figure(1)
      plt.clf() # Clear current figure
      plt.plot(x_arr, y_arr, 'ko-')
      plt.axhline(y=physical_system.wall_H, color='r', linestyle='--', linewidth=1)
      plt.title(f't={ctime:.4f}s')
      plt.xlabel('x')
      plt.ylabel('y')
      plt.axis('equal')
      plt.show(block=False)
      plt.pause(0.01)

  return {
    "status":  "success",
    "message": f"Simulation completed {Nsteps - 1} steps to t = {ctime:.4f} s.",
    "q_new":   q_new.tolist(),
    "u_new":   u0.tolist(),
    "ctime":   ctime,
  }

#
# Run it from the project root: python -m src.tools.Discrete_Elastic_Beams
#
if __name__ == "__main__":
  import src.core.config as config
  from src.models.physical_system import make_physical_system

  ps     = make_physical_system(config)
  result = discreteBeam(
    physical_system  = ps,
    control_position = (0.0, -0.07),
    control_angle    = 0.0,
    plotImage        = config.PLOT_IMAGE,
  )

  print(f"Status:  {result['status']}")
  print(f"Message: {result['message']}")
  if result["status"] == "success":
    q = result["q_new"]
    print(f"Tip:  x = {q[-2]:.6f} m,  y = {q[-1]:.6f} m")