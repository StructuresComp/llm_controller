import numpy as np
import matplotlib.pyplot as plt

from langchain.tools import tool

#Helper function
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

# Gradient and Hessian of Strains (axial stretch) on one edge in 2D"""
def get_strain_stretch_edge2D(node0, node1, l_k):
  '''
    Compute the axial stretch of a edge connecting node0 and node1.
    The code does not change even if we move from 2D to 3D.

    Parameters:
        node0 (array-like): Coordinates of the first node (start of the edge),
                            typically a 2-element array [x0, y0] for the 2D case.
        node1 (array-like): Coordinates of the second node (end of the edge),
                            typically a 2-element array [x1, y1].
        l_k (float): Reference (undeformed) length of the edge.

    Returns:
        epsX (float): Axial stretch of the edge, defined as:
                      epsX = (current_length / reference_length) - 1,
                      where current_length is the Euclidean distance between node0 and node1.

    Description:
        The function calculates the axial stretch of a 2D edge by:
        1. Computing the edge vector as the difference between node1 and node0.
        2. Calculating the current length of the edge as the Euclidean norm of the edge vector.
        3. Using the ratio of the current length to the reference length to compute the stretch.
  '''
  edge = node1 - node0
  edgeLen = np.linalg.norm(edge)
  epsX = edgeLen / l_k - 1
  return epsX

def grad_and_hess_strain_stretch_edge2D(node0, node1, l_k):
  '''
  Compute the gradient and hessian of the axial stretch of a 2D edge with
  respect to the dof vector (4 dofs: x,y coordinates of the two nodes)

  Inputs:
  node0: 2x1 vector - position of the first node
  node1: 2x1 vector - position of the last node
  l_k: reference length (undeformed) of the edge

  Outputs:
  dF: 4x1  vector - gradient of axial stretch between node0 and node 1.
  dJ: 4x4 vector - hessian of axial stretch between node0 and node 1.
  '''

  edge = node1 - node0
  edgeLen = np.linalg.norm(edge)
  tangent = edge / edgeLen
  epsX = get_strain_stretch_edge2D(node0, node1, l_k)

  dF_unit = tangent / l_k # gradient of stretch with respect to the edge vector
  dF = np.zeros((4))
  dF[0:2] = - dF_unit
  dF[2:4] = dF_unit

  # M (see below) is the Hessian of square(stretch) with respect to the edge vector
  Id3 = np.eye(2)
  M = 2.0 / l_k * ((1 / l_k - 1 / edgeLen) * Id3 + 1 / edgeLen * ( np.outer( edge, edge ) ) / edgeLen ** 2)

  # M is the Hessian of stretch with respect to the edge vector
  if epsX == 0: # Edge case
    M2 = np.zeros_like(M)
  else:
    M2 = 1.0/(2.0*epsX) * (M - 2.0*np.outer(dF_unit,dF_unit))

  dJ = np.zeros((4,4))
  dJ[0:2,0:2] = M2
  dJ[2:4,2:4] = M2
  dJ[0:2,2:4] = - M2
  dJ[2:4,0:2] = - M2

  return dF,dJ

# Gradient and Hessian of Strains (axial stretch) on one node in 2D
# Node-based formulation allows us to easily have coupled terms like stretch times curvature. Otherwise, curvature and twist are node-based, whereas, conventionally strain is edge-based. It is straightforward to formulate edge-based stretch.**
def get_strain_stretch2D(node0, node1, node2=None, l_0=1.0, l_1=1.0):
  '''
    Compute the axial stretch of an node (node1) based on the stretches on the two edges
    connected to that node. The first edge connects node0 and node1, whereas
    the second edge connects node1 and node2. The code does not change even if
    we move from 2D to 3D.

    Parameters:
        node0 (array-like): Coordinates of the first node (start of the first edge),
                            typically a 2-element array [x0, y0] for the 2D case.
        node1 (array-like): Coordinates of the second node (end of the first edge
                            and start of the second edge),
                            typically a 2-element array [x1, y1].
        node2 (array-like): Coordinates of the third node (end of the second edge),
                            typically a 2-element array [x2, y2].
        l_0 (float): Reference (undeformed) length of the first edge.
        l_1 (float): Reference (undeformed) length of the second edge.

    Returns:
        epsilon_1 (float): Axial stretch at node1, defined as the average of the
                            stretches on the two edges connected to node1. If
                            only two nodes are specified, the stretch is defined
                            as half of the stretch on the first edge.
  '''
  # Convert inputs to numpy arrays for vectorized operations
  node0 = np.array(node0, dtype=float)
  node1 = np.array(node1, dtype=float)

  if node2 is not None:
    # If all the nodes are specified (i.e., node1 is an internal node),
    # computation is as usual (average of the strains of the two edges).
    node2 = np.array(node2, dtype=float)

    stretch_firstEdge  = get_strain_stretch_edge2D(node0, node1, l_0)
    stretch_secondEdge = get_strain_stretch_edge2D(node1, node2, l_1)

    epsilon_1 = 0.5 * stretch_firstEdge + 0.5 * stretch_secondEdge
  else:
    # Simplified case if only two nodes are specified.
    epsilon_1 = 0.5 * get_strain_stretch_edge2D(node0, node1, l_0)

  return epsilon_1

def grad_and_hess_strain_stretch2D(node0, node1, node2=None, l_0=1.0, l_1=1.0):
  '''
  Compute the gradient and hessian of the axial stretch at node1 with
  respect to the dof vector (6 dofs: x,y coordinates of three nodes).

  Inputs:
  node0: 2x1 vector - position of the first node
  node1: 2x1 vector - position of the middle node
  node2: 2x1 vector - position of the last node
  l_0: reference length (undeformed) of the first edge
  l_1: reference length (undeformed) of the second edge

  Outputs:
  dF: 6x1  vector - gradient of axial stretch between node0 and node 1.
  dJ: 6x6 vector - hessian of axial stretch between node0 and node 1.

  Description:
  The gradient and hessian at node1 is the average of the gradients and hessians
  of the two edges (one between node0 and node1 and the other one between node1
  and node2).
  '''
  # Ensure inputs are numpy arrays
  node0 = np.array(node0, dtype=float)
  node1 = np.array(node1, dtype=float)

  if node2 is not None:
    node2 = np.array(node2, dtype=float)

    G_varepsilon = np.zeros(6)
    H_varepsilon = np.zeros((6, 6))

    G1, H1 = grad_and_hess_strain_stretch_edge2D(node0, node1, l_0)
    G2, H2 = grad_and_hess_strain_stretch_edge2D(node1, node2, l_1)
    G_varepsilon[0:4] = 0.5 * G1
    G_varepsilon[2:6] += 0.5 * G2
    H_varepsilon[0:4,0:4] = 0.5 * H1
    H_varepsilon[2:6,2:6] += 0.5 * H2
  else:
    G1, H1 = grad_and_hess_strain_stretch_edge2D(node0, node1, l_0)
    G_varepsilon = 0.5 * G1
    H_varepsilon = 0.5 * H1

  return G_varepsilon, H_varepsilon

def get_strain_curvature2D(node0, node1, node2):
  '''
  Compute the curvature (scalar) in 2D at node1 that is connected to node0 and
  node2.

  Inputs:
  node0: 2x1 vector - position of the first node
  node1: 2x1 vector - position of the middle node
  node2: 2x1 vector - position of the last node

  Outputs:
  kappa1: scalar - curvature at node1 (a scalar)

  Description:
  The function converts 2D nodes into 3D; computers the curvature binormal
  (a vector) and extracts the scalar curvature (z-component of curvature
  binormal)
  '''

  # Nodes in 3D
  node0 = np.array([node0[0], node0[1], 0.0])
  node1 = np.array([node1[0], node1[1], 0.0])
  node2 = np.array([node2[0], node2[1], 0.0])

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

  # Curvature
  kappa1 = kb[2]

  return kappa1

def grad_and_hess_strain_curvature2D(node0, node1, node2):
  '''
  Compute the gradient and hessian of the curvature at node1 with
  respect to the dof vector (6 dofs: x,y coordinates of three nodes).

  Inputs:
  node0: 2x1 vector - position of the first node
  node1: 2x1 vector - position of the middle node
  node2: 2x1 vector - position of the last node

  Outputs:
  dF: 6x1 vector - gradient of curvature at node 1
  dJ: 6x6 vector - hessian of curvature at node 1.

  Description:
  The gradient and hessian at node1 is the average of the gradients and hessians
  of the two edges (one between node0 and node1 and the other one between node1
  and node2).
  '''

  # Nodes in 3D
  node0 = np.array([node0[0], node0[1], 0.0])
  node1 = np.array([node1[0], node1[1], 0.0])
  node2 = np.array([node2[0], node2[1], 0.0])

  # Unit vectors along z-axis
  m2e = np.array([0, 0, 1])
  m2f = np.array([0, 0, 1])

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
  kappa1 = get_strain_curvature2D(node0, node1, node2)

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


  return gradKappa, DDkappa1

def get_energy_elastic_nn_generic(node0, node1, node2=None, l_eff=None, EA=None, h = None, energy_model_generic=None):
    """
    Compute the energy using the NN-based solution for elastic energy.

    Parameters:
        node0 (array-like): Coordinates of node 0.
        node1 (array-like): Coordinates of node 1.
        node2 (array-like, optional): Coordinates of node 2.
        l_eff (float): Effective length.
        EA (float): Axial stiffness (Young's modulus * Area).
        energy_model_generic (ElasticEnergy): Trained neural network for energy computation.

    Returns:
        float: Energy computed using the NN-based solution.
    """
    # Compute stretch and curvature using the existing function
    strain_stretch = get_strain_stretch2D(node0, node1, node2, l_eff, l_eff)
    strain_curvature = get_strain_curvature2D(node0, node1, node2)
    strain_curvature = strain_curvature / l_eff * h # Scale strain before passing onto the energy model

    # Use the NN model to compute energy
    strain_tensor = np.array([[strain_stretch, strain_curvature]], dtype=np.float64)
    nn_energy_output = energy_model_generic(strain_tensor).item()

    # Scale by EA and l_eff as in the original function
    E_elastic = (0.5 * EA * l_eff) * nn_energy_output

    return E_elastic

def grad_and_hess_energy_elastic_generic(node0, node1, node2 = None, l_eff = None, EA = None, h = None, energy_model_generic=None):

    if node2 is None:
      # Compute stretch and curvature using the existing function
      strain_stretch = get_strain_stretch2D(node0, node1, node2, l_eff, l_eff)
      strain_curvature = 0.0

      # Compute gradient and hessian of strains
      G_strain_stretch, H_strain_stretch = grad_and_hess_strain_stretch2D(node0, node1, node2, l_eff, l_eff)

      # Use NN module
      strain_tensor = np.array([[strain_stretch, strain_curvature]], dtype=np.float64)
      _, gradE_strain, hessE_strain = energy_model_generic.compute_energy_grad_hess(strain_tensor)

      # Scale
      gradE_strain = (0.5 * EA * l_eff) * gradE_strain
      hessE_strain = (0.5 * EA * l_eff) * hessE_strain

      G = gradE_strain[0] * G_strain_stretch
      H = gradE_strain[0] * H_strain_stretch + \
      hessE_strain[0,0] * np.outer(G_strain_stretch, G_strain_stretch)

      return G, H

    # Compute stretch and curvature using the existing function
    strain_stretch = get_strain_stretch2D(node0, node1, node2, l_eff, l_eff)
    strain_curvature = get_strain_curvature2D(node0, node1, node2)
    strain_curvature = strain_curvature / l_eff * h # Scale strain before passing onto the energy model

    # Compute gradient and hessian of strains
    G_strain_stretch, H_strain_stretch = grad_and_hess_strain_stretch2D(node0, node1, node2, l_eff, l_eff)
    G_strain_curvature, H_strain_curvature = grad_and_hess_strain_curvature2D(node0, node1, node2)
    # Scale back
    G_strain_curvature = G_strain_curvature * h / l_eff
    H_strain_curvature = H_strain_curvature * h / l_eff 
    # This scaling has been validated against FDM and I am reasonably confident.
    # Use test_FDM in test_beam_FDM.py to validate the correctness of the gradient and Hessian of the energy model

    # Use NN module
    strain_tensor = np.array([[strain_stretch, strain_curvature]], dtype=np.float64)
    _, gradE_strain, hessE_strain = energy_model_generic.compute_energy_grad_hess(strain_tensor)

    # Scale
    gradE_strain = (0.5 * EA * l_eff) * gradE_strain
    hessE_strain = (0.5 * EA * l_eff) * hessE_strain

    G = gradE_strain[0] * G_strain_stretch + gradE_strain[1] * G_strain_curvature
    H = gradE_strain[0] * H_strain_stretch + \
    hessE_strain[0,0] * np.outer(G_strain_stretch, G_strain_stretch) + \
    hessE_strain[0,1] * np.outer(G_strain_stretch, G_strain_curvature) + \
    gradE_strain[1] * H_strain_curvature + \
    hessE_strain[1,1] * np.outer(G_strain_curvature, G_strain_curvature) + \
    hessE_strain[1,0] * np.outer(G_strain_curvature, G_strain_stretch)

    return G, H

#
# The code above is from a previous example. The code below is for integration with LLM agents
#

# Step 1: Define a simple quadratic energy function for testing purposes
class ElasticEnergyQuadratic:
    def __init__(self, a=1.0, b=1.0, c=0.0):
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)

    def forward(self, x):
        """
        x: array-like of shape (2,) or (1,2)
        returns energy
        """
        x = np.asarray(x, dtype=np.float64).reshape(-1, 2)
        eps = x[:, 0]
        kap = x[:, 1]
        energy = self.a * eps**2 + self.b * kap**2 + self.c * eps * kap
        return energy

    def compute_energy_grad_hess(self, x):
        """
        Returns:
            energy: scalar
            grad: np.ndarray shape (2,)
            hess: np.ndarray shape (2,2)
        """
        x = np.asarray(x, dtype=np.float64).reshape(1, 2)
        eps = x[0, 0]
        kap = x[0, 1]

        energy = self.a * eps**2 + self.b * kap**2 + self.c * eps * kap

        grad = np.array([
            2.0 * self.a * eps + self.c * kap,
            2.0 * self.b * kap + self.c * eps
        ], dtype=np.float64)

        hess = np.array([
            [2.0 * self.a, self.c],
            [self.c, 2.0 * self.b]
        ], dtype=np.float64)

        return energy, grad, hess

# Step 2: Define a function to compute the local gradient and Hessian of the elastic energy for a stencil of three nodes
# Given a global configuration vector q, compute the global gradient and Hessian of the elastic energy
# by assembling the contributions from each local element (stencil of three nodes).
def assemble_global_grad_hess(q, l_eff, EA, h, energy_model):
    """
    q : global configuration vector, shape (2*N,)
    Returns:
        G_global : shape (2*N,)
        H_global : shape (2*N, 2*N)
    """
    q = np.asarray(q, dtype=np.float64)
    ndof = len(q)
    N = ndof // 2

    G_global = np.zeros(ndof, dtype=np.float64)
    H_global = np.zeros((ndof, ndof), dtype=np.float64)

    for k in range(1, N - 1):
        node0 = np.array([q[2*k - 2], q[2*k - 1]], dtype=np.float64)
        node1 = np.array([q[2*k],     q[2*k + 1]], dtype=np.float64)
        node2 = np.array([q[2*k + 2], q[2*k + 3]], dtype=np.float64)

        G_loc, H_loc = grad_and_hess_energy_elastic_generic(
            node0=node0,
            node1=node1,
            node2=node2,
            l_eff=l_eff,
            EA=EA,
            h=h,
            energy_model_generic=energy_model
        )

        # global DOF indices for stencil (k-1, k, k+1)
        idx = [2*k - 2, 2*k - 1, 2*k, 2*k + 1, 2*k + 2, 2*k + 3]

        # assemble gradient
        for a in range(6):
            G_global[idx[a]] += G_loc[a]

        # assemble Hessian
        for a in range(6):
            for b in range(6):
                H_global[idx[a], idx[b]] += H_loc[a, b]

    return G_global, H_global

# Step 3: Define a function to evaluate the candidate solution by computing the residual and sensitivity loss
# TODO: Need to wrap in a class so that LLMs do not have access to the unnecessary arguments and internal details of the energy model.
def evaluate_candidate(q, S_true, free_index, fixed_index, weight,
                       l_eff, EA, h, energy_model,
                       w_res=1.0, w_sens=1.0):

    G, J = assemble_global_grad_hess(q, l_eff, EA, h, energy_model)
    
    # External weight
    ndof = len(q)
    nv = ndof // 2
    f_ext = np.zeros(ndof, dtype=np.float64)  # Assuming zero external
    f_ext[1::2] = - weight / nv  # Apply weight to y-coordinates

    G_f = G[free_index]
    f_f = f_ext[free_index]

    J_ff = J[np.ix_(free_index, free_index)]
    J_fb = J[np.ix_(free_index, fixed_index)]

    residual = G_f - f_f
    loss_res = np.dot(residual, residual)

    S_pred = - np.linalg.solve(J_ff, J_fb)
    diffS = S_pred - S_true
    loss_sens = np.sum( diffS ** 2 )

    loss_total = w_res * loss_res + w_sens * loss_sens

    return {
        "loss_total": loss_total,
        "loss_res": loss_res,
        "loss_sens": loss_sens,
        "residual": residual,
        "S_pred": S_pred,
    }

