import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from autograd import grad, hessian

from logging import ERROR

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

# Gradient and Hessian of Elastic Energy - Stretching Energy in Conventional (Linear Elastic) Formulation

def get_energy_stretch_linear_elastic(node0, node1, node2 = None, l_eff = None, EA = None):
  strain_stretch = get_strain_stretch2D(node0, node1, node2, l_eff, l_eff)
  E_s = 0.5 * EA * strain_stretch**2.0 * l_eff
  return E_s

# TODO: In the code below, I assume that l_1 and l_2 are both l_eff.
# This only works if the mesh is uniform. The code should be updated so that it takes two lengths as inputs. The first length is mandatory. The second one is optional.

def grad_and_hess_energy_stretch_linear_elastic(node0, node1, node2 = None, l_eff = None, EA = None):

  strain_stretch = get_strain_stretch2D(node0, node1, node2, l_eff, l_eff)
  G_strain, H_strain = grad_and_hess_strain_stretch2D(node0, node1, node2, l_eff, l_eff)

  gradE_strain = EA * strain_stretch * l_eff
  hessE_strain = EA * l_eff

  G = gradE_strain * G_strain
  H = gradE_strain * H_strain + hessE_strain * np.outer(G_strain, G_strain)

  return G, H

# Gradient and Hessian of Elastic Energy - Bending Energy in Conventional (Linear Elastic) Formulation
def get_energy_bending_linear_elastic(node0, node1, node2 = None, l_eff = None, EI = None):
  if node2 is None:
    return 0
  strain_curvature = get_strain_curvature2D(node0, node1, node2)
  E_b = 0.5 * EI / l_eff * strain_curvature**2.0
  return E_b

def grad_and_hess_energy_bending_linear_elastic(node0, node1, node2 = None, l_eff = None, EI = None):

  if node2 is None:
    return np.zeros(4), np.zeros((4,4))

  strain_curvature = get_strain_curvature2D(node0, node1, node2)
  G_strain, H_strain = grad_and_hess_strain_curvature2D(node0, node1, node2)

  gradE_strain = EI * strain_curvature / l_eff
  hessE_strain = EI / l_eff

  G = gradE_strain * G_strain
  H = gradE_strain * H_strain + hessE_strain * np.outer(G_strain, G_strain)

  return G, H

# Define the custom SquareActivation function
class SquareActivation(nn.Module):
    def forward(self, x):
        return x**2

class ElasticEnergyGeneric(nn.Module):
    def __init__(self):
        super(ElasticEnergyGeneric, self).__init__()
        self.layers = nn.Sequential(
            SquareActivation(),
            nn.Linear(2, 4, dtype=torch.float64),  # Initialize with float64
            nn.ReLU(),
            nn.Linear(4, 8, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(8, 4, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(4, 1, dtype=torch.float64),
        )

    def forward(self, x):
        return self.layers(x)

    def compute_energy_grad_hess(self, x):
        """
        Compute energy, gradient, and Hessian with respect to the input.

        Parameters:
            x (numpy.ndarray): Input vector (2D), shape (2,).

        Returns:
            tuple: (energy, gradient, Hessian), where:
                - energy (float): Energy at the input.
                - gradient (numpy.ndarray): Gradient vector, shape (2,).
                - Hessian (numpy.ndarray): Hessian matrix, shape (2, 2).
        """

        # Ensure the input is a 2D tensor of shape (1, 2)
        if isinstance(x, torch.Tensor):
            x_tensor = x.clone().detach().reshape(1, -1).requires_grad_(True)  # Safely copy the tensor
        else:
            x_tensor = torch.tensor(x.reshape(1, -1), requires_grad=True, dtype=torch.float64)

        # Forward pass to compute energy (output is scalar)
        energy = self.forward(x_tensor).squeeze()  # Remove batch dimension, if any

        # Compute gradient (dE/dx)
        grad = torch.autograd.grad(energy, x_tensor, create_graph=True)[0].squeeze()  # Shape: (2,)

        # Compute Hessian (d²E/dx²)
        hess = torch.zeros((2, 2), dtype=torch.float64)  # Initialize Hessian
        for i in range(2):  # Iterate over each dimension
            grad_i = torch.autograd.grad(grad[i], x_tensor, retain_graph=True)[0].squeeze()  # Shape: (2,)
            hess[i, :] = grad_i  # Fill row i of the Hessian

        return energy.item(), grad.detach().numpy(), hess.detach().numpy()

def train_elastic_energy_generic_model(model, x_train, y_train, E_train, epochs=10000, lr=0.01, verbose=True):
    """
    Train the ElasticEnergyGeneric model to approximate the energy function.

    Parameters:
        model (nn.Module): The neural network model to train.
        x_train (numpy.ndarray): Training inputs (x-coordinates), shape (n_samples, 1).
        y_train (numpy.ndarray): Training inputs (y-coordinates), shape (n_samples, 1).
        E_train (numpy.ndarray): Training outputs (energies), shape (n_samples, 1).
        epochs (int): Number of training epochs.
        lr (float): Learning rate for the optimizer.
        verbose (bool): Whether to print training progress.

    Returns:
        model (nn.Module): Trained model.
        losses (list): List of loss values for each epoch.
    """
    # Combine x_train and y_train into a single input
    inputs_train = np.hstack((x_train, y_train))  # Shape: (n_samples, 2)

    # Convert training data to PyTorch tensors
    inputs_tensor = torch.tensor(inputs_train, dtype=torch.float64)  # Shape: (n_samples, 2)
    E_train_tensor = torch.tensor(E_train, dtype=torch.float64)      # Shape: (n_samples, 1)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # List to store loss values for plotting
    losses = []

    # Training loop
    for epoch in range(epochs):
        # Forward pass
        predictions = model(inputs_tensor)
        loss = criterion(predictions, E_train_tensor)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save loss for monitoring
        losses.append(loss.item())

        # Print progress
        if verbose and (epoch + 1) % (epochs // 20) == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}')

    return model, losses

# Gradient and Hessian of Elastic Energy - Stretching Energy in NN-based (Linear Elastic) Formulation
def get_energy_stretch_nn(node0, node1, node2=None, l_eff=None, EA=None, nn_model=None):
    """
    Compute the energy using the NN-based solution for strain stretch.

    Parameters:
        node0 (array-like): Coordinates of node 0.
        node1 (array-like): Coordinates of node 1.
        node2 (array-like, optional): Coordinates of node 2.
        l_eff (float): Effective length.
        EA (float): Axial stiffness (Young's modulus * Area).
        nn_model (ElasticEnergy): Trained neural network for energy computation.

    Returns:
        float: Energy computed using the NN-based solution.
    """
    # Compute strain stretch using the existing function
    strain_stretch = get_strain_stretch2D(node0, node1, node2, l_eff, l_eff)

    # Use the NN model to compute energy instead of strain_stretch**2.0
    strain_tensor = torch.tensor([[strain_stretch]], dtype=torch.float64)
    nn_energy_output = nn_model.forward(strain_tensor).item()
    # nn_energy_output, grad, hess = nn_model.compute_energy_grad_hess(strain_tensor)

    # Scale by EA and l_eff as in the original function
    E_s = 0.5 * EA * nn_energy_output * l_eff
    return E_s

def grad_and_hess_energy_stretch_nn(node0, node1, node2 = None, l_eff = None, EA = None, nn_model=None):

    strain_stretch = get_strain_stretch2D(node0, node1, node2, l_eff, l_eff) # Geometry, no ML
    G_strain, H_strain = grad_and_hess_strain_stretch2D(node0, node1, node2, l_eff, l_eff) # Geometry, no ML

    # Use NN module
    strain_tensor = torch.tensor([[strain_stretch]], dtype=torch.float64)
    _, gradE_strain, hessE_strain = nn_model.compute_energy_grad_hess(strain_tensor)

    # Scale
    gradE_strain = (0.5 * EA * l_eff) * gradE_strain
    hessE_strain = (0.5 * EA * l_eff) * hessE_strain

    G = gradE_strain * G_strain
    H = gradE_strain * H_strain + hessE_strain * np.outer(G_strain, G_strain)

    return G, H

# Gradient and Hessian of Elastic Energy - Bending Energy in NN-based (Linear Elastic) Formulation
def get_energy_bending_nn(node0, node1, node2 = None, l_eff = None, EA = None, h = None, nn_model=None):

  if node2 is None:
    return 0

  # Get curvature given three nodes
  strain_curvature = get_strain_curvature2D(node0, node1, node2)

  # Scale strain before passing onto the nn_model
  strain_curvature = strain_curvature / l_eff * h
  strain_tensor = torch.tensor([[strain_curvature]], dtype=torch.float64)

  nn_energy_output = nn_model.forward(strain_tensor).item()

  E_b = (0.5 * 0.25 * EA * l_eff) * nn_energy_output
  # Note that EI / EA = 1/4 * h^2.
  # Eb = 0.5 * EI / l_eff * kappa^2 = (0.5 * 0.25 * EA * l_eff) * (kappa*h/l_eff)^2
  return E_b

def grad_and_hess_energy_bending_nn(node0, node1, node2 = None, l_eff = None, EA = None, h = None, nn_model=None):

  if node2 is None:
    return 0, 0

  strain_curvature = get_strain_curvature2D(node0, node1, node2)
  G_strain, H_strain = grad_and_hess_strain_curvature2D(node0, node1, node2)

  # Use NN module
  # Scale strain before passing onto the nn_model
  strain_curvature = strain_curvature / l_eff * h
  strain_tensor = torch.tensor([[strain_curvature]], dtype=torch.float64)
  _, gradE_strain, hessE_strain = nn_model.compute_energy_grad_hess(strain_tensor)
  # Scale back
  gradE_strain = gradE_strain * h / l_eff
  hessE_strain = hessE_strain * h**2 / l_eff**2

  G = gradE_strain * G_strain
  H = gradE_strain * H_strain + hessE_strain * np.outer(G_strain, G_strain)

  G = (0.5 * 0.25 * EA * l_eff) * G
  H = (0.5 * 0.25 * EA * l_eff) * H

  return G, H

def get_energy_elastic_nn_generic(node0, node1, node2=None, l_eff=None, EA=None, h = None, nn_model_generic=None):
    """
    Compute the energy using the NN-based solution for elastic energy.

    Parameters:
        node0 (array-like): Coordinates of node 0.
        node1 (array-like): Coordinates of node 1.
        node2 (array-like, optional): Coordinates of node 2.
        l_eff (float): Effective length.
        EA (float): Axial stiffness (Young's modulus * Area).
        nn_model (ElasticEnergy): Trained neural network for energy computation.

    Returns:
        float: Energy computed using the NN-based solution.
    """
    # Compute stretch and curvature using the existing function
    strain_stretch = get_strain_stretch2D(node0, node1, node2, l_eff, l_eff)
    strain_curvature = get_strain_curvature2D(node0, node1, node2)
    strain_curvature = strain_curvature / l_eff * h # Scale strain before passing onto the nn_model

    # Use the NN model to compute energy
    strain_tensor = torch.tensor([[strain_stretch, strain_curvature]], dtype=torch.float64)
    nn_energy_output = nn_model_generic(strain_tensor).item()

    # Scale by EA and l_eff as in the original function
    E_elastic = (0.5 * EA * l_eff) * nn_energy_output

    return E_elastic

def grad_and_hess_energy_elastic_nn_generic(node0, node1, node2 = None, l_eff = None, EA = None, h = None, nn_model_generic=None):

    if node2 is None:
      # Compute stretch and curvature using the existing function
      strain_stretch = get_strain_stretch2D(node0, node1, node2, l_eff, l_eff)
      strain_curvature = 0.0

      # Compute gradient and hessian of strains
      G_strain_stretch, H_strain_stretch = grad_and_hess_strain_stretch2D(node0, node1, node2, l_eff, l_eff)

      # Use NN module
      strain_tensor = torch.tensor([[strain_stretch, strain_curvature]], dtype=torch.float64)
      _, gradE_strain, hessE_strain = nn_model_generic.compute_energy_grad_hess(strain_tensor)

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
    strain_curvature = strain_curvature / l_eff * h # Scale strain before passing onto the nn_model

    # Compute gradient and hessian of strains
    G_strain_stretch, H_strain_stretch = grad_and_hess_strain_stretch2D(node0, node1, node2, l_eff, l_eff)
    G_strain_curvature, H_strain_curvature = grad_and_hess_strain_curvature2D(node0, node1, node2)
    # Scale back
    G_strain_curvature = G_strain_curvature * h / l_eff
    H_strain_curvature = H_strain_curvature * h**2 / l_eff**2

    # Use NN module
    strain_tensor = torch.tensor([[strain_stretch, strain_curvature]], dtype=torch.float64)
    _, gradE_strain, hessE_strain = nn_model_generic.compute_energy_grad_hess(strain_tensor)

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

