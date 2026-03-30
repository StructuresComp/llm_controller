import sys
import pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]  # src/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.Energy_Learning_Tool import assemble_global_grad_hess, ElasticEnergyQuadratic

def test_FDM(n_nodes=11, eps=1e-7):
    # random configuration
    q = np.random.randn(2 * n_nodes)

    # simple constants
    l_eff = 1.0
    EA = 1.0
    h = 0.1

    # define your energy model
    energy_model = ElasticEnergyQuadratic(a=1.0, b=0.25, c=0.0)

    # analytic gradient and Hessian
    G, J = assemble_global_grad_hess(q, l_eff, EA, h, energy_model)

    ndof = q.size
    J_fd = np.zeros_like(J)

    # FD Hessian from gradient
    for i in range(ndof):
        dq = np.zeros(ndof)
        dq[i] = eps

        Gp, _ = assemble_global_grad_hess(q + dq, l_eff, EA, h, energy_model)
        Gm, _ = assemble_global_grad_hess(q - dq, l_eff, EA, h, energy_model)

        J_fd[:, i] = (Gp - Gm) / (2.0 * eps)

    err = J_fd - J

    print("||J||         =", np.linalg.norm(J))
    print("||J_fd||      =", np.linalg.norm(J_fd))
    print("||J_fd - J||  =", np.linalg.norm(err))
    print("max abs error =", np.max(np.abs(err)))

    return q, J, J_fd, err

if __name__ == "__main__":
    test_FDM()