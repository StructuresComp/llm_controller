import numpy as np
from typing import Any, Dict, Optional

from langchain.tools import tool

def summarize_sensitivity(
    S: np.ndarray,
    zero_tol: float = 1e-12) -> Dict[str, Any]:
    """
    Summarize an equilibrium sensitivity matrix with scalar diagnostics that are
    useful for model evaluation, experiment design, and instability detection.

    This function is designed as a lightweight tool for multi-agent workflows.
    It takes only the sensitivity matrix

        S = dq_f / dq_b

    and returns a collection of physically meaningful summaries, rather than
    forcing downstream agents to work with the full matrix directly.

    Parameters
    ----------
    S : np.ndarray
        Sensitivity matrix of shape (n_free, n_boundary), where:
        - n_free = number of free DOFs
        - n_boundary = number of boundary DOFs

        Each column measures how the free equilibrium coordinates respond to
        perturbations of one boundary DOF.

    zero_tol : float, optional
        Numerical tolerance used to determine whether singular values should be
        treated as zero. This is used when computing rank and condition number.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing the following fields:

        - "shape":
            Tuple (n_free, n_boundary)

        - "fro_norm":
            Frobenius norm of S, measuring overall response magnitude

        - "spectral_norm":
            Largest singular value of S, measuring maximum amplification

        - "min_singular_value":
            Smallest singular value of S

        - "max_singular_value":
            Largest singular value of S

        - "condition_number":
            sigma_max / sigma_min if sigma_min > zero_tol, otherwise np.inf

        - "rank":
            Numerical rank of S based on zero_tol

        - "column_norms":
            Euclidean norm of each column, indicating which boundary DOFs have
            the strongest influence on the system

        - "row_norms":
            Euclidean norm of each row, indicating which free DOFs are most
            sensitive to boundary perturbations

        - "mean_abs":
            Mean absolute value of entries of S

        - "max_abs":
            Maximum absolute entry of S

        - "stable_rank":
            Stable rank = ||S||_F^2 / ||S||_2^2 when spectral norm > zero_tol,
            otherwise 0.0

        - "is_zero_matrix":
            True if all entries of S are numerically zero

        - "dominant_boundary_dof":
            Index of the column with the largest column norm, or None if S is empty

        - "dominant_free_dof":
            Index of the row with the largest row norm, or None if S is empty

    Notes
    -----
    Typical uses:
    - high fro_norm      -> highly responsive regime
    - high spectral_norm -> strong amplification along at least one direction
    - high condition_number -> near-degeneracy / instability / anisotropic response
    - large column norm  -> corresponding boundary DOF is especially influential
    - large row norm     -> corresponding free DOF is especially sensitive

    This function does not assume any specific beam, rod, or simulator model.
    It only summarizes the matrix passed to it.
    """
    S = np.asarray(S, dtype=float)

    if S.ndim != 2:
        raise ValueError(f"S must be a 2D array, got shape {S.shape}")

    n_free, n_boundary = S.shape

    is_zero_matrix = np.all(np.abs(S) <= zero_tol)

    # Safe defaults for empty matrices
    if n_free == 0 or n_boundary == 0:
        return {
            "shape": (n_free, n_boundary),
            "fro_norm": 0.0,
            "spectral_norm": 0.0,
            "min_singular_value": 0.0,
            "max_singular_value": 0.0,
            "condition_number": np.inf,
            "rank": 0,
            "column_norms": np.zeros(n_boundary),
            "row_norms": np.zeros(n_free),
            "mean_abs": 0.0,
            "max_abs": 0.0,
            "stable_rank": 0.0,
            "is_zero_matrix": True,
            "dominant_boundary_dof": None,
            "dominant_free_dof": None,
        }

    fro_norm = float(np.linalg.norm(S, ord="fro"))
    column_norms = np.linalg.norm(S, axis=0)
    row_norms = np.linalg.norm(S, axis=1)
    mean_abs = float(np.mean(np.abs(S)))
    max_abs = float(np.max(np.abs(S)))

    singular_values = np.linalg.svd(S, compute_uv=False)
    sigma_max = float(singular_values[0]) if singular_values.size > 0 else 0.0
    sigma_min = float(singular_values[-1]) if singular_values.size > 0 else 0.0

    rank = int(np.sum(singular_values > zero_tol))

    if sigma_min > zero_tol:
        condition_number = float(sigma_max / sigma_min)
    else:
        condition_number = float(np.inf)

    if sigma_max > zero_tol:
        stable_rank = float((fro_norm ** 2) / (sigma_max ** 2))
    else:
        stable_rank = 0.0

    dominant_boundary_dof: Optional[int]
    dominant_free_dof: Optional[int]

    dominant_boundary_dof = int(np.argmax(column_norms)) if column_norms.size > 0 else None
    dominant_free_dof = int(np.argmax(row_norms)) if row_norms.size > 0 else None

    return {
        "shape": (n_free, n_boundary),
        "fro_norm": fro_norm,
        "spectral_norm": sigma_max,
        "min_singular_value": sigma_min,
        "max_singular_value": sigma_max,
        "condition_number": condition_number,
        "rank": rank,
        "column_norms": column_norms,
        "row_norms": row_norms,
        "mean_abs": mean_abs,
        "max_abs": max_abs,
        "stable_rank": stable_rank,
        "is_zero_matrix": is_zero_matrix,
        "dominant_boundary_dof": dominant_boundary_dof,
        "dominant_free_dof": dominant_free_dof,
    }

# Register the discreteBeam function as a LangChain tool for integration with agent workflows.
sensitivity_tool = tool(summarize_sensitivity)
