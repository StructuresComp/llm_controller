# ---------------------------------------------------------------------------
# src/benchmarks/losses.py
# ---------------------------------------------------------------------------
# Shared loss primitives for benchmark cases.
#
# Current shapes (C, U, S, L) all use the theta-profile fitting loss from
# shape_checker.py. Add composite or custom losses here as new cases require.
# ---------------------------------------------------------------------------

from src.tools.shape_checker import compute_shape_error


def profile_loss(nodes: list, shape_key: str) -> dict:
    """
    Profile-fitting loss for beam shapes (C, U, S, L).

    Returns the full compute_shape_error dict:
      error_percent, confidence_percent, rms_normalized,
      fitted_params, ideal_nodes, diagnostics.
    """
    return compute_shape_error(nodes, shape_key)
