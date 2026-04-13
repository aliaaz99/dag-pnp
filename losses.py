"""
Objective functions and gradients for DAG-PnP.

Equations reference the paper:
  DAG-PnP: Plug-and-Play Causal Discovery with Diffusion Priors
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Data fidelity  (eq. 1)
# ---------------------------------------------------------------------------

def data_loss(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """
    L_data(W) = 1/(2n) * ||X - X W||^2_F

    Args:
        X: (n, d) data matrix
        W: (d, d) weighted adjacency matrix
    Returns:
        Scalar loss.
    """
    n = X.shape[0]
    residual = X - X @ W
    return 0.5 / n * torch.sum(residual ** 2)


def grad_data_loss(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """
    Gradient of L_data w.r.t. W:
        nabla_W L_data = -1/n * X^T (X - X W)

    Args:
        X: (n, d) data matrix
        W: (d, d) weighted adjacency matrix
    Returns:
        (d, d) gradient tensor.
    """
    n = X.shape[0]
    residual = X - X @ W
    return -1.0 / n * (X.T @ residual)


# ---------------------------------------------------------------------------
# DAGMA acyclicity penalty  (eqs. 2-3)
# ---------------------------------------------------------------------------

def _M(W: torch.Tensor, s: float) -> torch.Tensor:
    """M(W) = s*I - W o W  (Hadamard / element-wise square of W)."""
    d = W.shape[0]
    I = torch.eye(d, device=W.device, dtype=W.dtype)
    return s * I - W * W


def dagma_penalty(W: torch.Tensor, s: float = 1.0) -> torch.Tensor:
    """
    h(W) = -log det(s I - W o W) + d log s   (eq. 2)

    Defined on W_s = {W : det(s I - W o W) > 0}.
    h(W) = 0  iff  the graph induced by W is a DAG.

    Args:
        W: (d, d) weighted adjacency matrix
        s: DAGMA parameter (s > 0)
    Returns:
        Scalar penalty (inf if outside W_s).
    """
    d = W.shape[0]
    M = _M(W, s)
    sign, logdet = torch.linalg.slogdet(M)
    if sign.item() <= 0:
        return torch.tensor(float("inf"), device=W.device, dtype=W.dtype)
    return -logdet + d * np.log(s)


def grad_dagma(W: torch.Tensor, s: float = 1.0) -> torch.Tensor:
    """
    Gradient of h w.r.t. W:
        nabla h(W) = 2 * (s I - W o W)^{-T} o W   (eq. 3)

    Args:
        W: (d, d) weighted adjacency matrix
        s: DAGMA parameter (s > 0)
    Returns:
        (d, d) gradient tensor.
    """
    M = _M(W, s)
    M_inv_T = torch.linalg.inv(M).T
    return 2.0 * M_inv_T * W


# ---------------------------------------------------------------------------
# Combined explicit objective  (eq. 5)
# ---------------------------------------------------------------------------

def lext(
    X: torch.Tensor,
    W: torch.Tensor,
    lambda_data: float,
    lambda_dag: float,
    s: float = 1.0,
) -> torch.Tensor:
    """
    L_ext(W) = lambda_data * L_data(W) + lambda_dag * h(W)   (eq. 5)
    """
    return lambda_data * data_loss(X, W) + lambda_dag * dagma_penalty(W, s)


def grad_lext(
    X: torch.Tensor,
    W: torch.Tensor,
    lambda_data: float,
    lambda_dag: float,
    s: float = 1.0,
) -> torch.Tensor:
    """
    nabla L_ext(W) = lambda_data * nabla L_data(W) + lambda_dag * nabla h(W)
    """
    return lambda_data * grad_data_loss(X, W) + lambda_dag * grad_dagma(W, s)
