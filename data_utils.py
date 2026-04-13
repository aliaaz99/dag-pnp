"""
Synthetic data generation and evaluation utilities for DAG-PnP experiments.

Generates random weighted DAGs and samples from the corresponding linear SEM:
    X = X W + E   =>   X = E (I - W)^{-1}

Weight range default
--------------------
The default weight_high is 1.0 (changed from 2.0) to match dag_config.WEIGHT_HIGH.
With weights in [0.5, 1.0] the DAGMA feasible set (parameterised via dagma_s(d))
safely contains the true W for all graph sizes used in the experiments.
If you need larger weights, pass weight_high explicitly and increase s accordingly.
"""

import numpy as np
import torch
from typing import Tuple


# ---------------------------------------------------------------------------
# Random DAG generation
# ---------------------------------------------------------------------------

def random_dag(
    d: int,
    sparsity: float = 0.3,
    weight_low: float = 0.5,
    weight_high: float = 1.0,   
    seed: int = None,
) -> np.ndarray:
    """
    Generate a random weighted DAG with d nodes.

    Uses a random topological ordering: edges are only added from node i to
    node j when i appears before j in the ordering, guaranteeing acyclicity.

    Args:
        d:           Number of nodes.
        sparsity:    Probability of each admissible edge being included.
        weight_low:  Minimum absolute edge weight.
        weight_high: Maximum absolute edge weight.
        seed:        Optional random seed.
    Returns:
        W: (d, d) numpy array, the weighted adjacency matrix.
           W[i, j] != 0 means a causal edge from variable i to variable j.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(d)          # random topological order

    W = np.zeros((d, d))
    for rank_i in range(d):
        for rank_j in range(rank_i + 1, d):
            if rng.random() < sparsity:
                i, j = order[rank_i], order[rank_j]
                w = rng.uniform(weight_low, weight_high)
                w *= rng.choice([-1, 1])
                W[i, j] = w
    return W


# ===========================================================================
# Scale-free DAG generator (Barabasi-Albert preferential attachment)
# ===========================================================================

def scale_free_dag(
    d:            int,
    m:            int   = 2,
    weight_low:   float = 0.5,
    weight_high:  float = 1.0,  
    seed:         int   = None,
) -> np.ndarray:
    """
    Generate a random weighted scale-free DAG via Barabasi-Albert preferential
    attachment with a random topological ordering.

    Each node (in topological order) selects m parents from the preceding nodes
    with probability proportional to their in-degree + 1 (uniform component
    prevents isolation of the first nodes).

    Args:
        d:           Number of nodes.
        m:           Number of parents each new node attaches to.
        weight_low:  Minimum absolute edge weight.
        weight_high: Maximum absolute edge weight.
        seed:        Optional random seed.
    Returns:
        W: (d, d) weighted adjacency matrix (DAG guaranteed by construction).
    """
    rng   = np.random.default_rng(seed)
    order = rng.permutation(d)       # random topological ordering
    W     = np.zeros((d, d))
    indegree = np.zeros(d)           # in-degree in the topological index space

    for rank_j in range(1, d):
        n_parents = min(m, rank_j)
        # Preferential attachment: prob proportional to in-degree + 1
        weights = indegree[:rank_j] + 1.0
        probs   = weights / weights.sum()
        parent_ranks = rng.choice(rank_j, size=n_parents, replace=False, p=probs)

        for pr in parent_ranks:
            i = order[pr]
            j = order[rank_j]
            w = rng.uniform(weight_low, weight_high) * rng.choice([-1, 1])
            W[i, j] = w
            indegree[pr] += 1

    return W

# ---------------------------------------------------------------------------
# Linear SEM sampling
# ---------------------------------------------------------------------------

def sample_sem(
    W: np.ndarray,
    n: int,
    noise_std: float = 1.0,
    seed: int = None,
) -> np.ndarray:
    """
    Sample n observations from the linear SEM  X = X W + E.

    Solving for X:  X (I - W) = E  =>  X = E (I - W)^{-1}.

    Args:
        W:         (d, d) weighted adjacency matrix (must be a DAG).
        n:         Number of samples.
        noise_std: Standard deviation of isotropic Gaussian noise E.
        seed:      Optional random seed.
    Returns:
        X: (n, d) data matrix.
    """
    rng = np.random.default_rng(seed)
    d = W.shape[0]
    E = rng.standard_normal((n, d)) * noise_std
    I_minus_W = np.eye(d) - W
    X = E @ np.linalg.inv(I_minus_W)
    return X


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def threshold_W(W: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """Binarise a weighted adjacency matrix by absolute value threshold."""
    return (np.abs(W) > threshold).astype(int)


def evaluate_dag(
    W_pred: np.ndarray,
    W_true: np.ndarray,
    threshold: float = 0.3,
) -> dict:
    """
    Evaluate DAG recovery against a ground-truth adjacency matrix.

    Metrics
    -------
    shd  : Structural Hamming Distance — fp + fn (each wrong/missing edge = 1).
    tpr  : True Positive Rate  (recall).
    fdr  : False Discovery Rate.
    precision : Precision.
    f1   : F1 score.
    """
    B_pred = threshold_W(W_pred, threshold)
    B_true = threshold_W(W_true, threshold)

    tp = int(np.sum((B_pred == 1) & (B_true == 1)))
    fp = int(np.sum((B_pred == 1) & (B_true == 0)))
    fn = int(np.sum((B_pred == 0) & (B_true == 1)))

    shd = fp + fn   # each wrong or missing directed edge counts once

    tpr       = tp / (tp + fn + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    fdr       = fp / (tp + fp + 1e-9)
    f1        = 2 * precision * tpr / (precision + tpr + 1e-9)

    return {
        "shd":       shd,
        "tpr":       float(tpr),
        "fdr":       float(fdr),
        "precision": float(precision),
        "f1":        float(f1),
        "tp":        tp,
        "fp":        fp,
        "fn":        fn,
    }


# ---------------------------------------------------------------------------
# Convenience: torch tensors from numpy
# ---------------------------------------------------------------------------

def to_tensor(
    arr: np.ndarray,
    device: str = "cpu",
    dtype=torch.float32,
) -> torch.Tensor:
    return torch.tensor(arr, device=device, dtype=dtype)
