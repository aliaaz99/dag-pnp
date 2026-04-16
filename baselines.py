"""
Baseline methods and variants for DAG-PnP experiments.

Provides:
  - scale_free_dag      : Barabasi-Albert scale-free weighted DAG generator
  - notears_baseline    : gradient descent with matrix-exponential acyclicity penalty
  - dagma_baseline      : DAG-PnP with IdentityDenoiser (= DAGMA, no learned prior)
  - graph_gen_baseline  : placeholder for a graph-generative model baseline
  - dag_pnp_fixed_step  : DAG-PnP with a fixed step size (Experiment 2 ablation)
  - dag_pnp_fixed_sigma : DAG-PnP with a fixed noise level (Experiment 3 ablation)
"""

import numpy as np
import torch
from scipy.linalg import expm
from typing import Callable, Dict, Optional, Tuple

from dag_pnp import (
    dag_pnp, IdentityDenoiser,
    _noise_schedule, _project_onto_K,
)
from losses_dag import grad_lext, data_loss, dagma_penalty
from dag_config import THRESHOLD


# ===========================================================================
# NOTEARS baseline
# ===========================================================================

def _notears_acyclicity(W: np.ndarray) -> float:
    """h(W) = tr(exp(W o W)) - d  using the matrix exponential."""
    # Clip W before expm to prevent overflow; large entries make exp blow up.
    WW = np.clip(W * W, None, 50.0)
    E  = expm(WW)
    val = float(np.trace(E) - W.shape[0])
    return val if np.isfinite(val) else float("inf")


def _notears_acyclicity_grad(W: np.ndarray) -> np.ndarray:
    """Gradient of NOTEARS acyclicity w.r.t. W: 2 * exp(W o W)^T o W."""
    WW   = np.clip(W * W, None, 50.0)
    E    = expm(WW)
    grad = 2.0 * E.T * W
    # Replace any NaN/inf with 0 so the optimiser can recover
    grad = np.where(np.isfinite(grad), grad, 0.0)
    return grad


def notears_baseline(
    X:           torch.Tensor,
    lambda_data: float = 1.0,
    lambda_dag:  float = 0.02,   # reduced from 0.1; consistent with DAGMA experiments
    num_iters:   int   = 1000,
    lr:          float = 0.001,
    W_max:       float = 10.0,   # increased from 3.0; prevents over-clipping at d>=20
    grad_clip:   float = 10.0,
    W_init:      Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Simplified NOTEARS via gradient descent with numerical safeguards.

    Minimises  lambda_data * L_data(W) + lambda_dag * h(W)
    where  h(W) = tr(exp(W o W)) - d  (matrix-exponential acyclicity penalty).

    Safeguards added vs. the naive version:
      - W is projected onto the Frobenius ball of radius W_max each step to
        prevent the matrix exponential from overflowing.
      - The combined gradient is clipped to grad_clip in Frobenius norm before
        the update, preventing single large steps from causing divergence.
      - NaN/inf entries in the acyclicity gradient are zeroed out.
    """
    X_np = X.numpy() if isinstance(X, torch.Tensor) else np.array(X)
    n, d = X_np.shape
    W    = np.zeros((d, d)) if W_init is None else W_init.copy()
    np.fill_diagonal(W, 0.0)
    history: Dict = {"loss": [], "h": []}

    for _ in range(num_iters):
        residual  = X_np - X_np @ W
        grad_data = -1.0 / n * (X_np.T @ residual)
        grad_h    = _notears_acyclicity_grad(W)
        grad      = lambda_data * grad_data + lambda_dag * grad_h

        # Gradient clipping (Frobenius norm)
        g_norm = np.linalg.norm(grad, "fro")
        if g_norm > grad_clip:
            grad = grad * (grad_clip / g_norm)

        W = W - lr * grad
        np.fill_diagonal(W, 0.0)

        # Project onto Frobenius norm ball to keep expm well-behaved
        w_norm = np.linalg.norm(W, "fro")
        if w_norm > W_max:
            W = W * (W_max / w_norm)

        history["loss"].append(float(0.5 / n * np.sum(residual ** 2)))
        history["h"].append(_notears_acyclicity(W))

    return W, history


# ===========================================================================
# DAGMA baseline  (= DAG-PnP with identity denoiser)
# ===========================================================================

def dagma_baseline(
    X: torch.Tensor,
    **kwargs,
) -> Tuple[np.ndarray, Dict]:
    """
    DAGMA: DAG-PnP driven by gradient + DAGMA penalty only, no learned prior.
    Equivalent to passing IdentityDenoiser to dag_pnp.
    All keyword arguments are forwarded to dag_pnp.
    """
    W_est, history = dag_pnp(X, IdentityDenoiser(), inject_noise=False, verbose=True, **kwargs)
    return W_est.detach().numpy(), history


# ===========================================================================
# Graph-generative baseline: greedy BIC score-based DAG learning
# ===========================================================================
#
# This implements a greedy forward search that maximises the Bayesian
# Information Criterion (BIC) over DAG structures.  It represents the
# score-based / generative paradigm for causal discovery — learning structure
# by scoring graphs rather than by continuous gradient optimisation on the
# SEM objective.
#
# The BIC score for variable j given parents Pa(j) in a linear Gaussian SEM:
#   score(j | Pa(j)) = -(n/2) * log(RSS_j / n) - |Pa(j)| * log(n) / 2
#
# The greedy search adds edges one at a time, always picking the edge i→j
# that gives the largest BIC improvement, subject to acyclicity.  Edge weights
# are recovered via OLS regression of each variable on its final parent set.
#
# If you have a trained graph-generative model (e.g. GFlowNet, DiGress used
# for structure learning), replace this function body with a call to it:
#   from my_gen_model import GraphGenModel
#   model = GraphGenModel.load("checkpoint.pt")
#   W_est = model.sample(d=X.shape[1])
#   return W_est, {}
# ===========================================================================

def _bic_local_score(
    X_np: np.ndarray, j: int, parents: list, penalty: float = 1.0
) -> float:
    """
    BIC local score for variable j given its parent set.

    score(j | Pa) = -(n/2) * log(RSS / n) - penalty * |Pa| * log(n) / 2

    A higher score is better (less negative = better fit penalised for size).
    penalty=1.0 gives the standard BIC; smaller values yield denser graphs.
    """
    n = X_np.shape[0]
    if len(parents) == 0:
        rss = float(np.sum(X_np[:, j] ** 2))
    else:
        Pa  = X_np[:, parents]
        # OLS coefficients via least squares
        beta, _, _, _ = np.linalg.lstsq(Pa, X_np[:, j], rcond=None)
        resid = X_np[:, j] - Pa @ beta
        rss   = float(np.sum(resid ** 2))

    rss = max(rss, 1e-12)   # numerical floor
    return -(n / 2.0) * np.log(rss / n) - penalty * len(parents) * np.log(n) / 2.0


def _is_acyclic(adj: np.ndarray) -> bool:
    """Return True if the binary adjacency matrix adj induces a DAG."""
    d = adj.shape[0]
    in_deg = adj.sum(axis=0).copy()              # column = child
    queue  = [i for i in range(d) if in_deg[i] == 0]
    n_visited = 0
    while queue:
        u = queue.pop()
        n_visited += 1
        for v in range(d):
            if adj[u, v]:
                in_deg[v] -= 1
                if in_deg[v] == 0:
                    queue.append(v)
    return n_visited == d


def graph_gen_baseline(
    X:          torch.Tensor,
    max_parents: int   = 5,
    penalty:     float = 1.0,
) -> Tuple[np.ndarray, Dict]:
    """
    Greedy BIC score-based DAG learner (graph-generative baseline).

    Performs a greedy forward edge-addition search:
      1. Start with an empty graph (no edges).
      2. At each step, evaluate every candidate directed edge i→j not yet
         present.  Compute the BIC gain from adding it (improvement in the
         local score of j).  Skip edges that would create a cycle.
      3. Add the edge with the highest positive BIC gain.
      4. Repeat until no edge improves the BIC score.
      5. Recover edge weights via OLS regression of each variable on its
         final parent set.

    Args:
        X:           (n, d) data tensor.
        max_parents: Maximum number of parents per node (caps search width;
                     set to None for unconstrained).
        penalty:     Multiplier on the log(n)/2 complexity term (1.0 = BIC,
                     smaller values give denser graphs).
    Returns:
        W_est:   (d, d) weighted adjacency matrix.
        history: dict with "bic_gains" (list of gains added per step).
    """
    X_np = X.numpy() if isinstance(X, torch.Tensor) else np.array(X)
    n, d = X_np.shape

    adj     = np.zeros((d, d), dtype=bool)   # binary adjacency (i→j)
    parents = [[] for _ in range(d)]         # parent lists

    # Pre-compute baseline scores (no parents)
    scores = [_bic_local_score(X_np, j, [], penalty) for j in range(d)]

    history: Dict = {"bic_gains": []}

    while True:
        best_gain = 0.0
        best_i, best_j, best_new_score = -1, -1, 0.0

        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                if adj[i, j]:
                    continue                        # edge already exists
                if max_parents and len(parents[j]) >= max_parents:
                    continue                        # parent cap reached

                # Tentatively add edge i→j and check acyclicity
                adj[i, j] = True
                if not _is_acyclic(adj):
                    adj[i, j] = False
                    continue

                new_score = _bic_local_score(X_np, j, parents[j] + [i], penalty)
                gain      = new_score - scores[j]

                if gain > best_gain:
                    best_gain      = gain
                    best_i, best_j = i, j
                    best_new_score = new_score

                adj[i, j] = False   # undo tentative addition

        if best_i == -1:
            break   # no improving edge found — converged

        # Commit the best edge
        adj[best_i, best_j] = True
        parents[best_j].append(best_i)
        scores[best_j] = best_new_score
        history["bic_gains"].append(float(best_gain))

    # Recover edge weights via OLS on final parent sets
    W_est = np.zeros((d, d))
    for j in range(d):
        if len(parents[j]) == 0:
            continue
        Pa   = X_np[:, parents[j]]
        beta, _, _, _ = np.linalg.lstsq(Pa, X_np[:, j], rcond=None)
        for k, i in enumerate(parents[j]):
            W_est[i, j] = beta[k]

    return W_est, history


# ===========================================================================
# DAG-PnP with fixed step size  (Experiment 2 ablation)
# ===========================================================================

def dag_pnp_fixed_step(
    X:           torch.Tensor,
    denoiser:    Callable,
    fixed_eta:   float = 0.005,
    num_iters:   int   = 1000,
    lambda_data: float = 1.0,
    lambda_dag:  float = 0.02,
    s:           float = 1.0,
    alpha:       float = 1.0,
    sigma_min:   float = 0.01,
    sigma_max:   float = 0.5,
    smooth_beta: float = 0.9,
    W_max:       float = 10.0,
    delta_min:   float = 1e-6,
    # eta/c/epsilon are accepted so that **shared_kwargs(d) can be splatted
    # directly into this function; they are ignored (fixed_eta is used instead).
    eta:         float = 0.005,
    c:           float = 0.3,
    epsilon:     float = 1e-6,
    project:     bool  = True,
    W_init: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    DAG-PnP with a constant step size (no barrier-aware adaptation).

    Identical to Algorithm 1 except gamma_k = fixed_eta at every iteration.
    Also records whether each gradient substep left the feasible set W_s
    (before projection), so Experiment 2 can count feasibility violations.

    Note: eta, c, epsilon are accepted but ignored so that the full
    shared_kwargs(d) dict can be passed as **kwargs without a TypeError.
    """
    d      = X.shape[1]
    device = X.device
    dtype  = X.dtype

    W = (torch.zeros(d, d, device=device, dtype=dtype)
         if W_init is None else W_init.clone())
    W = _project_onto_K(W, s, delta_min, W_max)

    history: Dict = {
        "loss": [], "h": [], "sigma": [], "gamma": [],
        # True at iteration k if the gradient step z = W - gamma*grad left W_s
        "infeasible_steps": [],
    }
    sigma_prev: Optional[float] = None

    for _ in range(num_iters):
        grad  = grad_lext(X, W, lambda_data, lambda_dag, s)
        gamma = fixed_eta                                   # fixed — no barrier rule
        z     = W - gamma * grad

        # Check feasibility of gradient substep (before projection)
        I  = torch.eye(d, device=device, dtype=dtype)
        M  = s * I - z * z
        sign, _ = torch.linalg.slogdet(M)
        left_feasible = (sign.item() <= 0)                 # True = violated W_s

        sigma_k = _noise_schedule(
            X, W, alpha, sigma_min, sigma_max, sigma_prev, smooth_beta
        )
        sigma_prev = sigma_k

        W_tilde = denoiser(z, sigma_k)
        W = _project_onto_K(W_tilde, s, delta_min, W_max) if project else W_tilde.clone()

        history["loss"].append(data_loss(X, W).item())
        history["h"].append(dagma_penalty(W, s).item())
        history["sigma"].append(sigma_k)
        history["gamma"].append(gamma)
        history["infeasible_steps"].append(left_feasible)

    return W, history


# ===========================================================================
# DAG-PnP with fixed noise level  (Experiment 3 ablation)
# ===========================================================================

def dag_pnp_fixed_sigma(
    X:           torch.Tensor,
    denoiser:    Callable,
    fixed_sigma: float = 0.1,
    num_iters:   int   = 1000,
    lambda_data: float = 1.0,
    lambda_dag:  float = 0.02,
    eta:         float = 0.005,
    c:           float = 0.3,
    epsilon:     float = 1e-6,
    s:           float = 1.0,
    W_max:       float = 10.0,
    delta_min:   float = 1e-6,
    # alpha/sigma_min/sigma_max/smooth_beta are accepted so that
    # **shared_kwargs(d) can be splatted directly; they are ignored because
    # this function uses a fixed sigma instead of the residual schedule.
    alpha:       float = 1.0,
    sigma_min:   float = 0.01,
    sigma_max:   float = 0.5,
    smooth_beta: float = 0.9,
    W_init: Optional[torch.Tensor] = None,
    verbose:   bool = True,
    log_every: int  = 20,
) -> Tuple[torch.Tensor, Dict]:
    """
    DAG-PnP with a constant denoiser noise level (no residual-based schedule).

    Identical to Algorithm 1 except sigma_k = fixed_sigma at every iteration,
    bypassing eq. (12).  Used in Experiment 3 to ablate the noise schedule.

    Note: alpha, sigma_min, sigma_max, smooth_beta are accepted but ignored so
    that the full shared_kwargs(d) dict can be passed as **kwargs without a
    TypeError.

    Args:
        fixed_sigma: constant noise level passed to the denoiser every iteration.
                     Typical choices: sigma_min (weak prior), sigma_max (strong
                     prior), or an intermediate value.
    """
    from dag_pnp import _barrier_step_size   # import here to avoid circular deps
    d      = X.shape[1]
    device = X.device
    dtype  = X.dtype

    W = (torch.zeros(d, d, device=device, dtype=dtype)
         if W_init is None else W_init.clone())
    W = _project_onto_K(W, s, delta_min, W_max)

    history: Dict = {"loss": [], "h": [], "sigma": [], "gamma": []}

    for k in range(num_iters):
        grad  = grad_lext(X, W, lambda_data, lambda_dag, s)
        gamma = _barrier_step_size(W, grad, s, eta, c, epsilon)
        z     = W - gamma * grad

        # Fixed noise level — skip eq. (12)
        # _alpha_k = 1 - fixed_sigma
        # _eps      = torch.randn_like(z)
        # z_noisy   = _alpha_k * z + (fixed_sigma) * _eps
        # W_tilde = denoiser(z_noisy, fixed_sigma)
        # W       = _project_onto_K(W_tilde, s, delta_min, W_max)


        z_binary  = (z.abs() > THRESHOLD).float() * z.sign()
        # z_noisy   = _alpha_k * z_binary + sigma_k * _eps
        W_struct  = denoiser(z_binary, fixed_sigma)   
        # Use denoiser output only as structural mask, keep gradient's weights
        W_tilde   = W_struct.sign() * z.abs()   # denoiser decides edges, gradient decides magnitudes

        history["loss"].append(data_loss(X, W).item())
        history["h"].append(dagma_penalty(W, s).item())
        history["sigma"].append(fixed_sigma)
        history["gamma"].append(gamma)
        if verbose and (k % log_every == 0 or k == num_iters - 1):
            print(f"Iter {k+1}/{num_iters}  loss={history['loss'][-1]:.4f}  "
                  f"h={history['h'][-1]:.4f}  sigma={fixed_sigma:.4f}  "
                  f"gamma={gamma:.6f}")
            

    return W, history
