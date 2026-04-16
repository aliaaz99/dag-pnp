"""
DAG-PnP: Plug-and-Play Causal Discovery with Diffusion Priors
Algorithm 1 from the paper.

Usage
-----
    from dag_pnp import dag_pnp
    from my_denoiser import MyDenoiser          # your external denoiser

    denoiser = MyDenoiser(...)                  # must implement __call__(W, sigma)
    W_est, history = dag_pnp(X, denoiser, ...)
"""

import numpy as np
import torch
from typing import Callable, Dict, Optional, Tuple

from losses import grad_lext, data_loss, dagma_penalty


# ===========================================================================
#  PLACEHOLDER DENOISER
#  Replace with your trained diffusion-model denoiser.
# ===========================================================================

class IdentityDenoiser:
    """
    Placeholder denoiser — returns the input unchanged.

    A real denoiser D_sigma : R^{d x d} -> R^{d x d} should be trained on
    samples of weighted DAG adjacency matrices (see e.g. DiGress [Vignac et
    al., ICLR 2023]) and map a noisy adjacency matrix toward the learned
    distribution of DAGs at noise level sigma.

    Interface contract
    ------------------
    - Input  W     : torch.Tensor of shape (d, d)  — noisy adjacency matrix
    - Input  sigma : float                          — noise level
    - Output       : torch.Tensor of shape (d, d)  — denoised adjacency matrix
    - The denoiser should NOT modify W in-place.
    - Diagonal entries of the output do not need to be zero;
      the projection step in Algorithm 1 enforces that.
    """

    def __call__(self, W: torch.Tensor, sigma: float) -> torch.Tensor:
        # TODO: replace this body with your diffusion-model denoiser
        return W.clone()


class GaussianShrinkageDenoiser:
    """
    Gaussian shrinkage (Wiener filter) denoiser.

    Assumes a zero-mean Gaussian prior on edge weights with standard deviation
    `prior_std`, and additive Gaussian noise with standard deviation `sigma`.
    The closed-form MMSE denoiser is a linear shrinkage toward zero:

        D_sigma(W) = (prior_std^2 / (prior_std^2 + sigma^2)) * W

    Properties
    ----------
    - When sigma -> 0  the factor -> 1  (identity; no change near convergence).
    - When sigma is large the factor -> 0  (strong shrinkage toward zero early on).
    - Encourages sparse, small-magnitude weights — a reasonable structural prior
      until a trained diffusion denoiser is available.
    - Satisfies the interface contract: does not modify W in-place, output shape
      matches input shape.

    Args
    ----
    prior_std : Prior standard deviation of edge weights (> 0).
                Set to the expected magnitude of true edge weights, i.e.
                (WEIGHT_LOW + WEIGHT_HIGH) / 2 from dag_config.
    """

    def __init__(self, prior_std: float = 0.75):
        if prior_std <= 0:
            raise ValueError(f"prior_std must be > 0, got {prior_std}")
        self.prior_std = prior_std

    def __call__(self, W: torch.Tensor, sigma: float) -> torch.Tensor:
        shrinkage = self.prior_std ** 2 / (self.prior_std ** 2 + sigma ** 2)
        return shrinkage * W.clone()
    

class GDSSDiffusionDenoiser:
    """
    Diffusion-model denoiser backed by a trained GDSS score network.

    Implements Tweedie's formula (posterior mean) for the VP-SDE:

        adj_hat0 = (adj_t + std(t)² · score(adj_t, t)) / mean_coeff(t)

    where:
      - adj_t   is the noisy adjacency (the PnP iterate W, treated as a
                sample from the VP-SDE marginal at diffusion time t)
      - t       is found by inverting std(t) = sigma via binary search
      - score   is  ∇_{adj_t} log p_t(adj_t)  from the trained score network
      - mean_coeff(t) = exp(-0.25·t²·(β₁-β₀) - 0.5·t·β₀)  (VP-SDE scaling)

    Node features are fixed to all-ones, consistent with `init: ones` training.
    All nodes are treated as active (flags = 1).

    Args
    ----
    model_adj    : trained ScoreNetworkA / ScoreNetworkA_Transformer
    model_x      : trained ScoreNetworkX / ScoreNetworkX_Transformer
    sde_adj      : VPSDE instance used for adjacency during training
    max_feat_num : F — number of node features (config.data.max_feat_num)
    device       : torch.device or str
    t_min        : lower bound for the t search (avoids t=0 instability)
    """

    def __init__(self, model_adj, model_x, sde_adj, max_feat_num: int,
                 device, t_min: float = 1e-4):
        self.model_adj    = model_adj
        self.model_x      = model_x
        self.sde          = sde_adj
        self.max_feat_num = max_feat_num
        self.device       = torch.device(device) if isinstance(device, str) else device
        self.t_min        = t_min
        # Build score_fn inline — avoids importing gdss/losses.py which would
        # collide with the top-level losses.py already in sys.modules.
        self.score_fn     = self._make_score_fn(sde_adj, model_adj)

    @staticmethod
    def _make_score_fn(sde, model):
        """
        Equivalent to gdss get_score_fn(sde, model, train=False, continuous=True)
        for VP-SDE, inlined to avoid the losses.py module-name collision.
        Returns score = -ε_pred / std  (i.e. ∇ log p_t).
        """
        model.eval()
        _m = model.module if hasattr(model, 'module') else model
        _takes_time = getattr(_m, 'takes_time', False)

        def score_fn(x, adj, flags, t):
            with torch.no_grad():
                out = model(x, adj, flags, t=t) if _takes_time else model(x, adj, flags)
            # marginal_prob(zeros, t)[1] gives std without touching x
            std = sde.marginal_prob(torch.zeros_like(adj), t)[1]
            return -out / std[:, None, None]

        return score_fn

    # ------------------------------------------------------------------
    @classmethod
    def from_checkpoint(cls, ckpt_path: str, device: str = 'cpu', t_min: float = 1e-4):
        """
        Load a GDSSDiffusionDenoiser directly from a saved .pth checkpoint.

        Args
        ----
        ckpt_path : absolute path to a GDSS checkpoint, e.g.
                    'gdss/checkpoints/sem_SF_d20/Apr15-15:58:08.pth'
        device    : 'cpu' or 'cuda:0' etc.
        t_min     : lower bound for the t search
        Returns
        -------
        GDSSDiffusionDenoiser ready for inference
        """
        import sys, os
        _gdss_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gdss')
        if _gdss_dir not in sys.path:
            sys.path.insert(0, _gdss_dir)

        # Import only sde.py and model files — neither depends on losses.py,
        # so there is no collision with the top-level losses.py.
        from sde import VPSDE, VESDE, subVPSDE
        from models.ScoreNetwork_A import ScoreNetworkA
        from models.ScoreNetwork_X import ScoreNetworkX, ScoreNetworkX_GMH
        from models.ScoreNetwork_Transformer import (ScoreNetworkA_Transformer,
                                                     ScoreNetworkX_Transformer)

        _constructors = {
            'ScoreNetworkX':             ScoreNetworkX,
            'ScoreNetworkX_GMH':         ScoreNetworkX_GMH,
            'ScoreNetworkA':             ScoreNetworkA,
            'ScoreNetworkA_Transformer': ScoreNetworkA_Transformer,
            'ScoreNetworkX_Transformer': ScoreNetworkX_Transformer,
        }

        def _load_model(params, state_dict):
            p = params.copy()
            model_type = p.pop('model_type')
            model = _constructors[model_type](**p)
            if state_dict and 'module.' in list(state_dict.keys())[0]:
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            return model.to(device)

        def _load_sde(cfg):
            if cfg.type == 'VP':
                return VPSDE(beta_min=cfg.beta_min, beta_max=cfg.beta_max, N=cfg.num_scales)
            elif cfg.type == 'VE':
                return VESDE(sigma_min=cfg.beta_min, sigma_max=cfg.beta_max, N=cfg.num_scales)
            elif cfg.type == 'subVP':
                from sde import subVPSDE
                return subVPSDE(beta_min=cfg.beta_min, beta_max=cfg.beta_max, N=cfg.num_scales)
            raise NotImplementedError(f'SDE type {cfg.type}')

        ckpt      = torch.load(ckpt_path, map_location=device)
        cfg       = ckpt['model_config']
        model_adj = _load_model(ckpt['params_adj'], ckpt['adj_state_dict'])
        model_x   = _load_model(ckpt['params_x'],   ckpt['x_state_dict'])
        sde_adj   = _load_sde(cfg.sde.adj)

        return cls(model_adj, model_x, sde_adj, cfg.data.max_feat_num, device, t_min)

    # ------------------------------------------------------------------
    def _sigma_to_t(self, sigma: float) -> float:
        """Binary search: find t in [t_min, T] such that std(t) ≈ sigma."""
        dummy = torch.zeros(1, 1, 1, device=self.device)
        lo, hi = self.t_min, float(self.sde.T)
        for _ in range(64):          # 64 bisections → error < 1e-18
            mid = (lo + hi) * 0.5
            t_mid = torch.tensor([mid], device=self.device)
            _, std = self.sde.marginal_prob(dummy, t_mid)
            if std.item() < sigma:
                lo = mid
            else:
                hi = mid
        return (lo + hi) * 0.5

    # ------------------------------------------------------------------
    @torch.no_grad()
    def __call__(self, W: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        Tweedie denoising at noise level sigma.

        Args
        ----
        W     : (d, d) noisy adjacency matrix  (current PnP iterate)
        sigma : noise level used to select diffusion time t via std(t)=sigma
        Returns
        -------
        (d, d) posterior-mean denoised adjacency estimate
        """
        d = W.shape[0]

        # ---- map sigma → t  ------------------------------------------
        t_val = self._sigma_to_t(sigma)
        t     = torch.tensor([t_val], device=self.device, dtype=torch.float32)

        # ---- prepare batched inputs  ----------------------------------
        adj_t = W.unsqueeze(0).to(device=self.device, dtype=torch.float32)  # (1,d,d)
        x     = torch.ones(1, d, self.max_feat_num, device=self.device)
        flags = torch.ones(1, d, device=self.device)

        # ---- VP-SDE marginal at t  ------------------------------------
        #   marginal_prob(x, t) returns (mean_coeff·x, std)
        #   Passing ones gives mean_coeff directly; no need to recompute
        #   log_mean_coeff manually from beta_0/beta_1.
        ones = torch.ones_like(adj_t)
        mean_c, std_t = self.sde.marginal_prob(ones, t)  # mean_c:(1,d,d), std_t:(1,)

        # ---- score  ∇_{adj_t} log p_t(adj_t)  ------------------------
        #   get_score_fn returns:  score = -ε_pred / std
        score = self.score_fn(x, adj_t, flags, t)              # (1, d, d)

        # ---- Tweedie's formula  --------------------------------------
        #   adj_hat0 = ( adj_t + std² · score ) / mean_coeff
        #            = ( adj_t − std · ε_pred ) / mean_coeff
        std_sq   = (std_t ** 2).view(1, 1, 1)
        adj_hat0 = (adj_t + std_sq * score) / mean_c           # (1, d, d)

        return adj_hat0.squeeze(0).cpu()



# ===========================================================================
#  STEP-SIZE RULE  (eq. 11 / Section III-C)
# ===========================================================================

def _barrier_step_size(
    W: torch.Tensor,
    grad: torch.Tensor,
    s: float,
    eta: float,
    c: float,
    epsilon: float,
) -> float:
    """
    Barrier-aware step size (eq. 11):

        gamma = min( eta,  c * delta(W) / ( ||nabla L_ext||_F + epsilon ) )

    where  delta(W) = sigma_min( s I - W o W )  is the distance-to-boundary
    proxy (eq. 10).

    Near a stationary point the gradient is small, delta/epsilon dominates,
    and the min selects the base step eta.  Near the DAGMA boundary the
    gradient is large, so the barrier term keeps the step safe.

    Args:
        W:       (d, d) current iterate
        grad:    (d, d) gradient nabla L_ext(W)
        s:       DAGMA parameter
        eta:     base step size (> 0)
        c:       safety constant (0 < c < 1)
        epsilon: smoothing constant (> 0, prevents /0 at stationary points)
    Returns:
        Scalar step size (float).
    """
    d = W.shape[0]
    M = s * torch.eye(d, device=W.device, dtype=W.dtype) - W * W
    # delta(W) = smallest singular value of M  (eq. 10)
    delta = torch.linalg.svdvals(M).min().item()
    grad_norm = torch.norm(grad, "fro").item()
    barrier = c * delta / (grad_norm + epsilon)
    return float(min(eta, barrier))


# ===========================================================================
#  NOISE SCHEDULE  (eq. 12 / Section III-D)
# ===========================================================================

def _noise_schedule(
    X: torch.Tensor,
    W: torch.Tensor,
    alpha: float,
    sigma_min: float,
    sigma_max: float,
    sigma_prev: Optional[float],
    smooth_beta: float,
) -> float:
    """
    Residual-based noise schedule (eq. 12):

        sigma_hat^2 = alpha * ||X - X W||^2_F / ||X||^2_F

    then clipped to [sigma_min, sigma_max] and optionally smoothed.

    Args:
        X:           (n, d) data matrix
        W:           (d, d) current iterate
        alpha:       scaling hyperparameter (> 0)
        sigma_min:   lower clip bound
        sigma_max:   upper clip bound
        sigma_prev:  previous noise level for exponential smoothing
                     (None to skip smoothing on first call)
        smooth_beta: EMA coefficient in [0, 1); 0 = no smoothing
    Returns:
        Scalar sigma (float).
    """
    residual_sq = torch.sum((X - X @ W) ** 2).item()
    data_sq     = torch.sum(X ** 2).item() + 1e-12
    sigma = float(np.sqrt(alpha * residual_sq / data_sq))
    sigma = float(np.clip(sigma, sigma_min, sigma_max))
    # Exponential smoothing for stability
    if sigma_prev is not None and smooth_beta > 0.0:
        sigma = smooth_beta * sigma_prev + (1.0 - smooth_beta) * sigma
    return sigma


# ===========================================================================
#  PROJECTION ONTO K  (eq. 9 / Section III-B)
# ===========================================================================

def _project_onto_K(
    W: torch.Tensor,
    s: float,
    delta_min: float,
    W_max: float,
) -> torch.Tensor:
    """
    Project W onto the compact feasible set

        K = { W in R^{d x d} : det(s I - W o W) >= delta_min,
                                ||W||_F <= W_max }

    Additionally enforces zero diagonal (no self-loops).

    Implementation:
      1. Zero out diagonal.
      2. Project onto Frobenius norm ball of radius W_max.
      3. If the determinant constraint is violated, iteratively shrink W
         toward 0 (which always satisfies det(sI) = s^d > delta_min).

    Args:
        W:         (d, d) matrix to project
        s:         DAGMA parameter
        delta_min: minimum determinant buffer (> 0)
        W_max:     Frobenius norm bound (> 0)
    Returns:
        W_proj: (d, d) projected matrix in K.
    """
    W = W.clone()
    d = W.shape[0]
    I = torch.eye(d, device=W.device, dtype=W.dtype)

    # (1) Enforce zero diagonal
    W.fill_diagonal_(0.0)

    # (2) Frobenius norm projection
    norm = torch.norm(W, "fro").item()
    if norm > W_max:
        W = W * (W_max / norm)

    # (3) Determinant constraint — shrink toward 0 if violated
    for _ in range(300):
        M = s * I - W * W
        sign, logdet = torch.linalg.slogdet(M)
        det = (sign * torch.exp(logdet)).item()
        if det >= delta_min:
            break
        W = W * 0.9

    return W


# ===========================================================================
#  ALGORITHM 1: DAG-PnP
# ===========================================================================

def dag_pnp(
    X: torch.Tensor,
    denoiser: Callable[[torch.Tensor, float], torch.Tensor],
    num_iters: int = 500,
    # Objective weights
    lambda_data: float = 1.0,
    lambda_dag:  float = 1.0,
    # Barrier-aware step size (eq. 11)
    eta:     float = 0.01,
    c:       float = 0.3,
    epsilon: float = 1e-6,
    # DAGMA domain
    s: float = 1.0,
    # Noise schedule (eq. 12)
    alpha:      float = 1.0,
    sigma_min:  float = 0.01,
    sigma_max:  float = 0.5,
    smooth_beta: float = 0.9,
    # Feasible set K (eq. 9)
    W_max:     float = 10.0,
    delta_min: float = 1e-6,
    # Initialization
    W_init: Optional[torch.Tensor] = None,
    # Logging
    verbose:   bool = True,
    log_every: int  = 20,
) -> Tuple[torch.Tensor, Dict]:
    """
    DAG-PnP — Algorithm 1.

    Alternates between:
      (i)  a gradient step on the SEM data-fidelity loss + DAGMA penalty, and
      (ii) a denoising step from a diffusion model trained on weighted DAGs.

    The barrier-aware step-size rule (eq. 11) prevents the gradient step from
    leaving the DAGMA feasible set W_s.  A projection enforces membership in
    the compact set K throughout.

    Args
    ----
    X            : (n, d) observed data matrix.
    denoiser     : Callable D(W, sigma) -> W_denoised.
                   Must accept a (d, d) torch.Tensor and a float noise level.
                   See IdentityDenoiser above for the expected interface.
    num_iters    : Total iterations K.
    lambda_data  : Weight lambda_data for the data-fidelity term.
    lambda_dag   : Weight lambda_dag  for the DAGMA acyclicity penalty.
    eta          : Base step size eta > 0.
    c            : Safety constant for the barrier step (0 < c < 1).
                   Must satisfy  2*c*W_max + c^2*delta_max < 1  (Lemma 1).
    epsilon      : Smoothing constant epsilon > 0 (prevents /0).
    s            : DAGMA parameter s > 0.
    alpha        : Noise-schedule scaling alpha > 0  (eq. 12).
    sigma_min    : Minimum denoiser noise level.
    sigma_max    : Maximum denoiser noise level.
    smooth_beta  : EMA coefficient for noise-level smoothing (0 = off).
    W_max        : Frobenius norm bound for K.
    delta_min    : Minimum determinant buffer for K (e.g. 1e-6).
    W_init       : Initial W^(0); defaults to the zero matrix.
    verbose      : Print progress.
    log_every    : Print every this many iterations.

    Returns
    -------
    W        : (d, d) estimated weighted adjacency matrix.
    history  : Dict with per-iteration 'loss', 'h', 'sigma', 'gamma'.
    """
    d      = X.shape[1]
    device = X.device
    dtype  = X.dtype

    # ------------------------------------------------------------------
    # Initialise W^(0) in K
    # ------------------------------------------------------------------
    if W_init is None:
        W = torch.zeros(d, d, device=device, dtype=dtype)
    else:
        W = W_init.clone().to(device=device, dtype=dtype)

    W = _project_onto_K(W, s, delta_min, W_max)

    history: Dict = {"loss": [], "h": [], "sigma": [], "gamma": []}
    sigma_prev: Optional[float] = None

    if verbose:
        print(f"DAG-PnP | d={d}  n={X.shape[0]}  iters={num_iters}")
        print(f"  lambda_data={lambda_data}  lambda_dag={lambda_dag}  "
              f"eta={eta}  c={c}  s={s}")
        print(f"  alpha={alpha}  sigma=[{sigma_min},{sigma_max}]  "
              f"W_max={W_max}  delta_min={delta_min}")
        print("-" * 65)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    for k in range(num_iters):

        # ---- Step 1: gradient of L_ext  (eqs. 1, 3, 5) ----
        grad = grad_lext(X, W, lambda_data, lambda_dag, s)

        # ---- Step 2: barrier-aware step size  (eq. 11) ----
        gamma = _barrier_step_size(W, grad, s, eta, c, epsilon)

        # ---- Step 3: gradient step  (eq. 6) ----
        z = W - gamma * grad

        # ---- Step 4: noise schedule  (eq. 12) ----
        sigma_k = _noise_schedule(
            X, W, alpha, sigma_min, sigma_max, sigma_prev, smooth_beta
        )
        sigma_prev = sigma_k

        # ---- Step 5: denoising step  (eq. 7) ----
        # D_{sigma_k}(z) — replace IdentityDenoiser with your model
        W_tilde = denoiser(z, sigma_k)

        # ---- Step 6: projection step  (eq. 8) ----
        W = _project_onto_K(W_tilde, s, delta_min, W_max)

        # ---- Logging ----
        loss_val = data_loss(X, W).item()
        h_val    = dagma_penalty(W, s).item()
        history["loss"].append(loss_val)
        history["h"].append(h_val)
        history["sigma"].append(sigma_k)
        history["gamma"].append(gamma)

        if verbose and (k % log_every == 0 or k == num_iters - 1):
            print(
                f"  iter {k:4d}/{num_iters} | "
                f"loss={loss_val:.5f}  h={h_val:.5f}  "
                f"sigma={sigma_k:.4f}  gamma={gamma:.3e}"
            )

    if verbose:
        print("-" * 65)
        print("Done.")

    return W, history
