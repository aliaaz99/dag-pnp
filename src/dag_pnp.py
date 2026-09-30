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

from losses_dag import grad_lext, data_loss, dagma_penalty
from dag_config import THRESHOLD

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
        import sys, os
        _gdss_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gdss')
        if _gdss_dir not in sys.path:
            sys.path.insert(0, _gdss_dir)
        from gdss.losses import get_score_fn  # gdss/losses.py

        self.model_adj    = model_adj
        self.model_x      = model_x
        self.sde          = sde_adj
        self.max_feat_num = max_feat_num
        self.device       = torch.device(device) if isinstance(device, str) else device
        self.t_min        = t_min
        self.score_fn     = get_score_fn(sde_adj, model_adj, train=False, continuous=True)

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
        from gdss.utils.loader import load_model_from_ckpt, load_sde

        ckpt       = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg        = ckpt['model_config']
        model_adj  = load_model_from_ckpt(ckpt['params_adj'], ckpt['adj_state_dict'], device)
        model_x    = load_model_from_ckpt(ckpt['params_x'],   ckpt['x_state_dict'],   device)
        sde_adj    = load_sde(cfg.sde.adj)

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
        Apply VP-SDE forward noise then Tweedie denoising at noise level sigma.

        Receives a clean W (the gradient-step output z), injects VP-SDE forward
        noise using the SDE's exact marginal_prob, and returns the posterior-mean
        denoised estimate via Tweedie's formula.

        Args
        ----
        W     : (d, d) clean adjacency matrix  (current PnP iterate after grad step)
        sigma : target noise std; used to find diffusion time t via std(t) = sigma
        Returns
        -------
        (d, d) posterior-mean denoised adjacency estimate
        """
        d = W.shape[0]

        # ---- map sigma → t  ------------------------------------------
        t_val = self._sigma_to_t(sigma)
        t     = torch.tensor([t_val], device=self.device, dtype=torch.float32)

        # ---- prepare batched inputs  ----------------------------------
        adj_0 = W.unsqueeze(0).to(device=self.device, dtype=torch.float32)  # (1,d,d)
        x     = torch.ones(1, d, self.max_feat_num, device=self.device)
        flags = torch.ones(1, d, device=self.device)

        # ---- VP-SDE exact forward noise  ------------------------------
        #   adj_t = mean_coeff(t) * adj_0 + std(t) * eps
        mean_x, std_t = self.sde.marginal_prob(adj_0, t)      # mean_x:(1,d,d), std_t:(1,)
        eps   = torch.randn_like(adj_0)
        adj_t = mean_x + std_t.view(-1, 1, 1) * eps           # (1, d, d)

        # ---- mean_coeff for Tweedie denominator  ---------------------
        log_mean_coeff = (
            -0.25 * t ** 2 * (self.sde.beta_1 - self.sde.beta_0)
            - 0.5  * t      *  self.sde.beta_0
        )
        mean_coeff_t = torch.exp(log_mean_coeff)               # (1,)

        # ---- score  ∇_{adj_t} log p_t(adj_t)  ------------------------
        #   get_score_fn returns:  score = -ε_pred / std
        score = self.score_fn(x, adj_t, flags, t)              # (1, d, d)

        # ---- Tweedie's formula  --------------------------------------
        #   adj_hat0 = ( adj_t + std² · score ) / mean_coeff
        #            = ( adj_t − std · ε_pred ) / mean_coeff
        std_sq   = (std_t        ** 2).view(1, 1, 1)
        mean_c   =  mean_coeff_t      .view(1, 1, 1)
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
    # print("c=",c,"delta=",delta,"grad_norm=",grad_norm,"epsilon=",epsilon,"kasr=",delta / (grad_norm + epsilon))
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


def _linear_noise_schedule(
    k: int,
    num_iters: int,
    sigma_min: float,
    sigma_max: float,
) -> float:
    """
    Linear (vanilla) noise schedule: sigma decreases linearly from sigma_max
    to sigma_min over the course of the optimisation, independent of the data.

        sigma_k = sigma_max - (sigma_max - sigma_min) * k / (num_iters - 1)

    At k=0            : sigma = sigma_max  (strong denoising early)
    At k=num_iters-1  : sigma = sigma_min  (light denoising near convergence)

    Args:
        k:          current iteration index (0-based)
        num_iters:  total number of iterations
        sigma_min:  noise level at the final iteration
        sigma_max:  noise level at the first iteration
    Returns:
        Scalar sigma (float).
    """
    if num_iters <= 1:
        return sigma_min
    t = k / (num_iters - 1)          # progress in [0, 1]
    return float(sigma_max - (sigma_max - sigma_min) * t)


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
    # Noise schedule (eq. 12) — used only when denoiser has no .sde
    alpha:      float = 1.0,
    sigma_min:  float = 0.01,
    sigma_max:  float = 0.5,
    smooth_beta: float = 0.9,
    # Feasible set K (eq. 9)
    W_max:     float = 10.0,
    delta_min: float = 1e-6,
    # Initialization
    W_init: Optional[torch.Tensor] = None,
    # Projection
    project: bool = True,
    # Optional L1 sparsity regularizer on W (0.0 = disabled)
    lambda_l1: float = 0.0,
    # Noise injection before denoiser (set False for IdentityDenoiser / DAGMA)
    inject_noise: bool = True,
    # Apply denoiser only every this many iterations; pure gradient step otherwise
    denoise_every: int = 5,
    # Logging
    verbose:   bool = True,
    log_every: int  = 100,
) -> Tuple[torch.Tensor, Dict]:
    """
    DAG-PnP — Algorithm 1.

    Alternates between:
      (i)  a gradient step on the SEM data-fidelity loss + DAGMA penalty, and
      (ii) a denoising step from a diffusion model trained on weighted DAGs.

    When the denoiser is a GDSSDiffusionDenoiser (has a .sde attribute):
      - A VP-SDE time schedule sweeps t from T down to t_min over num_iters.
      - Every denoise_every iterations, forward VP-SDE noise is injected into z,
        a Tweedie posterior-mean estimate W_struct is computed via the score network,
        and the result is mixed with z via an SNR-based weight:
            denoiser_w = std(t_k) / mean_coeff(t_k)   (high early, low late)
            W_tilde    = (1 - denoiser_w) * z  +  denoiser_w * W_struct
      - On all other iterations W_tilde = z (pure gradient step).

    For other denoisers (no .sde), the residual-based noise schedule (eq. 12)
    is used and the denoiser is applied every denoise_every iterations with
    full weight (W_tilde = W_struct).

    Args
    ----
    X             : (n, d) observed data matrix.
    denoiser      : Callable D(W, sigma) -> W_denoised, or GDSSDiffusionDenoiser.
    num_iters     : Total iterations K.
    lambda_data   : Weight for the data-fidelity term.
    lambda_dag    : Weight for the DAGMA acyclicity penalty.
    lambda_l1     : Weight for the L1 sparsity penalty on W (0.0 = disabled).
    eta           : Base step size eta > 0.
    c             : Safety constant for the barrier step (0 < c < 1).
    epsilon       : Smoothing constant (> 0, prevents /0).
    s             : DAGMA parameter s > 0.
    alpha         : Noise-schedule scaling (eq. 12, used when no .sde).
    sigma_min     : Minimum denoiser noise level (used when no .sde).
    sigma_max     : Maximum denoiser noise level (used when no .sde).
    smooth_beta   : EMA coefficient for noise-level smoothing (used when no .sde).
    W_max         : Frobenius norm bound for K.
    delta_min     : Minimum determinant buffer for K.
    W_init        : Initial W^(0); defaults to the zero matrix.
    project       : Whether to project onto K after each denoising step.
    inject_noise  : Set False for IdentityDenoiser / pure DAGMA (skips denoiser).
    denoise_every : Apply the denoiser every this many iterations.
    verbose       : Print progress.
    log_every     : Print every this many iterations.

    Returns
    -------
    W        : (d, d) estimated weighted adjacency matrix.
    history  : Dict with per-iteration 'loss', 'h', 'sigma', 'gamma', 'denoiser_w'.
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

    history: Dict = {"loss": [], "h": [], "sigma": [], "gamma": [], "denoiser_w": []}
    sigma_prev: Optional[float] = None

    # ------------------------------------------------------------------
    # VP-SDE time schedule (when denoiser has a .sde attribute)
    # ------------------------------------------------------------------
    _has_sde = inject_noise and hasattr(denoiser, 'sde')
    if _has_sde:
        n_denoise_steps = (num_iters + denoise_every - 1) // denoise_every
        t_vals = np.linspace(float(denoiser.sde.T), denoiser.t_min, n_denoise_steps)
        _dummy = torch.zeros(1, 1, 1, device=device)
        _, _std0 = denoiser.sde.marginal_prob(
            _dummy, torch.tensor([t_vals[0]], device=device)
        )
        sigma_start = _std0.item()

    if verbose:
        print(f"DAG-PnP | d={d}  n={X.shape[0]}  iters={num_iters}")
        print(f"  lambda_data={lambda_data}  lambda_dag={lambda_dag}  "
              f"eta={eta}  c={c}  s={s:.3f}  W_max={W_max:.2f}")
        if _has_sde:
            _dummy = torch.zeros(1, 1, 1, device=device)
            _, _std_end = denoiser.sde.marginal_prob(
                _dummy, torch.tensor([t_vals[-1]], device=device)
            )
            print(f"  noise: VP-SDE t[{t_vals[0]:.3f} -> {t_vals[-1]:.4f}]"
                  f"  =>  sigma[{sigma_start:.3f} -> {_std_end.item():.4f}]")
            print(f"  mixing: denoiser_weight = std(t) / mean_coeff(t)  (high->low)")
        else:
            print(f"  alpha={alpha}  sigma=[{sigma_min},{sigma_max}]  "
                  f"delta_min={delta_min}")
        print(f"  denoise_every={denoise_every}")
        print("-" * 65)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    for k in range(num_iters):

        # ---- Step 1: gradient of L_ext  (eqs. 1, 3, 5) ----
        grad = grad_lext(X, W, lambda_data, lambda_dag, s)
        if lambda_l1 > 0.0:
            grad = grad + lambda_l1 * torch.sign(W)

        # ---- Step 2: barrier-aware step size  (eq. 11) ----
        gamma = _barrier_step_size(W, grad, s, eta, c, epsilon)

        # ---- Step 3: gradient step  (eq. 6) ----
        z = W - gamma * grad

        # ---- Steps 4–5: noise schedule + denoising + SNR mixing ----
        if k % denoise_every == 0:

            if _has_sde:
                # Step 4 — VP-SDE renoising: z_tilde = mean(t_k)*z + std(t_k)*eps
                t_k   = torch.tensor([t_vals[k // denoise_every]], device=device, dtype=dtype)
                z_b   = z.unsqueeze(0)                                       # (1,d,d)
                mean_z, std_k = denoiser.sde.marginal_prob(z_b, t_k)        # (1,d,d),(1,)
                z_tilde = (mean_z + std_k.view(-1, 1, 1) * torch.randn_like(z_b)).squeeze(0)
                sigma_k = std_k.item()

                # Step 5 — Tweedie denoising on z_tilde via score network
                with torch.no_grad():
                    t_d    = torch.tensor([t_vals[k // denoise_every]], device=denoiser.device, dtype=torch.float32)
                    adj_t  = z_tilde.unsqueeze(0).to(denoiser.device, torch.float32)
                    x_feat = torch.ones(1, d, denoiser.max_feat_num, device=denoiser.device)
                    flags  = torch.ones(1, d, device=denoiser.device)
                    mean_c, std_td = denoiser.sde.marginal_prob(torch.ones_like(adj_t), t_d)
                    score  = denoiser.score_fn(x_feat, adj_t, flags, t_d)
                    std_sq = (std_td ** 2).view(1, 1, 1)
                    W_struct = ((adj_t + std_sq * score) / mean_c).squeeze(0).cpu().to(dtype)

                # SNR-based mixing: high weight early (explore), low weight late (exploit)
                denoiser_w = min((std_td / mean_c[0, 0, 0]).item(), 0.4)
                W_tilde = (1.0 - denoiser_w) * z + denoiser_w * W_struct.to(device)


            elif inject_noise:
                # Residual-based noise schedule (no VP-SDE)
                sigma_k = _noise_schedule(
                    X, W, alpha, sigma_min, sigma_max, sigma_prev, smooth_beta
                )
                sigma_prev = sigma_k
                W_struct   = denoiser(z, sigma_k)
                denoiser_w = 1.0
                W_tilde    = W_struct

            else:
                # DAGMA / identity denoiser: pure gradient descent
                sigma_k    = 0.0
                denoiser_w = 0.0
                W_tilde    = z

        else:
            # Pure gradient step — skip denoiser this iteration
            sigma_k    = 0.0
            denoiser_w = 0.0
            W_tilde    = z

        # ---- Step 6: projection step  (eq. 8) ----
        W = _project_onto_K(W_tilde, s, delta_min, W_max) if project else W_tilde

        # ---- Logging ----
        loss_val = data_loss(X, W).item()
        h_val    = dagma_penalty(W, s).item()
        history["loss"].append(loss_val)
        history["h"].append(h_val)
        history["sigma"].append(sigma_k)
        history["gamma"].append(gamma)
        history["denoiser_w"].append(denoiser_w)

        if verbose and (k % log_every == 0 or k == num_iters - 1):
            print(
                f"  iter {k:4d}/{num_iters} | "
                f"loss={loss_val:.5f}  h={h_val:.5f}  "
                f"dw={denoiser_w:.3f}  sigma={sigma_k:.4f}  gamma={gamma:.3e}"
            )

    if verbose:
        print("-" * 65)
        print("Done.")

    return W, history


