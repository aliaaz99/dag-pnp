"""
Shared hyperparameter configuration for all DAG-PnP experiments.

Why adaptive s and W_max?
--------------------------
The DAGMA feasible set requires  det(s*I - W o W) > 0,  i.e. all eigenvalues of
W o W must be < s.  By Gershgorin, the spectral radius of W o W is bounded by its
maximum row sum:

    rho(W o W) <= max_i  sum_j  W_{ij}^2  ~  sparsity * (d-1) * weight_high^2

With the original weight range [0.5, 2.0] and s=1.0 this bound is ~15 for d=30,
meaning the TRUE W typically lies OUTSIDE the feasible set.  The algorithm then
converges to the best point INSIDE the feasible set, which can be far from the
truth.  The two root causes are:

  1. s too small  -> true W outside feasible set -> wrong optimum
  2. W_max too small -> Frobenius projection clips true W -> algorithm can't reach it

Fix: set s slightly above the expected spectral radius, and W_max well above
the expected Frobenius norm of the true W.  Both scale with d.

Weight range
------------
Using weight_high = 1.0 instead of 2.0 keeps the SEM identifiable and keeps
the true W well inside the feasible set for moderate s values.  If you need
larger weights, increase s proportionally.
"""

import numpy as np

# ── data generation ──────────────────────────────────────────────────────────
WEIGHT_LOW  = 0.5
WEIGHT_HIGH = 1.0   
SPARSITY    = 0.3
THRESHOLD   = 0.3 

# ── fixed hyperparameters (same for all d) ───────────────────────────────────
LAMBDA_DATA = 1.0
LAMBDA_DAG  = 0.01  
ETA         = 0.005
ETA_D_REF   = 20     # reference graph size for step-size scaling
C           = 0.3
EPSILON     = 1e-6
ALPHA       = 1.0
SIGMA_MIN   = 0.01
SIGMA_MAX   = 0.5
SMOOTH_BETA = 0.0
DELTA_MIN   = 1e-6
NUM_ITERS   = 1000   


def dagma_eta(d: int) -> float:
    """
    d-adaptive step size cap: eta(d) = ETA * D_REF / d.

    Rationale: gradient magnitudes of L_data scale as O(d) through the
    covariance structure of the SEM.  Keeping eta * d constant ensures
    the effective step magnitude is independent of graph size and prevents
    boundary violations / divergence at large d (≥ 50).
    """
    return min(ETA, ETA * ETA_D_REF / d)

def dagma_s(d: int) -> float:
    """
    Minimum s such that the true W is guaranteed inside the DAGMA feasible set.

    Derivation: by Gershgorin, the largest eigenvalue of W o W is bounded by
        rho(W o W) <= sparsity * (d-1) * weight_high^2
    We use a 50% safety margin above this bound.
    """
    rho_bound = SPARSITY * (d - 1) * WEIGHT_HIGH ** 2
    return float(1.5 * rho_bound + 1.0)   # +1 keeps it above rho_bound always


def dagma_W_max(d: int) -> float:
    """
    W_max that is well above the expected Frobenius norm of the true W.

    E[||W_true||_F^2] = expected_edges * E[w^2]
    E[w^2] for uniform on [weight_low, weight_high] = (L^2 + L*H + H^2) / 3
    We set W_max = 3 * std above expected ||W_true||_F (covers ~99% of seeds).
    """
    expected_edges = SPARSITY * d * (d - 1) / 2.0
    mean_w2 = (WEIGHT_LOW**2 + WEIGHT_LOW * WEIGHT_HIGH + WEIGHT_HIGH**2) / 3.0
    expected_norm_sq = expected_edges * mean_w2
    # 3× the expected norm gives generous headroom
    return float(3.0 * np.sqrt(expected_norm_sq) + 1.0)


def shared_kwargs(d: int) -> dict:
    """
    Return the full SHARED_KWARGS dict for a given graph size d.

    All DAGMA/DAG-PnP runs should call this instead of using a hardcoded dict,
    so that s and W_max automatically scale with the graph size.
    """
    return dict(
        num_iters   = NUM_ITERS,
        lambda_data = LAMBDA_DATA,
        lambda_dag  = LAMBDA_DAG,
        # eta         = ETA,
        eta         = dagma_eta(d),   # d-adaptive: scales as 1/d
        c           = C,
        epsilon     = EPSILON,
        s           = dagma_s(d),
        alpha       = ALPHA,
        sigma_min   = SIGMA_MIN,
        sigma_max   = SIGMA_MAX,
        smooth_beta = SMOOTH_BETA,
        W_max       = dagma_W_max(d),
        delta_min   = DELTA_MIN,
    )


# ── quick sanity check ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"{'d':>6}  {'s':>8}  {'W_max':>8}  "
          f"{'E[||W||_F]':>12}  {'rho(WoW) bound':>16}")
    print("─" * 60)
    for d in [10, 15, 20, 30, 50]:
        exp_edges = SPARSITY * d * (d - 1) / 2
        mean_w2   = (WEIGHT_LOW**2 + WEIGHT_LOW*WEIGHT_HIGH + WEIGHT_HIGH**2) / 3
        e_norm    = np.sqrt(exp_edges * mean_w2)
        rho       = SPARSITY * (d - 1) * WEIGHT_HIGH**2
        print(f"{d:>6}  {dagma_eta(d):>8.4f}  {dagma_s(d):>8.2f}  {dagma_W_max(d):>8.2f}  "
              f"{e_norm:>12.2f}  {rho:>16.2f}")
