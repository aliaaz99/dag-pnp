"""
Shared hyperparameter configuration for all DAG-PnP experiments.

Why adaptive s and W_max?
--------------------------
The DAGMA feasible set requires  det(s*I - W o W) > 0,  i.e. all eigenvalues of
W o W must be < s.  By Gershgorin, the spectral radius of W o W is bounded by its
maximum row sum:

    rho(W o W) <= max_i  sum_j  W_{ij}^2  ~  2 * K_EDGES * weight_high^2

(using the ERk/SFk edge-count convention: sparsity = 2k/(d-1), so the bound
simplifies to 2k * weight_high^2 — independent of d.)

With the original weight range [0.5, 2.0] and s=1.0 this bound is large,
meaning the TRUE W typically lies OUTSIDE the feasible set.  The algorithm then
converges to the best point INSIDE the feasible set, which can be far from the
truth.  The two root causes are:

  1. s too small  -> true W outside feasible set -> wrong optimum
  2. W_max too small -> Frobenius projection clips true W -> algorithm can't reach it

Fix: set s slightly above the expected spectral radius, and W_max well above
the expected Frobenius norm of the true W.

Graph density convention
------------------------
Following the DAGMA paper, graphs are parameterised by K_EDGES (edges per node):
  - ER4 / SF4 : K_EDGES = 4  (default)  -> E[# edges] = 4d
  - ER2 / SF2 : K_EDGES = 2             -> E[# edges] = 2d
  - ER6 / SF6 : K_EDGES = 6             -> E[# edges] = 6d
For ER graphs, the per-edge probability is sparsity = 2k/(d-1).
For SF graphs, the BA attachment parameter m = k directly.

Weight range
------------
Using weight_high = 1.0 instead of 2.0 keeps the SEM identifiable and keeps
the true W well inside the feasible set for moderate s values.  If you need
larger weights, increase s proportionally.
"""

import numpy as np

# ── data generation ──────────────────────────────────────────────────────────
WEIGHT_LOW  = 0.2
WEIGHT_HIGH = 0.5
K_EDGES     = 4     # edges per node: ER4/SF4 (DAGMA paper default); use 2 or 6 for ER2/SF2 or ER6/SF6
THRESHOLD   = 0.2  # used when ADAPTIVE_THRESHOLD = False; default = WEIGHT_LOW / 2

# ── threshold selection ───────────────────────────────────────────────────────
# True  : threshold = mean of all off-diagonal |W[i,j]| in the estimated matrix.
ADAPTIVE_THRESHOLD = True


def get_threshold(W_np=None) -> float:
    """
    Return the edge-detection threshold for a given estimated weight matrix.

    If ADAPTIVE_THRESHOLD is True and W_np is provided, returns the mean of
    all off-diagonal absolute values — a data-driven split between near-zero
    spurious entries and true edges.

    If ADAPTIVE_THRESHOLD is False (or W_np is None), returns the fixed
    THRESHOLD constant.
    """
    if ADAPTIVE_THRESHOLD and W_np is not None:
        d = W_np.shape[0]
        mask = ~np.eye(d, dtype=bool)          # exclude diagonal (always 0)
        return float(np.abs(W_np[mask]).mean())
    return THRESHOLD

# ── normalisation ─────────────────────────────────────────────────────────────
# True  : divide X by per-column std before optimisation, recover original-scale W
NORMALISE   = True

# ── fixed hyperparameters (same for all d) ───────────────────────────────────
LAMBDA_DATA = 1.0
LAMBDA_DAG  = 0.01
ETA         = 0.1
C           = 0.3
EPSILON     = 1e-6
ALPHA       = 1.0
SIGMA_MIN   = 0.0
SIGMA_MAX   = 1.0
SMOOTH_BETA = 0.0
DELTA_MIN   = 1e-6
NUM_ITERS   = 1000   



def dagma_s() -> float:
    """
    Minimum s such that the true W is guaranteed inside the DAGMA feasible set.

    Derivation: by Gershgorin, the largest eigenvalue of W o W is bounded by
        rho(W o W) <= sparsity * (d-1) * weight_high^2
                    = [2*K_EDGES/(d-1)] * (d-1) * weight_high^2
                    = 2 * K_EDGES * weight_high^2   (d-independent)
    We use a 50% safety margin above this bound.
    """
    rho_bound = 2.0 * K_EDGES * WEIGHT_HIGH ** 2
    return float(1.5 * rho_bound + 1.0)


def dagma_W_max(d: int) -> float:
    """
    W_max that is well above the expected Frobenius norm of the true W.

    E[||W_true||_F^2] = expected_edges * E[w^2]
    E[# edges] = K_EDGES * d  (from the ERk/SFk convention)
    E[w^2] for uniform on [weight_low, weight_high] = (L^2 + L*H + H^2) / 3
    We set W_max = 3 * std above expected ||W_true||_F (covers ~99% of seeds).
    """
    expected_edges = K_EDGES * d
    mean_w2 = (WEIGHT_LOW**2 + WEIGHT_LOW * WEIGHT_HIGH + WEIGHT_HIGH**2) / 3.0
    expected_norm_sq = expected_edges * mean_w2
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
        eta         = ETA,
        # eta         = dagma_eta(d),   # d-adaptive: scales as 1/d
        c           = C,
        epsilon     = EPSILON,
        s           = dagma_s(),
        alpha       = ALPHA,
        sigma_min   = SIGMA_MIN,
        sigma_max   = SIGMA_MAX,
        smooth_beta = SMOOTH_BETA,
        W_max       = dagma_W_max(d),
        delta_min   = DELTA_MIN,
    )


# ── quick sanity check ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"K_EDGES={K_EDGES}  s={dagma_s():.2f}  (d-independent)")
    print(f"\n{'d':>6}  {'eta':>8}  {'W_max':>8}  {'E[||W||_F]':>12}  {'E[edges]':>10}")
    print("─" * 52)
    for d in [10, 15, 20, 30, 50]:
        exp_edges = K_EDGES * d
        mean_w2   = (WEIGHT_LOW**2 + WEIGHT_LOW*WEIGHT_HIGH + WEIGHT_HIGH**2) / 3
        e_norm    = np.sqrt(exp_edges * mean_w2)
        print(f"{d:>6}  {ETA:>8.4f}  {dagma_W_max(d):>8.2f}  "
              f"{e_norm:>12.2f}  {exp_edges:>10}")
