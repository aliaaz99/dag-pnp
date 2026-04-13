"""
Experiment 1: Main results table — with denoiser vs. without.

Matches the paper's planned Table 1:
  - Graph families : Erdos-Renyi (ER) and scale-free (SF)
  - Graph sizes    : d in {10, 20, 30}
  - Baselines      : NOTEARS, DAGMA, Greedy-BIC, DAG-PnP
  - Metrics        : SHD, acyclicity h(W), fit error L_data, F1

NOTE
----
Until a real diffusion denoiser is plugged in, DAG-PnP and DAGMA will be
identical (IdentityDenoiser).  Swap get_denoiser() for your trained model.

HYPERPARAMETER NOTE
-------------------
s and W_max are set adaptively per graph size d via dag_config.shared_kwargs(d).
See dag_config.py for the derivation.  Key changes vs. earlier broken runs:
  - lambda_dag : 0.1  -> 0.02  (was overwhelming the data gradient)
  - s          : 1.0  -> ~4-12 per d  (true W was outside feasible set)
  - W_max      : 10.0 -> ~9-27 per d  (projection was clipping the true W)
  - weight_high: 2.0  -> 1.0           (reduces true W norm; s stays moderate)
  - num_iters  : 500  -> 1000
"""

import os
import numpy as np
import torch
from collections import defaultdict
from typing import Dict

from data_utils import random_dag, sample_sem, evaluate_dag, to_tensor, scale_free_dag
from dag_pnp import dag_pnp, IdentityDenoiser, GaussianShrinkageDenoiser
from baselines import dagma_baseline, notears_baseline, graph_gen_baseline
from losses import data_loss, dagma_penalty
from plot_utils import print_results_table
from dag_config import shared_kwargs, SPARSITY, WEIGHT_LOW, WEIGHT_HIGH, THRESHOLD


# ===========================================================================
# !! PLUG IN YOUR DENOISER HERE !!
# ===========================================================================
def get_denoiser():
    # prior_std = midpoint of [WEIGHT_LOW, WEIGHT_HIGH] = (0.5 + 1.0) / 2
    return IdentityDenoiser()  # <-- ablation: no denoising
    # return GaussianShrinkageDenoiser(prior_std=0.75)  # <-- replace with real denoiser


# ===========================================================================
# Configuration
# ===========================================================================
D_VALUES  = [10, 20, 50]
N_SAMPLES = 500
SF_M      = 2
NUM_SEEDS = 5

METRICS = ["shd", "acyclicity", "fit_error", "f1"]


# ===========================================================================
# Helpers
# ===========================================================================
def _extra_metrics(W_np: np.ndarray, X: torch.Tensor, s: float) -> Dict:
    W_t = torch.tensor(W_np, dtype=X.dtype, device=X.device)
    return {
        "acyclicity": dagma_penalty(W_t, s).item(),
        "fit_error":  data_loss(X, W_t).item(),
    }


def _save_csv(raw: Dict, d: int, graph_family: str, save_dir: str):
    import csv
    fname = os.path.join(save_dir, f"exp1_raw_{graph_family}_d{d}.csv")
    all_keys = list(next(iter(raw.values())).keys())
    with open(fname, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "seed"] + all_keys)
        writer.writeheader()
        for method, mdict in raw.items():
            n_seeds = len(next(iter(mdict.values())))
            for seed in range(n_seeds):
                row = {"method": method, "seed": seed}
                row.update({k: mdict[k][seed] for k in all_keys})
                writer.writerow(row)
    return fname


# ===========================================================================
# Main
# ===========================================================================
def run_one_setting(d, n, graph_family, denoiser, save_dir):
    methods = ["NOTEARS", "DAGMA", "Greedy-BIC", "DAG-PnP"]
    raw: Dict = {m: defaultdict(list) for m in methods}

    # d-adaptive kwargs — key fix
    kw = shared_kwargs(d)
    s  = kw["s"]

    print(f"\n  s={s:.2f}  W_max={kw['W_max']:.2f}  "
          f"lambda_dag={kw['lambda_dag']}  num_iters={kw['num_iters']}")

    for seed in range(NUM_SEEDS):
        rng = seed * 1000

        if graph_family == "ER":
            W_true = random_dag(d, sparsity=SPARSITY, weight_low=WEIGHT_LOW,
                                weight_high=WEIGHT_HIGH, seed=rng)
        else:
            W_true = scale_free_dag(d, m=SF_M, weight_low=WEIGHT_LOW,
                                    weight_high=WEIGHT_HIGH, seed=rng)

        X_np = sample_sem(W_true, n, noise_std=1.0, seed=rng + 1)
        X    = to_tensor(X_np)

        def _record(method_name, W_est_np):
            m = evaluate_dag(W_est_np, W_true, THRESHOLD)
            m.update(_extra_metrics(W_est_np, X, s))
            for k, v in m.items():
                raw[method_name][k].append(v)

        # ---- NOTEARS (increased W_max; see baselines.py) ----
        W_n, _ = notears_baseline(
            X, lambda_data=1.0, lambda_dag=0.02, num_iters=kw["num_iters"], lr=0.001,
            W_max=kw["W_max"], grad_clip=10.0,
        )
        _record("NOTEARS", W_n)

        # ---- DAGMA ----
        W_d, _ = dagma_baseline(X, **kw)
        _record("DAGMA", W_d)

        # ---- Greedy-BIC ----
        W_g, _ = graph_gen_baseline(X, max_parents=min(5, d // 2))
        _record("Greedy-BIC", W_g)

        # ---- DAG-PnP ----
        W_p, _ = dag_pnp(X, denoiser, verbose=False, **kw)
        _record("DAG-PnP", W_p.detach().numpy())

        if (seed + 1) % 5 == 0:
            print(f"    seed {seed+1}/{NUM_SEEDS} done")

    summary = {
        method: {
            metric: (float(np.mean(vals)), float(np.std(vals)))
            for metric, vals in mdict.items()
        }
        for method, mdict in raw.items()
    }
    print_results_table(summary, d=d, n=n, graph_type=graph_family,
                        metrics=METRICS)

    csv_path = _save_csv(raw, d, graph_family, save_dir)
    print(f"  Raw results saved → {csv_path}\n")


def run_experiment1(save_dir: str = "results"):
    os.makedirs(save_dir, exist_ok=True)
    denoiser = get_denoiser()

    for graph_family in ["ER", "SF"]:
        for d in D_VALUES:
            print(f"\n{'='*60}")
            print(f"  {graph_family} graph   d={d}   n={N_SAMPLES}   seeds={NUM_SEEDS}")
            print(f"{'='*60}")
            run_one_setting(d, N_SAMPLES, graph_family, denoiser, save_dir)


if __name__ == "__main__":
    run_experiment1()
