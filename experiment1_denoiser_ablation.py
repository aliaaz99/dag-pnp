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
"""

import os
import time
import numpy as np
import torch
from collections import defaultdict
from typing import Dict

from data_utils import (random_dag, sample_sem, evaluate_dag, to_tensor,
                        scale_free_dag, normalise_X, denormalise_W)
from dag_pnp import dag_pnp, IdentityDenoiser, GaussianShrinkageDenoiser, GDSSDiffusionDenoiser
from baselines import dagma_baseline, notears_baseline, graph_gen_baseline
from losses_dag import data_loss, dagma_penalty
from plot_utils import print_results_table
from dag_config import shared_kwargs, K_EDGES, WEIGHT_LOW, WEIGHT_HIGH, NORMALISE, get_threshold


# ===========================================================================
# !! PLUG IN YOUR DENOISER HERE !!
# ===========================================================================
# def get_denoiser():
#     # prior_std = midpoint of [WEIGHT_LOW, WEIGHT_HIGH] = (0.5 + 1.0) / 2
#     return IdentityDenoiser()  # <-- ablation: no denoising
#     # return GaussianShrinkageDenoiser(prior_std=0.75)  # <-- replace with real denoiser

def get_denoiser():
    return GDSSDiffusionDenoiser.from_checkpoint(
        ckpt_path="gdss/checkpoints/sem_SF_d20/Apr16-01:12:06.pth", # for SF
        # ckpt_path="gdss/checkpoints/sem_ER_d20/Apr18-21_02_13.pth", # for ER
        device="cpu",   # or "cuda:0" if you have a GPU
    )

# ===========================================================================
# Configuration
# ===========================================================================
D_VALUES  = [20] # 50
SF_M      = K_EDGES
NUM_SEEDS = 10

METRICS = ["shd", "f1", "tpr", "precision", "acyclicity", "fit_error"]


# ===========================================================================
# Helpers
# ===========================================================================
def _extra_metrics(W_np: np.ndarray, X: torch.Tensor, s: float,
                   W_true_np: np.ndarray = None) -> Dict:
    W_t = torch.tensor(W_np, dtype=X.dtype, device=X.device)
    l_data = data_loss(X, W_t).item()
    metrics = {
        "acyclicity": dagma_penalty(W_t, s).item(),
        "fit_error":  l_data,
    }
    if W_true_np is not None:
        W_true_t = torch.tensor(W_true_np, dtype=X.dtype, device=X.device)
        l_data_true = data_loss(X, W_true_t).item()
        metrics["fit_error_ratio"] = (l_data / l_data_true
                                      if l_data_true > 0 else float("nan"))
    return metrics


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
def run_one_setting(d, n, graph_family, denoiser, save_dir, normalise=True):
    methods = ["NOTEARS", "DAGMA", "Greedy-BIC", "DAG-PnP"]
    raw: Dict = {m: defaultdict(list) for m in methods}
    times: Dict = {m: [] for m in methods}

    # d-adaptive kwargs — key fix
    kw = shared_kwargs(d)
    s  = kw["s"]

    print(f"\n  s={s:.2f}  W_max={kw['W_max']:.2f}  "
          f"lambda_dag={kw['lambda_dag']}  num_iters={kw['num_iters']}  "
          f"normalise={normalise}")

    for seed in range(NUM_SEEDS):
        rng = seed * 1000

        if graph_family == "ER":
            W_true = random_dag(d, k=K_EDGES, weight_low=WEIGHT_LOW,
                                weight_high=WEIGHT_HIGH, seed=rng)
        else:
            W_true = scale_free_dag(d, m=SF_M, weight_low=WEIGHT_LOW,
                                    weight_high=WEIGHT_HIGH, seed=rng)

        X_np = sample_sem(W_true, n, seed=rng + 1)
        X    = to_tensor(X_np)

        if normalise:
            X_opt, denorm = normalise_X(X)
        else:
            X_opt  = X
            denorm = np.ones((d, d), dtype=np.float32)

        def _record(method_name, W_norm_np):
            W_np = denormalise_W(W_norm_np, denorm)
            m = evaluate_dag(W_np, W_true, get_threshold(W_np))
            m.update(_extra_metrics(W_np, X, s, W_true_np=W_true))
            for k, v in m.items():
                raw[method_name][k].append(v)

        # ---- NOTEARS ----
        t0 = time.perf_counter()
        W_n_norm, _ = notears_baseline(
            X_opt, lambda_data=kw["lambda_data"], lambda_dag=kw["lambda_dag"],
            num_iters=kw["num_iters"], lr=kw["eta"],
            W_max=kw["W_max"], # grad_clip=10.0,
        )
        times["NOTEARS"].append(time.perf_counter() - t0)
        _record("NOTEARS", W_n_norm)

        # ---- DAGMA ----
        t0 = time.perf_counter()
        W_d_norm, _ = dagma_baseline(X_opt, **kw)
        times["DAGMA"].append(time.perf_counter() - t0)
        _record("DAGMA", W_d_norm)

        # ---- Greedy-BIC ----
        t0 = time.perf_counter()
        W_g, _ = graph_gen_baseline(X_opt, max_parents=min(5, d // 2))
        times["Greedy-BIC"].append(time.perf_counter() - t0)
        _record("Greedy-BIC", W_g) # placeholder: reuse DAGMA

        # ---- DAG-PnP ----
        t0 = time.perf_counter()
        W_p, _ = dag_pnp(X_opt, denoiser, inject_noise=True, verbose=False, lambda_l1=0.000,**kw)
        times["DAG-PnP"].append(time.perf_counter() - t0)
        _record("DAG-PnP", W_p.detach().numpy())
        # _record("DAG-PnP", W_p)      # placeholder: reuse DAGMA

        if (seed + 1) % 2 == 0:
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

    print(f"\n  Runtime (seconds)   d={d}   seeds={NUM_SEEDS}")
    print("  " + "─" * 44)
    print(f"  {'Method':<14}  {'Mean (s)':>10}  {'Std (s)':>10}")
    print("  " + "─" * 44)
    for m in methods:
        t_arr = times[m]
        print(f"  {m:<14}  {np.mean(t_arr):>10.3f}  {np.std(t_arr):>10.3f}")
    print("  " + "─" * 44)

    csv_path = _save_csv(raw, d, graph_family, save_dir)
    print(f"  Raw results saved → {csv_path}\n")


def run_experiment1(save_dir: str = "results", normalise: bool = NORMALISE):
    os.makedirs(save_dir, exist_ok=True)
    denoiser = get_denoiser()

    for graph_family in ["SF"]:
        for d in D_VALUES:
            print(f"\n{'='*60}")
            print(f"  {graph_family} graph   d={d}   n={d}   seeds={NUM_SEEDS}")
            print(f"{'='*60}")
            run_one_setting(d, d, graph_family, denoiser, save_dir,
                            normalise=normalise)


if __name__ == "__main__":
    run_experiment1()
