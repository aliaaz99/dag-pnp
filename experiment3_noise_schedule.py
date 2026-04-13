"""
Experiment 3: Residual-based noise schedule vs. fixed noise level (ablation).

Residual schedule (eq. 12) sets sigma_k proportional to the normalised data
residual — strong prior early, lighter denoising near convergence.
Compared against fixed-sigma variants at sigma_max and sigma_min.

Output files (written to save_dir):
  exp3_convergence.png
  exp3_raw.csv
"""

import os
import csv
import numpy as np
import torch
from collections import defaultdict
from typing import Dict

from data_utils import random_dag, sample_sem, evaluate_dag, to_tensor, scale_free_dag
from dag_pnp import dag_pnp, IdentityDenoiser, GaussianShrinkageDenoiser
from baselines import dag_pnp_fixed_sigma
from losses import data_loss, dagma_penalty
from plot_utils import print_results_table, plot_noise_schedule
from dag_config import (shared_kwargs, SPARSITY, WEIGHT_LOW, WEIGHT_HIGH,
                        SIGMA_MIN, SIGMA_MAX, THRESHOLD)


# ===========================================================================
# !! PLUG IN YOUR DENOISER HERE !!
# ===========================================================================
def get_denoiser():
    return IdentityDenoiser()  # <-- ablation: no denoising
    # return GaussianShrinkageDenoiser(prior_std=1.5)  # <-- replace with real denoiser


# ===========================================================================
# Configuration
# ===========================================================================
D_CONV    = 50
D_VALUES  = [10, 20, 30]
N_SAMPLES = 500
NUM_SEEDS = 10
SEED_CONV = 42
SF_M      = 2   # uncomment the scale_free_dag(...) lines below to use SF ground truth

SCHEDULE_VARIANTS = {
    "Residual (ours)":      "residual",
    f"Fixed σ={SIGMA_MAX}": "fixed_max",
    f"Fixed σ={SIGMA_MIN}": "fixed_min",
}
METRICS = ["shd", "acyclicity", "fit_error", "f1"]


# ===========================================================================
# Helpers
# ===========================================================================
def _extra_metrics(W_np, X, s):
    W_t = torch.tensor(W_np, dtype=X.dtype, device=X.device)
    return {
        "acyclicity": dagma_penalty(W_t, s).item(),
        "fit_error":  data_loss(X, W_t).item(),
    }


def _save_metrics_csv(all_raw, save_dir):
    fname = os.path.join(save_dir, "exp3_raw.csv")
    metric_names = []
    for d in all_raw:
        for label in all_raw[d]:
            for k in all_raw[d][label]:
                if k not in metric_names:
                    metric_names.append(k)
    with open(fname, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["d", "variant", "seed"] + metric_names)
        writer.writeheader()
        for d, raw in all_raw.items():
            for label, mdict in raw.items():
                n_seeds = len(next(iter(mdict.values())))
                for seed in range(n_seeds):
                    row = {"d": d, "variant": label, "seed": seed}
                    row.update({k: mdict[k][seed] for k in metric_names
                                if seed < len(mdict.get(k, []))})
                    writer.writerow(row)
    return fname


# ===========================================================================
# Part (a): convergence curves
# ===========================================================================
def run_convergence(denoiser, save_dir):
    print("\n── Part (a): convergence curves ──")
    kw = shared_kwargs(D_CONV)
    s  = kw["s"]

    W_true = random_dag(D_CONV, sparsity=SPARSITY, weight_low=WEIGHT_LOW,
                        weight_high=WEIGHT_HIGH, seed=SEED_CONV)
    # To use SF ground truth instead, uncomment the next line and comment out random_dag(...) above.
    # W_true = scale_free_dag(D_CONV, m=SF_M, weight_low=WEIGHT_LOW,
    #                         weight_high=WEIGHT_HIGH, seed=SEED_CONV)
    X_np   = sample_sem(W_true, N_SAMPLES, seed=SEED_CONV + 1)
    X      = to_tensor(X_np)

    histories = {}

    _, h_res = dag_pnp(
        X, denoiser,
        verbose=False, **kw,
    )
    histories["Residual (ours)"] = h_res

    _, h_max = dag_pnp_fixed_sigma(X, denoiser, fixed_sigma=SIGMA_MAX, **kw)
    histories[f"Fixed σ={SIGMA_MAX}"] = h_max

    _, h_min = dag_pnp_fixed_sigma(X, denoiser, fixed_sigma=SIGMA_MIN, **kw)
    histories[f"Fixed σ={SIGMA_MIN}"] = h_min

    plot_noise_schedule(
        histories,
        save_path=os.path.join(save_dir, "exp3_convergence.png"),
    )

    print(f"\n  {'Variant':<25}  {'Final loss':>12}  {'Final h(W)':>12}  "
          f"{'Final sigma':>12}")
    print("  " + "─" * 66)
    for label, hist in histories.items():
        print(f"  {label:<25}  {hist['loss'][-1]:>12.5f}  "
              f"{hist['h'][-1]:>12.5f}  {hist['sigma'][-1]:>12.4f}")


# ===========================================================================
# Part (b): metrics table across seeds and graph sizes
# ===========================================================================
def run_metrics_table(denoiser, save_dir):
    print("\n── Part (b): final metrics across seeds ──")
    all_raw = {}

    for d in D_VALUES:
        kw = shared_kwargs(d)
        s  = kw["s"]
        raw: Dict = {label: defaultdict(list) for label in SCHEDULE_VARIANTS}

        for seed in range(NUM_SEEDS):
            rng = seed * 1000
            W_true = random_dag(d, sparsity=SPARSITY, weight_low=WEIGHT_LOW,
                                weight_high=WEIGHT_HIGH, seed=rng)
            # To use SF ground truth instead, uncomment the next line and comment out random_dag(...) above.
            # W_true = scale_free_dag(d, m=SF_M, weight_low=WEIGHT_LOW,
            #                         weight_high=WEIGHT_HIGH, seed=rng)
            X_np = sample_sem(W_true, N_SAMPLES, seed=rng + 1)
            X    = to_tensor(X_np)

            def _record(label, W_np):
                m = evaluate_dag(W_np, W_true, THRESHOLD)
                m.update(_extra_metrics(W_np, X, s))
                for k, v in m.items():
                    raw[label][k].append(v)

            W_r, _ = dag_pnp(X, denoiser,
                             verbose=False, **kw)
            _record("Residual (ours)", W_r.detach().numpy())

            W_mx, _ = dag_pnp_fixed_sigma(X, denoiser, fixed_sigma=SIGMA_MAX, **kw)
            _record(f"Fixed σ={SIGMA_MAX}", W_mx.detach().numpy())

            W_mn, _ = dag_pnp_fixed_sigma(X, denoiser, fixed_sigma=SIGMA_MIN, **kw)
            _record(f"Fixed σ={SIGMA_MIN}", W_mn.detach().numpy())

            if (seed + 1) % 5 == 0:
                print(f"    d={d}  seed {seed+1}/{NUM_SEEDS} done")

        summary = {
            label: {
                metric: (float(np.mean(vals)), float(np.std(vals)))
                for metric, vals in mdict.items()
            }
            for label, mdict in raw.items()
        }
        print_results_table(summary, d=d, n=N_SAMPLES, graph_type="ER",
                            metrics=METRICS)
        all_raw[d] = raw

    raw_path = _save_metrics_csv(all_raw, save_dir)
    print(f"  Raw results saved → {raw_path}")


# ===========================================================================
# Entry point
# ===========================================================================
def run_experiment3(save_dir: str = "results"):
    os.makedirs(save_dir, exist_ok=True)
    denoiser = get_denoiser()
    run_convergence(denoiser, save_dir)
    run_metrics_table(denoiser, save_dir)


if __name__ == "__main__":
    run_experiment3()
