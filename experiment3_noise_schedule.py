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

from data_utils import (random_dag, sample_sem, evaluate_dag, to_tensor,
                        scale_free_dag, normalise_X, denormalise_W)
from dag_pnp import dag_pnp, IdentityDenoiser, GaussianShrinkageDenoiser, GDSSDiffusionDenoiser
from baselines import dag_pnp_fixed_sigma
from losses_dag import data_loss, dagma_penalty
from plot_utils import print_results_table, plot_noise_schedule
from dag_config import (shared_kwargs, K_EDGES, WEIGHT_LOW, WEIGHT_HIGH,
                        SIGMA_MIN, SIGMA_MAX, NORMALISE, get_threshold)


# ===========================================================================
# !! PLUG IN YOUR DENOISER HERE !!
# ===========================================================================
# def get_denoiser():
#     return IdentityDenoiser()  # <-- ablation: no denoising
#     # return GaussianShrinkageDenoiser(prior_std=1.5)  # <-- replace with real denoiser


def get_denoiser():
    # return IdentityDenoiser()
    return GDSSDiffusionDenoiser.from_checkpoint(
        ckpt_path="gdss/checkpoints/sem_SF_d20/Apr16-01:12:06.pth",
        device="cpu",   # or "cuda:0" if you have a GPU
    )

# ===========================================================================
# Configuration
# ===========================================================================
D_CONV    = 20     # graph size for convergence panels 1 & 2
D_SIGMA   = [20]  # graph sizes shown in the sigma panel (panel 3)
N_SAMPLES = 500
NUM_SEEDS = 1
SEED_CONV = 42
SF_M      = K_EDGES

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


def _run_single_d(d, denoiser, seed, n_samples, normalise=True):
    """Run all three schedule variants for a single (d, seed) and return histories."""
    kw = shared_kwargs(d)
    rng = seed
    # W_true = random_dag(d, k=K_EDGES, weight_low=WEIGHT_LOW,
    #                     weight_high=WEIGHT_HIGH, seed=rng)
    # To use SF ground truth instead, uncomment the next line and comment out above.
    W_true = scale_free_dag(d, m=SF_M, weight_low=WEIGHT_LOW,
                            weight_high=WEIGHT_HIGH, seed=rng)
    X_np = sample_sem(W_true, n_samples, seed=rng + 1)
    X    = to_tensor(X_np)

    if normalise:
        X_opt, _ = normalise_X(X)
    else:
        X_opt = X

    _, h_res = dag_pnp(X_opt, denoiser, verbose=True, **kw)
    _, h_max = dag_pnp_fixed_sigma(X_opt, denoiser, fixed_sigma=SIGMA_MAX, **kw)
    _, h_min = dag_pnp_fixed_sigma(X_opt, denoiser, fixed_sigma=SIGMA_MIN, **kw)

    return (
        X, W_true, kw,
        {
            "Residual (ours)":      h_res,
            f"Fixed σ={SIGMA_MAX}": h_max,
            f"Fixed σ={SIGMA_MIN}": h_min,
        },
    )


# ===========================================================================
# Part (a): convergence curves
# ===========================================================================
def run_convergence(denoiser, save_dir, normalise=True):
    print("\n── Part (a): convergence curves ──")

    sigma_histories: Dict[int, Dict[str, Dict]] = {}
    for d in D_SIGMA:
        kw = shared_kwargs(d)
        print(f"\n  d={d}  s={kw['s']:.2f}  W_max={kw['W_max']:.2f}")
        _, _, _, hist_d = _run_single_d(d, denoiser, seed=SEED_CONV,
                                        n_samples=N_SAMPLES, normalise=normalise)
        sigma_histories[d] = hist_d

        print(f"  {'Variant':<25}  {'Final loss':>12}  {'Final h(W)':>12}  "
              f"{'Final sigma':>12}")
        print("  " + "─" * 66)
        for label, hist in hist_d.items():
            print(f"  {label:<25}  {hist['loss'][-1]:>12.5f}  "
                  f"{hist['h'][-1]:>12.5f}  {hist['sigma'][-1]:>12.4f}")

    histories_main = sigma_histories[D_CONV]

    plot_noise_schedule(
        histories=histories_main,
        sigma_histories=sigma_histories,
        sigma_min=SIGMA_MIN,
        sigma_max=SIGMA_MAX,
        save_path=os.path.join(save_dir, "exp3_convergence.png"),
    )


# ===========================================================================
# Part (b): metrics table across seeds and graph sizes
# ===========================================================================
def run_metrics_table(denoiser, save_dir, normalise=True):
    print("\n── Part (b): final metrics across seeds ──")
    all_raw = {}

    for d in D_SIGMA:
        kw = shared_kwargs(d)
        s  = kw["s"]
        raw: Dict = {label: defaultdict(list) for label in SCHEDULE_VARIANTS}

        for seed in range(NUM_SEEDS):
            rng = seed * 1000
            # W_true = random_dag(d, k=K_EDGES, weight_low=WEIGHT_LOW,
            #                     weight_high=WEIGHT_HIGH, seed=rng)
            # To use SF ground truth instead, uncomment the next line and comment out random_dag(...) above.
            W_true = scale_free_dag(d, m=SF_M, weight_low=WEIGHT_LOW,
                                    weight_high=WEIGHT_HIGH, seed=rng)
            X_np = sample_sem(W_true, N_SAMPLES, seed=rng + 1)
            X    = to_tensor(X_np)

            if normalise:
                X_opt, denorm = normalise_X(X)
            else:
                X_opt  = X
                denorm = np.ones((d, d), dtype=np.float32)

            def _record(label, W_norm_np):
                W_np = denormalise_W(W_norm_np, denorm)
                m = evaluate_dag(W_np, W_true, get_threshold(W_np))
                m.update(_extra_metrics(W_np, X, s))
                for k, v in m.items():
                    raw[label][k].append(v)

            W_r, _ = dag_pnp(X_opt, denoiser, verbose=True, **kw)
            _record("Residual (ours)", W_r.detach().numpy())

            W_mx, _ = dag_pnp_fixed_sigma(X_opt, denoiser, fixed_sigma=SIGMA_MAX, **kw)
            _record(f"Fixed σ={SIGMA_MAX}", W_mx.detach().numpy())

            W_mn, _ = dag_pnp_fixed_sigma(X_opt, denoiser, fixed_sigma=SIGMA_MIN, **kw)
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
def run_experiment3(save_dir: str = "results", normalise: bool = NORMALISE):
    os.makedirs(save_dir, exist_ok=True)
    denoiser = get_denoiser()
    run_convergence(denoiser, save_dir, normalise=normalise)
    run_metrics_table(denoiser, save_dir, normalise=normalise)


if __name__ == "__main__":
    run_experiment3()
