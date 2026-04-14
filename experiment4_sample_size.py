"""
Experiment 4: Effect of sample size (n/d ratio) on DAG recovery.

Sweeps n/d from data-scarce to data-rich for a fixed graph size d, and
compares NOTEARS, DAGMA, and DAG-PnP on SHD and F1.

The key hypothesis: the diffusion prior matters most when n/d is small
(data is uninformative between equivalent DAGs), and the gap between
DAG-PnP and DAGMA should widen at low n/d.

Output files (written to save_dir):
  exp4.png        — combined 2-panel figure (SHD and F1 vs. n/d)
  exp4_raw.csv    — per-seed raw results
"""

import os
import csv
import numpy as np
import torch
from collections import defaultdict
from typing import Dict

from data_utils import random_dag, sample_sem, evaluate_dag, to_tensor, scale_free_dag
from dag_pnp import dag_pnp, IdentityDenoiser, GaussianShrinkageDenoiser
from baselines import dagma_baseline, notears_baseline
from plot_utils import plot_experiment4
from dag_config import shared_kwargs, K_EDGES, WEIGHT_LOW, WEIGHT_HIGH, THRESHOLD


# ===========================================================================
# !! PLUG IN YOUR DENOISER HERE !!
# ===========================================================================
def get_denoiser():
    """
    Return a trained diffusion denoiser.  Replace with, e.g.:
        from my_denoiser import DiffusionDenoiser
        return DiffusionDenoiser.load("checkpoints/dag_diffusion.pt")
    """
    return IdentityDenoiser()  # <-- ablation: no denoising
    # return GaussianShrinkageDenoiser(prior_std=0.75)  # <-- replace with real denoiser


# ===========================================================================
# Configuration
# ===========================================================================
D_VALUES  = [20]        # graph sizes to compare
NUM_SEEDS = 10              # random seeds per n/d point

# n/d ratios to sweep (x-axis of the plot)
# Start at 2 — at n/d=1 the system X=XW+E is under-determined
ND_RATIOS = [2, 5, 10, 25, 50]
SF_M      = K_EDGES


def _save_raw_results_csv(all_raw: Dict, save_dir: str) -> str:
    """Save per-seed sample-size sweep results as a flat CSV."""
    fname = os.path.join(save_dir, "exp4_raw.csv")
    with open(fname, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["d", "method", "nd_ratio", "n", "seed", "shd", "f1"],
        )
        writer.writeheader()
        for d, raw in all_raw.items():
            for method, metric_dict in raw.items():
                for nd in ND_RATIOS:
                    shd_vals = metric_dict.get("shd", {}).get(nd, [])
                    f1_vals  = metric_dict.get("f1",  {}).get(nd, [])
                    for seed, (shd, f1) in enumerate(zip(shd_vals, f1_vals)):
                        writer.writerow({
                            "d": d, "method": method, "nd_ratio": nd,
                            "n": int(nd * d), "seed": seed, "shd": shd, "f1": f1,
                        })
    return fname


# ===========================================================================
# Main
# ===========================================================================
def run_experiment4(save_dir: str = "results"):
    os.makedirs(save_dir, exist_ok=True)
    denoiser  = get_denoiser()
    methods   = ["NOTEARS", "DAGMA", "DAG-PnP"]
    style_map = {"NOTEARS": "notears", "DAGMA": "dagma", "DAG-PnP": "dag_pnp"}

    # all_raw[d][method][metric][nd_ratio] -> list of values across seeds
    all_raw: Dict = {}

    for d in D_VALUES:
        kw  = shared_kwargs(d)
        raw = {m: {"shd": defaultdict(list), "f1": defaultdict(list)} for m in methods}
        all_raw[d] = raw

        print(f"\n{'='*60}")
        print(f"  Experiment 4: sample size sweep   d={d}   seeds={NUM_SEEDS}")
        print(f"  s={kw['s']:.2f}  W_max={kw['W_max']:.2f}")
        print(f"{'='*60}")
        print(f"  {'n/d':>6}  {'n':>6}  " +
              "  ".join(f"{m+' SHD':>12}" for m in methods))
        print("  " + "─" * (10 + 14 * len(methods)))

        for nd in ND_RATIOS:
            n         = int(nd * d)
            seed_shds = {m: [] for m in methods}

            for seed in range(NUM_SEEDS):
                rng_offset = seed * 500
                W_true = random_dag(d, k=K_EDGES, weight_low=WEIGHT_LOW,
                                    weight_high=WEIGHT_HIGH, seed=rng_offset)
                # To use SF ground truth instead, uncomment the next line and comment out above.
                # W_true = scale_free_dag(d, m=SF_M, weight_low=WEIGHT_LOW,
                #                         weight_high=WEIGHT_HIGH, seed=rng_offset)
                X_np = sample_sem(W_true, n, seed=rng_offset + 1)
                X    = to_tensor(X_np)

                # ---- NOTEARS ----
                W_n, _ = notears_baseline(
                    X, lambda_data=kw["lambda_data"], lambda_dag=kw["lambda_dag"],
                    num_iters=kw["num_iters"], lr=0.001,
                    W_max=kw["W_max"], grad_clip=10.0,
                )
                m_n = evaluate_dag(W_n, W_true, THRESHOLD)
                raw["NOTEARS"]["shd"][nd].append(m_n["shd"])
                raw["NOTEARS"]["f1"][nd].append(m_n["f1"])
                seed_shds["NOTEARS"].append(m_n["shd"])

                # ---- DAGMA ----
                W_d, _ = dagma_baseline(X, **kw)
                m_d = evaluate_dag(W_d, W_true, THRESHOLD)
                raw["DAGMA"]["shd"][nd].append(m_d["shd"])
                raw["DAGMA"]["f1"][nd].append(m_d["f1"])
                seed_shds["DAGMA"].append(m_d["shd"])

                # ---- DAG-PnP ----
                W_p, _ = dag_pnp(X, denoiser, verbose=False, **kw)
                m_p = evaluate_dag(W_p.detach().numpy(), W_true, THRESHOLD)
                raw["DAG-PnP"]["shd"][nd].append(m_p["shd"])
                raw["DAG-PnP"]["f1"][nd].append(m_p["f1"])
                seed_shds["DAG-PnP"].append(m_p["shd"])

            print(f"  {nd:>6}  {n:>6}  " + "  ".join(
                f"{np.mean(seed_shds[m]):>8.2f} ± {np.std(seed_shds[m]):<4.2f}"
                for m in methods
            ))

    d = D_VALUES[0]
    plot_results = {
        style_map[m]: {
            "shd": dict(all_raw[d][m]["shd"]),
            "f1":  dict(all_raw[d][m]["f1"]),
        }
        for m in methods
    }
    plot_experiment4(
        plot_results,
        n_over_d_values=ND_RATIOS,
        save_path=os.path.join(save_dir, "exp4.png"),
    )

    raw_path = _save_raw_results_csv(all_raw, save_dir)
    print(f"\n  Raw results saved to {raw_path}")


if __name__ == "__main__":
    run_experiment4()
