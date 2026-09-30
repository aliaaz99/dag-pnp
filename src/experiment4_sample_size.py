"""Sample-size experiment.

Use ../rerun.py for path-independent execution and configuration recording.
"""

import os
from pathlib import Path
import csv
import numpy as np
import torch
from collections import defaultdict
from typing import Dict

from data_utils import random_dag, sample_sem, evaluate_dag, to_tensor, scale_free_dag, normalise_X, denormalise_W
from dag_pnp import dag_pnp, IdentityDenoiser, GaussianShrinkageDenoiser, GDSSDiffusionDenoiser
from baselines import dagma_baseline, notears_baseline
from plot_utils import plot_experiment4
from dag_config import shared_kwargs, K_EDGES, WEIGHT_LOW, WEIGHT_HIGH, get_threshold


# ===========================================================================
# Pretrained checkpoint selection
# ===========================================================================
def get_denoiser():
    return GDSSDiffusionDenoiser.from_checkpoint(
        ckpt_path=str(Path(__file__).resolve().parents[1] / "checkpoints/SF.pth"),
        device="cpu",
    )


# ===========================================================================
# Configuration
# ===========================================================================
D_VALUES  = [20]        # graph sizes to compare
NUM_SEEDS = 10              # random seeds per n/d point

# n/d ratios to sweep (x-axis of the plot)
ND_RATIOS = [1, 5, 10]
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
        col = 20
        print(f"  {'n/d':>6}  {'n':>6}  " +
              "  ".join(f"{m+' SHD':>{col}}  {m+' F1':>{col}}" for m in methods))
        print("  " + "─" * (14 + (col * 2 + 4) * len(methods)))

        for nd in ND_RATIOS:
            n         = int(nd * d)
            seed_shds = {m: [] for m in methods}
            seed_f1s  = {m: [] for m in methods}

            for seed in range(NUM_SEEDS):
                rng_offset = seed * 1000
                W_true = scale_free_dag(d, m=SF_M, weight_low=WEIGHT_LOW,
                                        weight_high=WEIGHT_HIGH, seed=rng_offset)
                X_np = sample_sem(W_true, n, seed=rng_offset + 1)
                X_raw = to_tensor(X_np)
                # Normalize the low-data setting.
                if nd == 1:
                    X, denorm = normalise_X(X_raw)
                else:
                    X, denorm = X_raw, np.ones((d, d), dtype=np.float32)

                # ---- NOTEARS ----
                W_n, _ = notears_baseline(
                    X, lambda_data=kw["lambda_data"], lambda_dag=kw["lambda_dag"],
                    num_iters=kw["num_iters"], lr=kw["eta"],
                    W_max=kw["W_max"], lambda_l1=0.01,
                )
                W_n = denormalise_W(W_n, denorm)
                m_n = evaluate_dag(W_n, W_true, get_threshold(W_n))
                raw["NOTEARS"]["shd"][nd].append(m_n["shd"])
                raw["NOTEARS"]["f1"][nd].append(m_n["f1"])
                seed_shds["NOTEARS"].append(m_n["shd"])
                seed_f1s["NOTEARS"].append(m_n["f1"])

                # ---- DAGMA ----
                W_d, _ = dagma_baseline(X, **kw)
                W_d = denormalise_W(W_d, denorm)
                m_d = evaluate_dag(W_d, W_true, get_threshold(W_d))
                raw["DAGMA"]["shd"][nd].append(m_d["shd"])
                raw["DAGMA"]["f1"][nd].append(m_d["f1"])
                seed_shds["DAGMA"].append(m_d["shd"])
                seed_f1s["DAGMA"].append(m_d["f1"])

                # ---- DAG-PnP ----
                W_p, _ = dag_pnp(X, denoiser, verbose=False, denoise_every=(5 if nd == 1 else 10), **kw)
                W_p = denormalise_W(W_p.detach().numpy(), denorm)
                m_p = evaluate_dag(W_p, W_true, get_threshold(W_p))
                raw["DAG-PnP"]["shd"][nd].append(m_p["shd"])
                raw["DAG-PnP"]["f1"][nd].append(m_p["f1"])
                seed_shds["DAG-PnP"].append(m_p["shd"])
                seed_f1s["DAG-PnP"].append(m_p["f1"])

            print(f"  {nd:>6}  {n:>6}  " + "  ".join(
                f"{np.mean(seed_shds[m]):>6.2f} ± {np.std(seed_shds[m]):<5.2f}"
                f"  {np.mean(seed_f1s[m]):>6.3f} ± {np.std(seed_f1s[m]):<5.3f}"
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
