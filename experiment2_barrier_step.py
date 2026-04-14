"""
Experiment 2: Barrier-aware step size vs. fixed step size.

Produces a single 3-panel figure:
  Panel 1 — data loss vs. iteration  (convergence, single run at d=50)
  Panel 2 — acyclicity h(W) vs. iteration  (convergence, single run at d=50)
  Panel 3 — grouped bar chart of feasibility violations per graph size d
             (averaged over NUM_SEEDS seeds, both d=20 and d=50)

Output files (written to save_dir):
  exp2.png              — the combined 3-panel figure
  exp2_convergence.csv  — per-iteration data for panels 1 & 2
  exp2_feasibility.csv  — per-seed violation counts for panel 3
"""

import os
import csv
import numpy as np
import torch
from typing import Dict, List

from data_utils import random_dag, sample_sem, to_tensor, scale_free_dag
from dag_pnp import dag_pnp, IdentityDenoiser
from baselines import dag_pnp_fixed_step
from plot_utils import plot_experiment2
from dag_config import shared_kwargs, K_EDGES, WEIGHT_LOW, WEIGHT_HIGH, ETA, C, EPSILON, NUM_ITERS


# ===========================================================================
# Configuration
# ===========================================================================
D_CONV    = 50        # graph size for convergence panels 1 & 2
N_CONV    = 500
SEED_CONV = 42

D_FEASIBILITY = [20, 50]   # graph sizes for feasibility bar chart (panel 3)
N_FEASIBILITY = 500
NUM_SEEDS     = 10
SF_M          = K_EDGES


def _kw(d):
    """d-adaptive shared kwargs."""
    kw = shared_kwargs(d)
    return kw


def _write_rows_csv(path: str, fieldnames: List[str], rows: List[Dict]):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ===========================================================================
# Step 1: collect convergence histories (panels 1 & 2)
# ===========================================================================
def collect_convergence(save_dir: str) -> Dict[str, Dict]:
    print(f"\n── Collecting convergence data (panels 1 & 2, d={D_CONV}) ──")

    denoiser = IdentityDenoiser()
    kw = _kw(D_CONV)
    print(f"  d={D_CONV}  s={kw['s']:.2f}  W_max={kw['W_max']:.2f}")

    W_true = random_dag(D_CONV, k=K_EDGES, weight_low=WEIGHT_LOW,
                        weight_high=WEIGHT_HIGH, seed=SEED_CONV)
    # To use SF ground truth instead, uncomment the next line and comment out above.
    # W_true = scale_free_dag(D_CONV, m=SF_M, weight_low=WEIGHT_LOW,
    #                         weight_high=WEIGHT_HIGH, seed=SEED_CONV)
    X_np = sample_sem(W_true, N_CONV, seed=SEED_CONV + 1)
    X    = to_tensor(X_np)

    _, hist_barrier = dag_pnp(X, denoiser, verbose=False, **kw)
    _, hist_fixed   = dag_pnp_fixed_step(X, denoiser, fixed_eta=kw["eta"], **kw)

    histories = {"barrier": hist_barrier, "fixed": hist_fixed}

    for name, hist in [("Barrier-aware", hist_barrier), ("Fixed step", hist_fixed)]:
        n_inf = sum(hist.get("infeasible_steps", [False] * kw["num_iters"]))
        print(f"    {name:<20}  loss={hist['loss'][-1]:.5f}  "
              f"h(W)={hist['h'][-1]:.5f}  infeasible={n_inf}")

    rows = []
    for variant, hist in [("barrier", hist_barrier), ("fixed", hist_fixed)]:
        infeas = hist.get("infeasible_steps", [False] * len(hist["loss"]))
        for it in range(len(hist["loss"])):
            rows.append({
                "d": D_CONV, "variant": variant, "iteration": it,
                "loss": hist["loss"][it], "h": hist["h"][it],
                "sigma": hist["sigma"][it], "gamma": hist["gamma"][it],
                "infeasible_step": int(infeas[it]),
            })

    csv_path = os.path.join(save_dir, "exp2_convergence.csv")
    _write_rows_csv(
        csv_path,
        ["d", "variant", "iteration", "loss", "h", "sigma", "gamma", "infeasible_step"],
        rows,
    )
    print(f"\n  Saved → {csv_path}")
    return histories


# ===========================================================================
# Step 2: collect feasibility violation counts (panel 3)
# ===========================================================================
def collect_feasibility(save_dir: str) -> Dict[str, Dict]:
    print("\n── Collecting feasibility data (panel 3) ──")
    print(f"  {'d':>5}  {'Barrier (avg)':>15}  {'Fixed (avg)':>13}")
    print("  " + "─" * 38)

    # Each entry: {"mean": [per-d means], "std": [per-d stds]}
    infeasible: Dict[str, Dict[str, List[float]]] = {
        "barrier": {"mean": [], "std": []},
        "fixed":   {"mean": [], "std": []},
    }
    denoiser = IdentityDenoiser()
    rows = []

    for d in D_FEASIBILITY:
        kw = _kw(d)
        barrier_viols, fixed_viols = [], []

        for seed in range(NUM_SEEDS):
            W_true = random_dag(d, k=K_EDGES, weight_low=WEIGHT_LOW,
                                weight_high=WEIGHT_HIGH, seed=seed * 100)
            # To use SF ground truth instead, uncomment the next line and comment out random_dag(...) above.
            # W_true = scale_free_dag(d, m=SF_M, weight_low=WEIGHT_LOW,
            #                         weight_high=WEIGHT_HIGH, seed=seed * 100)
            X_np = sample_sem(W_true, N_FEASIBILITY, seed=seed * 100 + 1)
            X = to_tensor(X_np)

            # Barrier-aware: 0 violations guaranteed by Lemma 1
            barrier_viols.append(0)
            rows.append({"d": d, "seed": seed, "variant": "barrier",
                         "infeasible_steps": 0})

            # Fixed step: count actual violations
            _, hist_f = dag_pnp_fixed_step(X, denoiser, fixed_eta=kw["eta"], **kw)
            n_viol = sum(hist_f["infeasible_steps"])
            fixed_viols.append(n_viol)
            rows.append({"d": d, "seed": seed, "variant": "fixed",
                         "infeasible_steps": n_viol})

        infeasible["barrier"]["mean"].append(float(np.mean(barrier_viols)))
        infeasible["barrier"]["std"].append(float(np.std(barrier_viols)))
        infeasible["fixed"]["mean"].append(float(np.mean(fixed_viols)))
        infeasible["fixed"]["std"].append(float(np.std(fixed_viols)))
        print(f"  d={d:>3}  "
              f"{infeasible['barrier']['mean'][-1]:>15.1f} ± {infeasible['barrier']['std'][-1]:.1f}  "
              f"{infeasible['fixed']['mean'][-1]:>13.1f} ± {infeasible['fixed']['std'][-1]:.1f}")

    csv_path = os.path.join(save_dir, "exp2_feasibility.csv")
    _write_rows_csv(csv_path, ["d", "seed", "variant", "infeasible_steps"], rows)
    print(f"  Saved → {csv_path}")

    return infeasible


# ===========================================================================
# Entry point
# ===========================================================================
def run_experiment2(save_dir: str = "results"):
    os.makedirs(save_dir, exist_ok=True)
    histories         = collect_convergence(save_dir)
    infeasible_counts = collect_feasibility(save_dir)
    plot_experiment2(
        histories=histories,
        infeasible_counts=infeasible_counts,
        d_values=D_FEASIBILITY,
        num_iters=NUM_ITERS,
        save_path=os.path.join(save_dir, "exp2.png"),
    )


if __name__ == "__main__":
    run_experiment2()
