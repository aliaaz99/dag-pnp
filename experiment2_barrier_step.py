"""
Experiment 2: Barrier-aware step size vs. fixed step size — numerical results.

Evaluates four configurations defined by the cross-product of:
  (1) Step-size rule  : fixed η  vs. barrier-aware (adaptive) η
  (2) Projection step : off      vs. on  (_project_onto_K after denoiser)

For each configuration and each d in D_VALUES, runs NUM_SEEDS random seeds
and reports mean ± std of:
  SHD, F1, L_data, h(W), infeasible steps (gradient steps that left W_s)

Outputs (written to save_dir):
  exp2_results.csv      — per-seed raw values
  exp2_summary.csv      — mean ± std summary (one row per config × d)
  exp2_summary.tex      — LaTeX tabular (copy-paste ready)
"""

import os
import csv
import numpy as np
import torch
from collections import defaultdict
from typing import Dict, List

from data_utils import random_dag, sample_sem, evaluate_dag, to_tensor, scale_free_dag
from dag_pnp import dag_pnp, IdentityDenoiser, GDSSDiffusionDenoiser
from baselines import dag_pnp_fixed_step
from losses_dag import data_loss, dagma_penalty
from dag_config import shared_kwargs, K_EDGES, WEIGHT_LOW, WEIGHT_HIGH, NORMALISE, get_threshold


# ===========================================================================
# Configuration
# ===========================================================================
D_VALUES  = [20]
N_SAMPLES = 20
NUM_SEEDS = 10
SF_M      = K_EDGES

CONFIGS = [
    # (label,              step_type,  project)
    ("Fixed + No Proj",    "fixed",    False),
    ("Fixed + Proj",       "fixed",    True),
    ("Barrier + No Proj",  "barrier",  False),
    ("Barrier + Proj",     "barrier",  True),
]

METRICS = ["shd", "f1", "l_data", "h_w", "infeasible_steps"]


# ===========================================================================
# Helpers
# ===========================================================================
def _run_config(X, denoiser, step_type, project, kw):
    """
    Run one configuration and return (W_np, infeasible_count).
    infeasible_count = number of gradient steps that left W_s.
    Barrier-aware is 0 by Lemma 1.
    """
    if step_type == "fixed":
        W, hist = dag_pnp_fixed_step(
            X, denoiser,
            fixed_eta=kw["eta"],
            project=project,
            denoise_every = 10,
            **kw,
        )
        infeasible = int(sum(hist["infeasible_steps"]))
    else:  # barrier
        W, _ = dag_pnp(X, denoiser, verbose=False, project=project, denoise_every=10, **kw)
        infeasible = 0   # guaranteed by Lemma 1

    return W.detach().numpy(), infeasible


def _eval(W_np, W_true, X, s):
    """Compute all metrics for a single estimate."""
    dag_metrics = evaluate_dag(W_np, W_true, get_threshold(W_np))
    W_t = torch.tensor(W_np, dtype=X.dtype, device=X.device)
    l_data = data_loss(X, W_t).item()
    h_w    = dagma_penalty(W_t, s).item()   # inf if outside feasible set
    return {
        "shd":    dag_metrics["shd"],
        "f1":     dag_metrics["f1"],
        "l_data": l_data,
        "h_w":    h_w if np.isfinite(h_w) else float("nan"),
    }


def _write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_table(summary, d):
    """Print a formatted mean ± std table for one graph size."""
    col_w  = 18
    headers = ["SHD ↓", "F1 ↑", "L_data ↓", "h(W) ↓", "Infeas. steps ↓"]
    keys    = ["shd", "f1", "l_data", "h_w", "infeasible_steps"]
    header  = f"{'Configuration':<24}" + "".join(f"{h:>{col_w}}" for h in headers)
    sep     = "─" * len(header)

    print(f"\n{sep}")
    print(f"  d = {d}   n = {N_SAMPLES}   seeds = {NUM_SEEDS}")
    print(sep)
    print(header)
    print(sep)

    for label, _, _ in CONFIGS:
        row_str = f"{label:<24}"
        for k in keys:
            mean, std = summary[label][k]
            row_str += f"  {mean:>6.3f} ± {std:<7.3f}"
        print(row_str)

    print(sep + "\n")


def _latex_table(all_summary):
    """Return a LaTeX tabular string covering all d values."""
    keys    = ["shd", "f1", "l_data", "h_w", "infeasible_steps"]
    headers = ["SHD $\\downarrow$", "F1 $\\uparrow$",
               "$L_{\\text{data}}$ $\\downarrow$",
               "$h(W)$ $\\downarrow$",
               "Infeas. steps $\\downarrow$"]

    lines = [
        r"\begin{tabular}{ll" + "r" * len(keys) + "}",
        r"\toprule",
        r"$d$ & Configuration & " + " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for d, summary in sorted(all_summary.items()):
        lines.append(r"\multicolumn{" + str(2 + len(keys)) + r"}{l}{"
                     + f"$d = {d}$" + r"} \\")
        for label, _, _ in CONFIGS:
            cells = []
            for k in keys:
                mean, std = summary[label][k]
                if np.isnan(mean):
                    cells.append("---")
                else:
                    cells.append(f"{mean:.3f} $\\pm$ {std:.3f}")
            lines.append(f"  & {label} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


# ===========================================================================
# Main experiment
# ===========================================================================
def get_denoiser():
    return GDSSDiffusionDenoiser.from_checkpoint(
        ckpt_path="gdss/checkpoints/sem_SF_d20/Apr16-01:12:06.pth",
        device="cpu",
    )


def run_experiment2(save_dir: str = "results", normalise: bool = NORMALISE):  # False overrides NORMALISE: barrier must stay active
    """
    normalise=False (default): run on raw X so that the barrier mechanism is
    visibly active.  Set normalise=True to compare with the normalised setting
    (note: normalisation shrinks W*, keeping it away from the feasible
    boundary, so the barrier-aware and fixed-step methods become identical).
    """
    os.makedirs(save_dir, exist_ok=True)
    denoiser = get_denoiser()

    rows_raw: List[Dict] = []
    all_summary: Dict[int, Dict] = {}

    for d in D_VALUES:
        kw = shared_kwargs(d)
        s  = kw["s"]
        print(f"\n{'='*65}")
        print(f"  Experiment 2   d={d}   n={N_SAMPLES}   seeds={NUM_SEEDS}  "
              f"normalise={normalise}")
        print(f"  s={s:.2f}  W_max={kw['W_max']:.2f}  eta={kw['eta']}")
        print(f"{'='*65}")

        # raw[label][metric] = list of values across seeds
        raw: Dict[str, Dict[str, List]] = {
            label: defaultdict(list) for label, _, _ in CONFIGS
        }

        for seed in range(NUM_SEEDS):
            rng    = seed * 100
            W_true = scale_free_dag(d, m=SF_M, weight_low=WEIGHT_LOW,
                                    weight_high=WEIGHT_HIGH, seed=rng)
            X_np = sample_sem(W_true, N_SAMPLES, seed=rng + 1)
            X    = to_tensor(X_np)

            if normalise:
                from data_utils import normalise_X, denormalise_W
                X_run, denorm = normalise_X(X)
            else:
                X_run  = X
                denorm = None

            for label, step_type, project in CONFIGS:
                W_norm, n_infeas = _run_config(X_run, denoiser, step_type, project, kw)
                W_np = (denormalise_W(W_norm, denorm)
                        if normalise else W_norm)
                m = _eval(W_np, W_true, X, s)
                m["infeasible_steps"] = n_infeas

                for k_metric, v in m.items():
                    raw[label][k_metric].append(v)

                rows_raw.append({
                    "d": d, "config": label, "seed": seed,
                    **{k_metric: v for k_metric, v in m.items()},
                })

            if (seed + 1) % 2 == 0:
                print(f"  seed {seed + 1}/{NUM_SEEDS} done")

        # Summarise
        summary = {
            label: {
                metric: (
                    float(np.nanmean(vals)),
                    float(np.nanstd(vals)),
                )
                for metric, vals in mdict.items()
            }
            for label, mdict in raw.items()
        }
        all_summary[d] = summary
        _print_table(summary, d)

    # ---- Save per-seed CSV ----
    raw_path = os.path.join(save_dir, "exp2_results.csv")
    _write_csv(
        raw_path,
        ["d", "config", "seed", "shd", "f1", "l_data", "h_w", "infeasible_steps"],
        rows_raw,
    )
    print(f"  Raw results  → {raw_path}")

    # ---- Save summary CSV ----
    rows_summary = []
    for d, summary in sorted(all_summary.items()):
        for label, _, _ in CONFIGS:
            row = {"d": d, "config": label}
            for metric in METRICS:
                mean, std = summary[label][metric]
                row[f"{metric}_mean"] = round(mean, 5)
                row[f"{metric}_std"]  = round(std, 5)
            rows_summary.append(row)

    summary_cols = ["d", "config"] + [
        f"{m}_{s}" for m in METRICS for s in ("mean", "std")
    ]
    summary_path = os.path.join(save_dir, "exp2_summary.csv")
    _write_csv(summary_path, summary_cols, rows_summary)
    print(f"  Summary      → {summary_path}")

    # ---- Save LaTeX ----
    tex_path = os.path.join(save_dir, "exp2_summary.tex")
    with open(tex_path, "w") as f:
        f.write(_latex_table(all_summary))
    print(f"  LaTeX table  → {tex_path}")


if __name__ == "__main__":
    run_experiment2()
