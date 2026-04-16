"""
Diagnostic script for Experiment 1 outlier seeds.

For DAGMA (and optionally DAG-PnP) at a chosen graph size and family,
re-runs all seeds and records rich per-seed diagnostics designed to
distinguish between the three main hypotheses for high L_data:

  H1. Weight scale overshoot  — W_hat has correct support but ‖W_hat‖ >> ‖W*‖
  H2. High-variance data      — L_data(W*) is also large for those seeds
  H3. W_max clipping          — ‖W_hat‖_F ≈ W_max, projection is binding

Prints a side-by-side comparison of outlier vs normal seeds and saves a
detailed CSV.

Usage:
    python diag_exp1.py              # uses defaults below
    python diag_exp1.py --d 50 --graph SF --method dagma
    python diag_exp1.py --d 50 --graph ER --method notears
"""

import os
import csv
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from data_utils import random_dag, sample_sem, evaluate_dag, to_tensor, scale_free_dag
from dag_pnp import dag_pnp, IdentityDenoiser
from baselines import dagma_baseline, notears_baseline
from losses_dag import data_loss, dagma_penalty
from dag_config import shared_kwargs, K_EDGES, WEIGHT_LOW, WEIGHT_HIGH, THRESHOLD, ADAPTIVE_THRESHOLD, get_threshold


# ===========================================================================
# Configuration defaults (override via CLI)
# ===========================================================================
D         = 50
GRAPH     = "ER"      # "ER" or "SF"
METHOD    = "dagma"   # "dagma", "notears", or "dag_pnp"
NUM_SEEDS = 10
N_SAMPLES = 500
SAVE_DIR  = "results"
OUTLIER_MULTIPLIER = 3.0   # seed is an outlier if l_data > median * this


# ===========================================================================
# Ground-truth graph helpers
# ===========================================================================

def _get_W_true(d, graph, seed):
    """Regenerate the ground-truth W for a given (d, graph, seed)."""
    rng = seed * 1000
    if graph == "ER":
        return random_dag(d, k=K_EDGES, weight_low=WEIGHT_LOW,
                          weight_high=WEIGHT_HIGH, seed=rng)
    else:
        return scale_free_dag(d, m=K_EDGES, weight_low=WEIGHT_LOW,
                              weight_high=WEIGHT_HIGH, seed=rng)


def _graph_stats(W_true: np.ndarray, X_np: np.ndarray) -> dict:
    """
    Structural statistics of the ground-truth DAG that explain variance amplification.

    Key quantities
    --------------
    n_edges        : total directed edges
    max_in_degree  : max parents per node  (fan-in)
    max_out_degree : max children per node (fan-out)
    max_path_len   : longest directed path (depth of DAG)
    transfer_fro   : ‖(I-W)^{-1}‖_F — SEM amplification factor; high values
                     mean certain leaf nodes have very large variance
    max_col_var    : max column variance of X (the dominant column)
    col_var_ratio  : max / min column variance — spread across columns
    """
    d = W_true.shape[0]
    B = (np.abs(W_true) > 1e-9).astype(int)   # binary adjacency

    n_edges        = int(B.sum())
    max_in_degree  = int(B.sum(axis=0).max())  # column sums = in-degree
    max_out_degree = int(B.sum(axis=1).max())  # row sums    = out-degree

    # Longest directed path: BFS/DP over topological order
    topo_depth = np.zeros(d, dtype=int)
    in_deg = B.sum(axis=0).copy().astype(int)
    queue  = [i for i in range(d) if in_deg[i] == 0]
    visited = 0
    while queue:
        u = queue.pop(0)
        visited += 1
        for v in range(d):
            if B[u, v]:
                topo_depth[v] = max(topo_depth[v], topo_depth[u] + 1)
                in_deg[v] -= 1
                if in_deg[v] == 0:
                    queue.append(v)
    max_path_len = int(topo_depth.max()) if visited == d else -1

    # SEM transfer matrix ‖(I-W)^{-1}‖_F: measures total variance amplification
    try:
        transfer = np.linalg.inv(np.eye(d) - W_true)
        transfer_fro = float(np.linalg.norm(transfer, "fro"))
    except np.linalg.LinAlgError:
        transfer_fro = float("nan")

    # Column variance of the sampled data
    col_vars = np.var(X_np, axis=0)
    max_col_var   = float(col_vars.max())
    min_col_var   = float(col_vars.min())
    col_var_ratio = float(max_col_var / max(min_col_var, 1e-9))

    return {
        "g_n_edges":       n_edges,
        "g_max_in_deg":    max_in_degree,
        "g_max_out_deg":   max_out_degree,
        "g_max_path_len":  max_path_len,
        "g_transfer_fro":  transfer_fro,
        "g_max_col_var":   max_col_var,
        "g_col_var_ratio": col_var_ratio,
    }


def _plot_dag(W_true: np.ndarray, seed: int, graph: str, d: int,
              save_path: str) -> None:
    """
    Draw the ground-truth DAG for one outlier seed and save to disk.

    Nodes are arranged by topological depth (left → right).
    Edge colour encodes sign: blue = positive weight, red = negative.
    Edge width encodes magnitude.
    """
    try:
        import networkx as nx
    except ImportError:
        print("  [plot] networkx not installed — skipping DAG plot. "
              "Run: pip install networkx")
        return

    G = nx.DiGraph()
    G.add_nodes_from(range(d))
    for i in range(d):
        for j in range(d):
            if abs(W_true[i, j]) > 1e-9:
                G.add_edge(i, j, weight=float(W_true[i, j]))

    # Topological depth layout: x = depth, y = position within depth level
    try:
        topo = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        topo = list(range(d))

    depth = {}
    for node in topo:
        preds = list(G.predecessors(node))
        depth[node] = max((depth[p] for p in preds), default=-1) + 1

    levels = {}
    for node, dep in depth.items():
        levels.setdefault(dep, []).append(node)

    pos = {}
    for dep, nodes in levels.items():
        for rank, node in enumerate(nodes):
            pos[node] = (dep, -(rank - len(nodes) / 2))

    edge_weights = [abs(G[u][v]["weight"]) for u, v in G.edges()]
    edge_colors  = ["steelblue" if G[u][v]["weight"] > 0 else "crimson"
                    for u, v in G.edges()]
    max_w = max(edge_weights) if edge_weights else 1.0
    edge_widths  = [1.0 + 3.0 * w / max_w for w in edge_weights]

    fig, ax = plt.subplots(figsize=(max(10, d // 3), max(6, d // 5)))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=300,
                           node_color="lightyellow", edgecolors="dimgray")
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors,
                           width=edge_widths, arrows=True,
                           arrowsize=12, connectionstyle="arc3,rad=0.1")

    n_edges = G.number_of_edges()
    ax.set_title(
        f"Ground-truth DAG — seed {seed}  ({graph}{K_EDGES}, d={d}, "
        f"{n_edges} edges)\n"
        f"blue = positive weight  |  red = negative weight  |  "
        f"width ∝ |weight|",
        fontsize=9,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved → {save_path}")


# ===========================================================================
# Per-seed diagnostics
# ===========================================================================
def _run_seed(d, n, graph, method, seed, kw):
    rng = seed * 1000
    if graph == "ER":
        W_true = random_dag(d, k=K_EDGES, weight_low=WEIGHT_LOW,
                            weight_high=WEIGHT_HIGH, seed=rng)
    else:
        W_true = scale_free_dag(d, m=K_EDGES, weight_low=WEIGHT_LOW,
                                weight_high=WEIGHT_HIGH, seed=rng)

    X_np = sample_sem(W_true, n, seed=rng + 1)
    X    = to_tensor(X_np)
    # X_norm = X / X.std(axis=0)
    # X = X_norm  # use normalised data for both methods to remove scale differences

    if method == "dagma":
        W_np, hist = dagma_baseline(X, **kw)
    elif method == "notears":
        W_np, hist = notears_baseline(
            X,
            lambda_data=kw["lambda_data"], lambda_dag=kw["lambda_dag"],
            num_iters=kw["num_iters"], lr=0.001,
            W_max=kw["W_max"], grad_clip=10.0,
        )
    else:
        W_t, hist = dag_pnp(X, IdentityDenoiser(), verbose=False, **kw)
        W_np = W_t.detach().numpy()

    # ---- structural metrics ----
    thr   = get_threshold(W_np)
    dag_m = evaluate_dag(W_np, W_true, thr)
    W_t   = torch.tensor(W_np, dtype=X.dtype)

    # ---- L_data diagnostics ----
    l_data_hat  = data_loss(X, W_t).item()
    W_true_t    = torch.tensor(W_true, dtype=X.dtype)
    l_data_true = data_loss(X, W_true_t).item()   # oracle lower bound
    l_data_zero = data_loss(X, torch.zeros_like(W_t)).item()  # trivial baseline

    # ---- weight scale diagnostics ----
    W_fro       = float(np.linalg.norm(W_np, "fro"))
    W_true_fro  = float(np.linalg.norm(W_true, "fro"))
    W_max_entry = float(np.abs(W_np).max())
    W_true_max  = float(np.abs(W_true).max())

    # ---- projection / clipping check ----
    W_max_bound = kw["W_max"]
    fro_ratio   = W_fro / W_true_fro if W_true_fro > 0 else float("nan")
    hitting_max = W_fro >= 0.95 * W_max_bound   # within 5 % of the Frobenius cap

    # ---- weight error on shared support ----
    bin_hat  = (np.abs(W_np)    > thr).astype(float)
    bin_true = (np.abs(W_true)  > thr).astype(float)
    shared   = (bin_hat * bin_true).astype(bool)
    if shared.any():
        weight_err_shared = float(np.mean(np.abs(W_np[shared] - W_true[shared])))
        weight_scale_shared = float(
            np.mean(np.abs(W_np[shared]) / (np.abs(W_true[shared]) + 1e-9))
        )
    else:
        weight_err_shared   = float("nan")
        weight_scale_shared = float("nan")

    # ---- acyclicity ----
    h_W = dagma_penalty(W_t, kw["s"]).item()
    if not np.isfinite(h_W):
        h_W = float("nan")

    # ---- data variance ----
    X_var = float(np.var(X_np))   # mean per-entry variance of X

    # ---- graph structural statistics ----
    gstats = _graph_stats(W_true, X_np)

    return {
        "seed":              seed,
        # structural recovery
        "shd":               dag_m["shd"],
        "f1":                dag_m["f1"],
        "tpr":               dag_m["tpr"],
        "fdr":               dag_m["fdr"],
        # fit quality
        "l_data_hat":        l_data_hat,
        "l_data_true":       l_data_true,
        "l_data_zero":       l_data_zero,
        "l_data_excess":     l_data_hat - l_data_true,
        "l_data_ratio":      l_data_hat / l_data_true if l_data_true > 0 else float("nan"),
        # weight scale
        "W_fro":             W_fro,
        "W_true_fro":        W_true_fro,
        "W_fro_ratio":       fro_ratio,
        "W_max_entry":       W_max_entry,
        "W_true_max":        W_true_max,
        "hitting_W_max":     hitting_max,
        # shared-support weight quality
        "weight_err_shared":   weight_err_shared,
        "weight_scale_shared": weight_scale_shared,
        # acyclicity
        "h_W":               h_W,
        # data scale
        "X_var":             X_var,
        # ground-truth graph structure
        **gstats,
    }


# ===========================================================================
# Comparison table printer
# ===========================================================================
def _print_comparison(normal, outliers):
    keys = [
        # ── graph structure ──────────────────────────────────────────
        ("g_n_edges",         "# edges in W_true"),
        ("g_max_in_deg",      "Max in-degree (parents)"),
        ("g_max_out_deg",     "Max out-degree (children)"),
        ("g_max_path_len",    "Longest causal path"),
        ("g_transfer_fro",    "‖(I-W)^{-1}‖_F  [amplification]"),
        ("g_max_col_var",     "Max column Var(X)"),
        ("g_col_var_ratio",   "Max/min column Var(X) ratio"),
        # ── fit quality ──────────────────────────────────────────────
        ("l_data_hat",        "L_data(W_hat)"),
        ("l_data_true",       "L_data(W*)  [oracle]"),
        ("l_data_ratio",      "L_data ratio hat/oracle"),
        ("l_data_excess",     "L_data excess hat-oracle"),
        ("X_var",             "Var(X)  [mean data scale]"),
        # ── recovery ─────────────────────────────────────────────────
        ("shd",               "SHD"),
        ("f1",                "F1"),
        # ── weight scale ─────────────────────────────────────────────
        ("W_fro",             "‖W_hat‖_F"),
        ("W_true_fro",        "‖W_true‖_F"),
        ("W_fro_ratio",       "‖W_hat‖_F / ‖W_true‖_F"),
        ("W_max_entry",       "max|W_hat_ij|"),
        ("weight_err_shared", "Mean |w_hat - w*| on true edges"),
        ("weight_scale_shared","Mean |w_hat|/|w*| on true edges"),
        ("h_W",               "h(W)  [acyclicity]"),
        ("hitting_W_max",     "Hitting W_max cap?"),
    ]

    n_col = 25
    print(f"\n{'─'*70}")
    print(f"{'Metric':<36}  {'Normal (mean)':>{n_col}}  {'Outlier (mean)':>{n_col}}")
    print(f"{'─'*70}")

    for k, label in keys:
        n_vals = [r[k] for r in normal  if np.isfinite(float(r[k])) if r[k] is not True and r[k] is not False]
        o_vals = [r[k] for r in outliers if np.isfinite(float(r[k])) if r[k] is not True and r[k] is not False]
        # handle booleans separately
        if k == "hitting_W_max":
            n_str = f"{sum(r[k] for r in normal)}/{len(normal)} seeds"
            o_str = f"{sum(r[k] for r in outliers)}/{len(outliers)} seeds"
        else:
            n_str = f"{np.mean(n_vals):>.4f}" if n_vals else "—"
            o_str = f"{np.mean(o_vals):>.4f}" if o_vals else "—"
        print(f"{label:<36}  {n_str:>{n_col}}  {o_str:>{n_col}}")

    print(f"{'─'*70}\n")


# ===========================================================================
# Per-seed detail table
# ===========================================================================
def _print_seed_detail(records, outlier_seeds):
    print(f"\n{'Seed':>5}  {'L_data':>9}  {'Ratio':>6}  "
          f"{'SHD':>4}  {'F1':>5}  "
          f"{'Edges':>5}  {'MaxIn':>5}  {'MaxOut':>6}  {'Path':>4}  "
          f"{'‖(I-W)⁻¹‖':>10}  {'MaxColVar':>9}  {'VarRatio':>8}  "
          f"{'Flag'}")
    print("─" * 115)
    for r in records:
        flag = "  ← OUTLIER" if r["seed"] in outlier_seeds else ""
        print(
            f"{r['seed']:>5}  "
            f"{r['l_data_hat']:>9.2f}  "
            f"{r['l_data_ratio']:>6.2f}  "
            f"{r['shd']:>4}  "
            f"{r['f1']:>5.3f}  "
            f"{r['g_n_edges']:>5}  "
            f"{r['g_max_in_deg']:>5}  "
            f"{r['g_max_out_deg']:>6}  "
            f"{r['g_max_path_len']:>4}  "
            f"{r['g_transfer_fro']:>10.2f}  "
            f"{r['g_max_col_var']:>9.2f}  "
            f"{r['g_col_var_ratio']:>8.2f}  "
            f"{flag}"
        )
    print()


# ===========================================================================
# Entry point
# ===========================================================================
def run_diagnostic(d=D, graph=GRAPH, method=METHOD, n=N_SAMPLES,
                   num_seeds=NUM_SEEDS, save_dir=SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)
    kw = shared_kwargs(d)

    print(f"\n{'='*70}")
    print(f"  Diagnostic: method={method}  graph={graph}  d={d}  n={n}")
    print(f"  s={kw['s']:.2f}  W_max={kw['W_max']:.2f}  "
          f"threshold={'adaptive' if ADAPTIVE_THRESHOLD else THRESHOLD}")
    print(f"{'='*70}")

    records = []
    for seed in range(num_seeds):
        r = _run_seed(d, n, graph, method, seed, kw)
        records.append(r)
        print(f"  seed {seed:>2d}  L_data={r['l_data_hat']:8.3f}  "
              f"L_data*={r['l_data_true']:6.3f}  ratio={r['l_data_ratio']:5.2f}  "
              f"SHD={r['shd']:3d}  F1={r['f1']:.3f}  "
              f"‖W‖_F={r['W_fro']:.2f}  scale={r['W_fro_ratio']:.2f}")

    # ---- identify outliers ----
    l_data_vals  = np.array([r["l_data_hat"] for r in records])
    median_l     = float(np.median(l_data_vals))
    outlier_mask = l_data_vals > median_l * OUTLIER_MULTIPLIER
    outlier_seeds = {r["seed"] for r, m in zip(records, outlier_mask) if m}
    normal_seeds  = {r["seed"] for r, m in zip(records, outlier_mask) if not m}

    print(f"\n  Median L_data = {median_l:.3f}  |  "
          f"outlier threshold = {median_l * OUTLIER_MULTIPLIER:.3f}")
    print(f"  Outlier seeds : {sorted(outlier_seeds)}")
    print(f"  Normal  seeds : {sorted(normal_seeds)}")

    normal   = [r for r in records if r["seed"] in normal_seeds]
    outliers = [r for r in records if r["seed"] in outlier_seeds]

    _print_seed_detail(records, outlier_seeds)
    _print_comparison(normal, outliers)

    # ---- plot ground-truth DAG for each outlier seed ----
    for seed in sorted(outlier_seeds):
        W_true_seed = _get_W_true(d, graph, seed)
        plot_path = os.path.join(
            save_dir, f"diag_dag_{graph}_d{d}_seed{seed}.png"
        )
        _plot_dag(W_true_seed, seed, graph, d, plot_path)

    # ---- diagnosis summary ----
    print("  ── Hypothesis check ──")
    if outliers:
        o_ratio = np.mean([r["l_data_ratio"] for r in outliers if np.isfinite(r["l_data_ratio"])])
        n_ratio = np.mean([r["l_data_ratio"] for r in normal  if np.isfinite(r["l_data_ratio"])])
        scale_o = np.mean([r["W_fro_ratio"]  for r in outliers if np.isfinite(r["W_fro_ratio"])])
        scale_n = np.mean([r["W_fro_ratio"]  for r in normal  if np.isfinite(r["W_fro_ratio"])])
        xvar_o  = np.mean([r["X_var"]        for r in outliers])
        xvar_n  = np.mean([r["X_var"]        for r in normal])
        clip_o  = sum(r["hitting_W_max"] for r in outliers)

        print(f"  H1 Weight scale overshoot : "
              f"‖W_hat‖/‖W*‖ outlier={scale_o:.2f} vs normal={scale_n:.2f}  "
              f"({'LIKELY' if abs(scale_o - scale_n) > 0.3 else 'unlikely'})")
        print(f"  H2 High-variance data     : "
              f"Var(X) outlier={xvar_o:.2f} vs normal={xvar_n:.2f}  "
              f"L_data ratio outlier={o_ratio:.2f} vs normal={n_ratio:.2f}  "
              f"({'LIKELY' if xvar_o > 2 * xvar_n else 'unlikely'})")
        print(f"  H3 W_max clipping         : "
              f"{clip_o}/{len(outliers)} outlier seeds hit W_max  "
              f"({'LIKELY' if clip_o > 0 else 'unlikely'})")

    # ---- additional: scale-normalised metric and convergence diagnostics ----
    print("\n  ── Scale-normalised L_data (L_data / Var(X)) ──")
    print(f"  {'Seed':>5}  {'L_data':>10}  {'Var(X)':>10}  "
          f"{'L_data/Var(X)':>15}  {'L_data*/Var(X)':>16}  flag")
    print("  " + "─" * 65)
    for r in records:
        flag = "← OUTLIER" if r["seed"] in outlier_seeds else ""
        print(f"  {r['seed']:>5}  {r['l_data_hat']:>10.3f}  {r['X_var']:>10.3f}  "
              f"  {r['l_data_hat']/r['X_var']:>13.4f}  "
              f"  {r['l_data_true']/r['X_var']:>14.4f}  {flag}")

    if outliers and normal:
        norm_hat_o = np.mean([r["l_data_hat"]  / r["X_var"] for r in outliers])
        norm_hat_n = np.mean([r["l_data_hat"]  / r["X_var"] for r in normal])
        norm_tru_o = np.mean([r["l_data_true"] / r["X_var"] for r in outliers])
        norm_tru_n = np.mean([r["l_data_true"] / r["X_var"] for r in normal])
        print(f"\n  L_data/Var(X)  outlier={norm_hat_o:.4f}  normal={norm_hat_n:.4f}")
        print(f"  L_data*/Var(X) outlier={norm_tru_o:.4f}  normal={norm_tru_n:.4f}")
        print(f"  → If the ratio L_data/Var(X) is still high for outliers after "
              f"normalising,\n    it confirms that the optimiser fails on high-variance "
              f"instances,\n    not just that L_data is measured on a larger scale.")

    print("\n  ── Recommended actions ──")
    print("  1. Normalise X before running DAGMA:  X_norm = X / X.std(axis=0)")
    print("     This removes the scale-amplification effect and makes L_data comparable")
    print("     across seeds regardless of the graph's spectral structure.")
    print("  2. Flag seeds with Var(X) > threshold as 'high-variance instances'")
    print("     and report metrics separately or with a normalised L_data.")
    print("  3. Use L_data / L_data(W*) as the reported fit metric instead of raw")
    print("     L_data — this normalises away data-scale differences.")
    print("  4. Check convergence: re-run outlier seeds with verbose=True and")
    print("     inspect whether h(W) and L_data have plateaued by iteration 1000.")
    print("  5. For outlier seeds, try reducing eta (e.g. eta/3) to check whether")
    print("     smaller steps recover the good-fit regime.")

    # ---- save CSV ----
    csv_path = os.path.join(save_dir, f"diag_{method}_{graph}_d{d}.csv")
    fieldnames = list(records[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({k: (int(v) if isinstance(v, bool) else v)
                             for k, v in r.items()})
    print(f"\n  Full diagnostics saved → {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d",      type=int,   default=D)
    parser.add_argument("--graph",  type=str,   default=GRAPH)
    parser.add_argument("--method", type=str,   default=METHOD)
    parser.add_argument("--n",      type=int,   default=N_SAMPLES)
    parser.add_argument("--seeds",  type=int,   default=NUM_SEEDS)
    args = parser.parse_args()

    run_diagnostic(
        d=args.d, graph=args.graph, method=args.method,
        n=args.n, num_seeds=args.seeds,
    )
