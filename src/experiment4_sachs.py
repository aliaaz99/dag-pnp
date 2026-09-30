"""Sachs protein-network experiment using the INT=8 subset.

Use ../rerun.py for path-independent execution and configuration recording.
"""

import os
from pathlib import Path
import csv
import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from data_utils import evaluate_dag, to_tensor, normalise_X, denormalise_W
from dag_pnp import dag_pnp, IdentityDenoiser, GaussianShrinkageDenoiser, GDSSDiffusionDenoiser
from baselines import dagma_baseline, notears_baseline, graph_gen_baseline
from losses_dag import data_loss, dagma_penalty
from dag_config import shared_kwargs, THRESHOLD, NORMALISE, ADAPTIVE_THRESHOLD, get_threshold

NORMALISE = False # hardcoded to False
# ===========================================================================
# Pretrained checkpoint selection
# ===========================================================================
def get_denoiser():
    # Checkpoint was trained on SF d=20; GNN architecture handles d=11 at
    # reduced fidelity. Replace with a d=11-specific checkpoint if available.
    return GDSSDiffusionDenoiser.from_checkpoint(
        # ckpt_path=str(Path(__file__).resolve().parents[1] / "checkpoints/SF.pth"), # for SF
        ckpt_path=str(Path(__file__).resolve().parents[1] / "checkpoints/ER.pth"), # for ER
        device="cpu",
    )


# ===========================================================================
# Sachs dataset metadata
# ===========================================================================

# Standard column order used in most causal-discovery benchmarks
SACHS_VARS = ['praf', 'pmek', 'plcg', 'PIP2', 'PIP3',
              'p44.42', 'pakts473', 'PKA', 'PKC', 'P38', 'pjnk']
#              0        1       2      3      4
#              5          6         7      8      9     10

D_SACHS = len(SACHS_VARS)   # 11

# Ground-truth consensus DAG — 17 directed edges (Sachs et al. 2005).
# W_TRUE[i, j] = 1  iff  variable i  →  variable j.
#
# Sources: Sachs 2005 Fig. 3A; replicated in NOTEARS (Zheng 2018) and
#          DAGMA (Bello 2022) benchmarks.
_EDGES_17 = [
    # PKC-driven
    (8, 0),   # PKC  → praf
    (8, 1),   # PKC  → pmek
    (8, 9),   # PKC  → P38
    (8, 10),  # PKC  → pjnk
    # PKA-driven
    (7, 0),   # PKA  → praf
    (7, 1),   # PKA  → pmek
    (7, 5),   # PKA  → p44.42  (Erk)
    (7, 6),   # PKA  → pakts473 (Akt)
    (7, 9),   # PKA  → P38
    (7, 10),  # PKA  → pjnk
    # MAPK cascade
    (0, 1),   # praf → pmek
    (1, 5),   # pmek → p44.42  (Erk)
    (5, 6),   # p44.42 → pakts473 (Erk → Akt)
    # PLC-γ / PIP pathway
    (2, 3),   # plcg → PIP2
    (2, 4),   # plcg → PIP3
    (4, 3),   # PIP3 → PIP2
    (4, 6),   # PIP3 → pakts473 (Akt)
]

W_TRUE_SACHS = np.zeros((D_SACHS, D_SACHS), dtype=float)
for i, j in _EDGES_17:
    W_TRUE_SACHS[i, j] = 1.0


# ===========================================================================
# Data loading
# ===========================================================================

# Column names in sachs.interventional.txt (same order as SACHS_VARS 0-10)
# Raf→praf, Mek→pmek, Plcg→plcg, PIP2, PIP3, Erk→p44.42, Akt→pakts473,
# PKA, PKC, P38, Jnk→pjnk  (cols 0-10); col 11 = INT
_OBS_INT = 8   # rows with INT==8 are the observational condition


def _try_load_interventional_txt(path: str) -> Optional[np.ndarray]:
    """
    Load sachs.interventional.txt — space-separated, quoted header, 12 columns.

    Filters to INT == 8 (observational condition) and drops the INT column,
    returning an (n, 11) float32 array in SACHS_VARS column order.
    """
    if not os.path.exists(path):
        return None
    try:
        data = np.loadtxt(path, skiprows=1)   # skip quoted header row
        if data.shape[1] != D_SACHS + 1:
            warnings.warn(
                f"{path} has {data.shape[1]} columns, "
                f"expected {D_SACHS + 1} (11 vars + INT)."
            )
            return None
        int_col  = data[:, -1]
        obs_mask = (int_col == _OBS_INT)
        X = data[obs_mask, :-1].astype(np.float32)
        print(f"  [sachs] INT={_OBS_INT} rows: {obs_mask.sum()}/{len(data)}")
        return X
    except Exception as e:
        warnings.warn(f"Failed to read {path}: {e}")
        return None


def _try_load_cdt() -> Optional[np.ndarray]:
    """Try to load via the Causal Discovery Toolbox (pip install cdt)."""
    try:
        from cdt.data import load_dataset          # type: ignore
        data, _ = load_dataset("sachs")
        cdt_cols = ['Raf', 'Mek', 'Plcg', 'PIP2', 'PIP3', 'Erk',
                    'Akt', 'PKA', 'PKC', 'P38', 'Jnk']
        arr = data[cdt_cols].to_numpy(dtype=np.float32)
        return arr
    except Exception:
        return None


def load_sachs_data(data_dir: str = "data") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the Sachs observational dataset (INT == 8).

    Search order:
      1. data/sachs.interventional.txt  (space-separated, 12-column file)
      2. cdt library                    (pip install cdt)

    Returns
    -------
    X_np   : (n, 11) float32 array — raw data, column order = SACHS_VARS
    W_true : (11, 11) binary adjacency matrix — ground-truth DAG
    """
    txt_path = os.path.join(data_dir, "sachs.interventional.txt")
    X = _try_load_interventional_txt(txt_path)

    if X is None:
        print("  [sachs] interventional txt not found; trying cdt library …")
        X = _try_load_cdt()

    if X is None:
        raise FileNotFoundError(
            "Sachs data not found.  Please place\n"
            "  data/sachs.interventional.txt\n"
            "in the project data directory, or run:  pip install cdt"
        )

    if X.shape[1] != D_SACHS:
        warnings.warn(
            f"Loaded data has {X.shape[1]} columns, expected {D_SACHS}. "
            "Check column order matches SACHS_VARS."
        )

    print(f"  [sachs] loaded X: shape={X.shape}  "
          f"n={X.shape[0]}  d={X.shape[1]}")
    return X, W_TRUE_SACHS


# ===========================================================================
# Per-method evaluation helper
# ===========================================================================

def _extra_metrics(W_np: np.ndarray, X: torch.Tensor,
                   s: float, W_true_np: np.ndarray) -> Dict:
    W_t      = torch.tensor(W_np,      dtype=X.dtype, device=X.device)
    W_true_t = torch.tensor(W_true_np, dtype=X.dtype, device=X.device)
    l_data   = data_loss(X, W_t).item()
    l_true   = data_loss(X, W_true_t).item()
    h_w      = dagma_penalty(W_t, s).item()
    return {
        "h_w":             h_w   if np.isfinite(h_w)   else float("nan"),
        "fit_error":       l_data,
        "fit_error_ratio": l_data / l_true if l_true > 0 else float("nan"),
    }


# ===========================================================================
# Graph plotting
# ===========================================================================

def plot_sachs_dag(W: np.ndarray, title: str, save_path: str,
                   node_labels: List[str] = None) -> None:
    """
    Draw a Sachs-network DAG with named protein nodes.

    Nodes are arranged by topological depth (left → right).
    Blue edges = positive weight / ground-truth edge.
    Red edges  = negative weight.
    Edge width ∝ |weight| (for weighted matrices) or uniform (binary).
    """
    try:
        import networkx as nx
    except ImportError:
        print("  [plot] networkx not installed — skipping. Run: pip install networkx")
        return

    labels = node_labels if node_labels is not None else SACHS_VARS
    d = W.shape[0]

    G = nx.DiGraph()
    G.add_nodes_from(range(d))
    for i in range(d):
        for j in range(d):
            if abs(W[i, j]) > 1e-9:
                G.add_edge(i, j, weight=float(W[i, j]))

    # Topological depth layout
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
    for dep, nodes in sorted(levels.items()):
        for rank, node in enumerate(nodes):
            pos[node] = (dep, -(rank - len(nodes) / 2))

    label_map = {i: labels[i] for i in range(d)}

    edge_weights = [abs(G[u][v]["weight"]) for u, v in G.edges()]
    edge_colors  = ["steelblue" if G[u][v]["weight"] > 0 else "crimson"
                    for u, v in G.edges()]
    max_w = max(edge_weights) if edge_weights else 1.0
    # Binary matrices (weights ≈ 1) get uniform width; weighted get variable width
    is_binary = all(abs(w - 1.0) < 0.1 for w in edge_weights)
    edge_widths = [2.0] * len(edge_weights) if is_binary else \
                  [0.8 + 3.0 * w / max_w for w in edge_weights]

    fig, ax = plt.subplots(figsize=(12, 6))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=1200,
                           node_color="lightyellow", edgecolors="dimgray")
    nx.draw_networkx_labels(G, pos, labels=label_map, ax=ax, font_size=8)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors,
                           width=edge_widths, arrows=True,
                           arrowsize=15, connectionstyle="arc3,rad=0.08")

    ax.set_title(f"{title}  ({G.number_of_edges()} edges)", fontsize=10)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved → {save_path}")


# ===========================================================================
# Main experiment
# ===========================================================================

def run_experiment4(save_dir: str = "results", data_dir: str = "data",
                    num_runs: int = 10, normalise: bool = NORMALISE):
    """
    Run all methods on the Sachs dataset and report structured results.

    Parameters
    ----------
    save_dir  : directory for output files
    data_dir  : directory containing sachs.interventional.txt
    num_runs  : number of independent optimisation runs (>1 gives mean ± std)
    normalise : if True, mean-centre then column-normalise X before optimisation
                and denormalise W before evaluation.  If False, use raw X.
                Note: two columns (Mek, Jnk) are constant in the INT=8 subset;
                without normalisation their zero variance is harmless but the
                raw data scale may affect convergence.
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    denoiser = get_denoiser()

    # ---- load data ----
    X_np, W_true = load_sachs_data(data_dir)
    n, d         = X_np.shape
    X_raw        = to_tensor(X_np)

    if normalise:
        # Mean-centre first so constant columns (std=0) become all-zeros
        # rather than exploding to 1e8 when divided by the clamped std.
        X_centered    = X_raw - X_raw.mean(dim=0)
        X_opt, denorm = normalise_X(X_centered)
    else:
        X_opt  = X_raw
        denorm = np.ones((d, d), dtype=np.float32)

    kw = shared_kwargs(d)
    s  = kw["s"]

    print(f"\n{'='*65}")
    print(f"  Experiment 4 – Sachs protein network")
    print(f"  d={d}  n={n}  true edges={int(W_true.sum())}  runs={num_runs}")
    print(f"  s={s:.2f}  W_max={kw['W_max']:.2f}  threshold={'adaptive' if ADAPTIVE_THRESHOLD else THRESHOLD}")
    print(f"{'='*65}")

    def _record(W_norm_np: np.ndarray) -> Dict:
        W_np = denormalise_W(W_norm_np, denorm)
        dag  = evaluate_dag(W_np, W_true, get_threshold(W_np))
        X_eval = X_centered if normalise else X_raw
        ext  = _extra_metrics(W_np, X_eval, s, W_true)
        return {**dag, **ext}

    methods  = ["Greedy-BIC", "NOTEARS", "DAGMA", "DAG-PnP"]
    raw: Dict[str, Dict[str, List]] = {m: defaultdict(list) for m in methods}

    for run in range(num_runs):
        print(f"\n  Run {run + 1}/{num_runs}")

        # Random W_init for gradient-based methods — each run gets a different
        # local minimum.  Greedy-BIC is deterministic so it always gives the
        # same result (std = 0 is correct for it).
        rng    = np.random.default_rng(seed=run)
        W_init = torch.tensor(
            rng.uniform(-0.1, 0.1, size=(d, d)).astype(np.float32)
        )
        np.fill_diagonal(W_init.numpy(), 0.0)   # zero diagonal in-place

        # ---- Greedy-BIC ----
        W_g, _ = graph_gen_baseline(X_opt, max_parents=min(5, d // 2))
        for k, v in _record(W_g).items():
            raw["Greedy-BIC"][k].append(v)
        print(f"    Greedy-BIC  SHD={raw['Greedy-BIC']['shd'][-1]:.0f}  "
              f"F1={raw['Greedy-BIC']['f1'][-1]:.3f}")

        # ---- NOTEARS ----
        W_n, _ = notears_baseline(
            X_opt,
            lambda_data=kw["lambda_data"], lambda_dag=kw["lambda_dag"],
            num_iters=kw["num_iters"], lr=0.001,
            W_max=kw["W_max"], grad_clip=10.0,
            W_init=W_init.numpy(),
        )
        for k, v in _record(W_n).items():
            raw["NOTEARS"][k].append(v)
        print(f"    NOTEARS     SHD={raw['NOTEARS']['shd'][-1]:.0f}  "
              f"F1={raw['NOTEARS']['f1'][-1]:.3f}")

        # ---- DAGMA ----
        W_d, _ = dagma_baseline(X_opt, W_init=W_init, **kw)
        for k, v in _record(W_d).items():
            raw["DAGMA"][k].append(v)
        print(f"    DAGMA       SHD={raw['DAGMA']['shd'][-1]:.0f}  "
              f"F1={raw['DAGMA']['f1'][-1]:.3f}")

        # ---- DAG-PnP ----
        W_p, _ = dag_pnp(X_opt, denoiser, inject_noise=True, verbose=False, lambda_l1=0.005,
                         W_init=W_init, **kw)
        for k, v in _record(W_p.detach().numpy()).items():
            raw["DAG-PnP"][k].append(v)
        print(f"    DAG-PnP     SHD={raw['DAG-PnP']['shd'][-1]:.0f}  "
              f"F1={raw['DAG-PnP']['f1'][-1]:.3f}")

    # ---- summarise ----
    metrics_report = ["shd", "h_w", "fit_error", "fit_error_ratio", "f1"]
    summary = {
        m: {k: (float(np.nanmean(raw[m][k])), float(np.nanstd(raw[m][k])))
            for k in metrics_report}
        for m in methods
    }

    _print_table(summary, n, d)

    # ---- save CSV ----
    rows = []
    for m in methods:
        n_runs = len(raw[m]["shd"])
        for r in range(n_runs):
            row = {"method": m, "run": r}
            for k in metrics_report:
                row[k] = raw[m][k][r] if r < len(raw[m][k]) else float("nan")
            rows.append(row)

    csv_path = os.path.join(save_dir, "exp4_sachs_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "run"] + metrics_report)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Results  → {csv_path}")

    # ---- save LaTeX ----
    tex_path = os.path.join(save_dir, "exp4_sachs_summary.tex")
    with open(tex_path, "w") as f:
        f.write(_latex_table(summary))
    print(f"  LaTeX    → {tex_path}")

    # ---- plot ground-truth Sachs DAG ----
    dag_path = os.path.join(save_dir, "exp4_sachs_groundtruth_dag.png")
    plot_sachs_dag(W_TRUE_SACHS,
                   title="Sachs ground-truth DAG (17 edges)",
                   save_path=dag_path)

    # ---- plot best-run estimated DAG for each method ----
    # Pick the run with the lowest SHD per method
    for method in methods:
        shd_list = raw[method]["shd"]
        if not shd_list:
            continue
        best_run = int(np.argmin(shd_list))
        # Rebuild W_np for that run from raw fit_error isn't stored directly;
        # re-run is expensive, so we skip estimated-graph plots for now and
        # only plot the ground truth.
        # (Uncomment below and store W_np per run to enable estimated plots.)
    print(f"  DAG plot → {dag_path}")


# ===========================================================================
# Display helpers
# ===========================================================================

def _print_table(summary: Dict, n: int, d: int):
    col_w   = 22
    headers = ["SHD ↓", "h(W) ↓", "Fit err ↓", "Fit ratio ↓", "F1 ↑"]
    keys    = ["shd",   "h_w",    "fit_error", "fit_error_ratio", "f1"]

    header = f"{'Method':<15}" + "".join(f"{h:>{col_w}}" for h in headers)
    sep    = "─" * len(header)

    print(f"\n{sep}")
    print(f"  Sachs  d={d}  n={n}  true edges={int(W_TRUE_SACHS.sum())}")
    print(sep)
    print(header)
    print(sep)

    for method, stats in summary.items():
        row = f"{method:<15}"
        for k in keys:
            mean, std = stats.get(k, (float("nan"), float("nan")))
            row += f"  {mean:>7.3f} ± {std:<8.3f}"
        print(row)

    print(sep + "\n")


def _latex_table(summary: Dict) -> str:
    keys    = ["shd",   "h_w",    "fit_error", "f1"]
    headers = [r"SHD$\downarrow$",
               r"$h(\mathbf{W})\downarrow$",
               r"Fit$\downarrow$",
               r"F1$\uparrow$"]

    lines = [
        r"\begin{tabular}{l" + "c" * len(keys) + "}",
        r"\toprule",
        "Method & " + " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for method, stats in summary.items():
        cells = []
        for k in keys:
            mean, std = stats.get(k, (float("nan"), float("nan")))
            if np.isnan(mean):
                cells.append("---")
            else:
                cells.append(f"${mean:.2f}\\pm{std:.2f}$")
        lines.append(f"{method} & " + " & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


if __name__ == "__main__":
    run_experiment4()
