"""
Shared plotting and printing utilities for DAG-PnP experiments.

  print_results_table   — Experiment 1: methods × metrics table
  plot_experiment2      — Experiment 2: single 3-panel figure
                           (loss | h(W) vs. iteration, feasibility bar chart)
  plot_noise_schedule   — Experiment 3: 3-panel noise schedule comparison
                           (loss | h(W) | sigma_k vs. iteration)
  plot_experiment4      — Experiment 4: single 2-panel figure
                           (SHD | F1 vs. n/d ratio)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Consistent visual style for all experiments
# ---------------------------------------------------------------------------
STYLE = {
    "notears":  {"color": "#4C72B0", "linestyle": "--", "marker": "s",
                 "label": "NOTEARS"},
    "dagma":    {"color": "#E07B39", "linestyle": "-",  "marker": "o",
                 "label": "DAGMA (no prior)"},
    "dag_pnp":  {"color": "#2CA02C", "linestyle": "-",  "marker": "^",
                 "label": "DAG-PnP (ours)"},
    "barrier":  {"color": "#2CA02C", "linestyle": "-",  "marker": None,
                 "label": "Barrier-aware (ours)"},
    "fixed":    {"color": "#D62728", "linestyle": "--", "marker": None,
                 "label": "Fixed step"},
}

plt.rcParams.update({
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  10,
    "figure.dpi":       130,
})


# ===========================================================================
# Experiment 1 — results table
# ===========================================================================

def print_results_table(
    results:     Dict[str, Dict[str, tuple]],
    d:           int,
    n:           int,
    graph_type:  str        = "ER",
    metrics:     List[str]  = ("shd", "acyclicity", "fit_error", "f1"),
):
    """
    Pretty-print a mean ± std results table matching the paper's Table 1.

    Columns align with the paper plan:
      shd          — Structural Hamming Distance   (lower is better)
      acyclicity   — final DAGMA h(W) value        (lower is better; 0 = exact DAG)
      fit_error    — final data loss L_data(W)     (lower is better)
      f1           — F1 score on binary edges      (higher is better)

    Parameters
    ----------
    results    : method_name -> {metric -> (mean, std)}
    d, n       : graph size and number of samples (for header)
    graph_type : "ER" or "SF" (Erdos-Renyi / scale-free)
    metrics    : column order (subset of the keys above)
    """
    col_w  = 18
    labels = {
        "shd":        "SHD ↓",
        "acyclicity": "h(W) ↓",
        "fit_error":  "Fit err ↓",
        "f1":         "F1 ↑",
        "tpr":        "TPR ↑",
        "fdr":        "FDR ↓",
    }
    header = f"{'Method':<22}" + "".join(
        f"{labels.get(m, m.upper()):>{col_w}}" for m in metrics
    )
    sep = "─" * len(header)

    print(f"\n{sep}")
    print(f"  {graph_type} graph   d = {d}   n = {n}")
    print(sep)
    print(header)
    print(sep)

    for method, stats in results.items():
        row = f"{method:<22}"
        for m in metrics:
            mean, std = stats.get(m, (float("nan"), float("nan")))
            row += f"  {mean:>6.3f} ± {std:<6.3f}"
        print(row)

    print(sep + "\n")


# ===========================================================================
# Experiment 2 — single 3-panel figure
# ===========================================================================

def plot_experiment2(
    histories:         Dict[str, Dict],
    infeasible_counts: Dict[str, List[int]],
    d_values:          List[int],
    num_iters:         int,
    save_path:         Optional[str] = None,
):
    """
    Single 3-panel figure for Experiment 2.

      Panel 1 — data loss vs. iteration  (barrier vs. fixed)
      Panel 2 — acyclicity h(W) vs. iteration  (barrier vs. fixed)
      Panel 3 — grouped bar chart of feasibility violations per graph size d

    Parameters
    ----------
    histories         : {"barrier": history_dict, "fixed": history_dict}
                        Each dict must have keys "loss" and "h".
    infeasible_counts : {"barrier": [avg_count_per_d], "fixed": [avg_count_per_d]}
    d_values          : graph sizes corresponding to infeasible_counts entries
    num_iters         : total iterations (used in the y-axis label of panel 3)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # ---- Panels 1 & 2: convergence curves ----
    for key, hist in histories.items():
        s     = STYLE.get(key, {"color": "gray", "linestyle": "-", "label": key})
        iters = range(len(hist["loss"]))
        kw    = dict(color=s["color"], linestyle=s["linestyle"],
                     label=s["label"], linewidth=2.0)
        axes[0].plot(iters, hist["loss"], **kw)
        axes[1].plot(iters, hist["h"],    **kw)

    axes[0].set_title("Data fidelity loss")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel(r"$\mathcal{L}_{\mathrm{data}}(W)$")
    axes[0].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

    axes[1].set_title("DAGMA acyclicity penalty")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel(r"$h(W)$")
    axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

    for ax in axes[:2]:
        ax.legend()
        ax.grid(True, alpha=0.3)

    # ---- Panel 3: feasibility violations bar chart ----
    ax3   = axes[2]
    x     = np.arange(len(d_values))
    width = 0.35

    for i, (key, counts) in enumerate(infeasible_counts.items()):
        s = STYLE.get(key, {"color": "gray", "label": key})
        ax3.bar(x + i * width, counts, width,
                label=s["label"], color=s["color"], alpha=0.85, edgecolor="white")

    ax3.set_xticks(x + width / 2)
    ax3.set_xticklabels([f"$d={d}$" for d in d_values])
    ax3.set_ylabel(f"Infeasible steps\n(avg / {num_iters} iters)")
    ax3.set_title("Feasibility violations vs. graph size")
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Barrier-aware vs. fixed step size", fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"[plot_utils] Saved: {save_path}")
    plt.show()


# ===========================================================================
# Experiment 3 — noise schedule comparison
# ===========================================================================

def plot_noise_schedule(
    histories:  Dict[str, Dict],
    save_path:  Optional[str] = None,
):
    """
    Three-panel figure comparing noise schedule variants:
      left  — data loss vs. iteration
      mid   — acyclicity h(W) vs. iteration
      right — sigma_k vs. iteration (shows schedule dynamics)

    Parameters
    ----------
    histories : {variant_label -> history_dict}
                variant_label examples: "residual (ours)", "fixed sigma_max", "fixed sigma_min"
                Each history_dict must have keys "loss", "h", "sigma".
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    colors    = ["#2CA02C", "#D62728", "#4C72B0", "#E07B39"]
    linestyles = ["-", "--", "-.", ":"]

    for idx, (label, hist) in enumerate(histories.items()):
        c  = colors[idx % len(colors)]
        ls = linestyles[idx % len(linestyles)]
        iters = range(len(hist["loss"]))
        kw = dict(color=c, linestyle=ls, label=label, linewidth=2.0)
        axes[0].plot(iters, hist["loss"],  **kw)
        axes[1].plot(iters, hist["h"],     **kw)
        axes[2].plot(iters, hist["sigma"], **kw)

    panel_cfg = [
        (r"$\mathcal{L}_{\mathrm{data}}(W)$", "Data fidelity loss"),
        (r"$h(W)$",                            "Acyclicity penalty"),
        (r"$\sigma_k$",                        "Noise level schedule"),
    ]
    for ax, (ylabel, title) in zip(axes, panel_cfg):
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Residual-based vs. fixed noise schedule", fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"[plot_utils] Saved: {save_path}")
    plt.show()


# ===========================================================================
# Experiment 4 — single 2-panel figure (SHD and F1 vs n/d)
# ===========================================================================

def plot_experiment4(
    results:         Dict[str, Dict],
    n_over_d_values: List[float],
    save_path:       Optional[str] = None,
):
    """
    Single 2-panel figure for Experiment 4.

      Panel 1 — SHD vs. n/d ratio  (lower is better)
      Panel 2 — F1  vs. n/d ratio  (higher is better)

    Both panels share the same x-axis and legend, with ±1 std shaded bands.

    Parameters
    ----------
    results         : {style_key -> {"shd": {nd -> [values]},
                                     "f1":  {nd -> [values]}}}
                      style_key must be one of STYLE keys
                      ("notears", "dagma", "dag_pnp").
    n_over_d_values : x-axis values (list of n/d ratios).
    """
    metric_cfg = [
        ("shd", r"SHD  $\downarrow$",  "SHD (lower is better)"),
        ("f1",  r"F1   $\uparrow$",    "F1 (higher is better)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, (metric, ylabel, title) in zip(axes, metric_cfg):
        for key, metric_dict in results.items():
            s      = STYLE.get(key, {"color": "gray", "linestyle": "-",
                                     "marker": "o", "label": key})
            nd_res = metric_dict.get(metric, {})
            means  = np.array([np.mean(nd_res[nd]) for nd in n_over_d_values])
            stds   = np.array([np.std(nd_res[nd])  for nd in n_over_d_values])

            ax.plot(n_over_d_values, means,
                    color=s["color"], linestyle=s["linestyle"],
                    marker=s.get("marker", "o"), markersize=6,
                    label=s["label"], linewidth=2.0)
            ax.fill_between(n_over_d_values,
                            means - stds, means + stds,
                            color=s["color"], alpha=0.15)

        ax.set_xlabel(r"$n \,/\, d$  (samples per variable)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Effect of sample size on DAG recovery", fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"[plot_utils] Saved: {save_path}")
    plt.show()
