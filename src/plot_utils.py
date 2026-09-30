"""Printing and sample-size plotting used by the retained experiments."""

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.ticker as mticker

from typing import Dict, List, Optional

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
        "fit_error":       "Fit err ↓",
        "fit_error_ratio": "Fit ratio ↓",
        "f1":              "F1 ↑",
        "tpr":             "Recall ↑",
        "precision":       "Precision ↑",
        "accuracy":        "Accuracy ↑",
        "fdr":             "FDR ↓",
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

def plot_experiment4(
    results:         Dict[str, Dict],
    n_over_d_values: List[float],
    save_path:       Optional[str] = None,
):
    """
    Single-row 2-panel figure for Experiment 4.

      Panel 0 — SHD vs. n/d ratio  (lower is better)
      Panel 1 — F1  vs. n/d ratio  (higher is better)

    ±1 std shaded bands are drawn around each mean curve.

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

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))

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
                            color=s["color"], alpha=0.12)

        ax.set_xlabel(r"$n \,/\, d$  (samples per variable)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Effect of sample size on DAG recovery", fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"[plot_utils] Saved: {save_path}")
    plt.show()
