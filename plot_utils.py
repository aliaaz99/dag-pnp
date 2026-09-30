"""
Shared plotting and printing utilities for DAG-PnP experiments.

  print_results_table   — Experiment 1: methods × metrics table
  plot_experiment2      — Experiment 2: single 3-panel figure
                           panels 1&2 = loss | h(W) for a single d (d=50)
                           panel  3   = feasibility bar chart for multiple d values
  plot_noise_schedule   — Experiment 3: single 3-panel figure
                           panels 1&2 = loss | h(W) for a single d (d=50)
                           panel  3   = sigma_k for multiple d values + reference lines
  plot_experiment4      — Experiment 4: single 2-panel figure
                           panels = SHD | F1 vs. n/d  (single d)
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


# ===========================================================================
# Experiment 2 — single 3-panel figure
# ===========================================================================

def plot_experiment2(
    histories:         Dict[str, Dict],
    infeasible_counts: Dict[str, Dict],
    d_values:          List[int],
    num_iters:         int,
    save_path:         Optional[str] = None,
):
    """
    Single-row 3-panel figure for Experiment 2.

      Panel 0 — data loss vs. iteration        (single d=50 run)
      Panel 1 — acyclicity h(W) vs. iteration  (single d=50 run)
      Panel 2 — feasibility violation bar chart (all d values with error bars)

    Parameters
    ----------
    histories         : {"barrier": hist, "fixed": hist}
                        Each hist has keys "loss", "loss_std", "h", "h_std"
                        containing arrays averaged over seeds (with inf→nan
                        handling so fixed-step divergence shows as a line break).
    infeasible_counts : {"barrier": {"mean": [...], "std": [...]},
                         "fixed":   {"mean": [...], "std": [...]}}
                        averaged over seeds, one entry per element of d_values
    d_values          : graph sizes for the feasibility bar chart x-axis
    num_iters         : total iterations (used in y-axis label of the bar chart)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax_loss, ax_h, ax_bar = axes

    for key, hist in histories.items():
        s         = STYLE.get(key, {"color": "gray", "linestyle": "-", "label": key})
        iters     = np.arange(len(hist["loss"]))
        mean_loss = np.array(hist["loss"],     dtype=float)
        std_loss  = np.array(hist["loss_std"], dtype=float)
        mean_h    = np.array(hist["h"],        dtype=float)
        std_h     = np.array(hist["h_std"],    dtype=float)

        ax_loss.plot(iters, mean_loss,
                     color=s["color"], linestyle=s["linestyle"], linewidth=2.0,
                     label=s["label"])
        ax_loss.fill_between(iters, mean_loss - std_loss, mean_loss + std_loss,
                             color=s["color"], alpha=0.15)

        ax_h.plot(iters, mean_h,
                  color=s["color"], linestyle=s["linestyle"], linewidth=2.0,
                  label=s["label"])
        ax_h.fill_between(iters, mean_h - std_h, mean_h + std_h,
                          color=s["color"], alpha=0.15)

    ax_loss.set_title("Data fidelity loss (mean ± std)")
    ax_loss.set_xlabel("Iteration")
    ax_loss.set_ylabel(r"$\mathcal{L}_{\mathrm{data}}(W)$")
    ax_loss.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    ax_loss.legend(fontsize=9)
    ax_loss.grid(True, alpha=0.3)

    ax_h.set_title("Acyclicity penalty (mean ± std)")
    ax_h.set_xlabel("Iteration")
    ax_h.set_ylabel(r"$h(W)$")
    ax_h.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    ax_h.legend(fontsize=9)
    ax_h.grid(True, alpha=0.3)

    # ---- feasibility bar chart with error bars (all d values) ----
    x     = np.arange(len(d_values))
    width = 0.35
    for i, (key, data) in enumerate(infeasible_counts.items()):
        s     = STYLE.get(key, {"color": "gray", "label": key})
        means = data["mean"]
        stds  = data["std"]
        ax_bar.bar(x + i * width, means, width,
                    capsize=5, # yerr=stds,
                   error_kw={"elinewidth": 1.5, "ecolor": "black", "capthick": 1.5},
                   label=s["label"], color=s["color"], alpha=0.85, edgecolor="white")
    ax_bar.set_xticks(x + width / 2)
    ax_bar.set_xticklabels([f"$d={d}$" for d in d_values])
    ax_bar.set_ylabel(f"Infeasible steps\n(avg / {num_iters} iters)")
    ax_bar.set_title("Feasibility violations vs. graph size")
    ax_bar.legend(fontsize=9)
    ax_bar.grid(True, alpha=0.3, axis="y")

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
    histories:       Dict[str, Dict],
    sigma_histories: Dict[int, Dict[str, Dict]],
    sigma_min:       Optional[float] = None,
    sigma_max:       Optional[float] = None,
    save_path:       Optional[str]   = None,
):
    """
    Single-row 3-panel figure for Experiment 3.

      Panel 0 — data loss vs. iteration        (single d=50, all schedule variants)
      Panel 1 — acyclicity h(W) vs. iteration  (single d=50, all schedule variants)
      Panel 2 — sigma_k vs. iteration:
                  • only the adaptive (Residual) schedule, shown for each d value
                  • sigma_min and sigma_max drawn as horizontal reference lines

    Parameters
    ----------
    histories       : {variant_label -> history_dict}  — from a single d (e.g. d=50)
                      Each history_dict must have keys "loss", "h", "sigma".
    sigma_histories : {d -> {variant_label -> history_dict}}  — for the sigma panel.
                      Only the adaptive (non-"Fixed") variant is plotted per d.
    sigma_min       : lower bound reference line drawn on the sigma panel.
    sigma_max       : upper bound reference line drawn on the sigma panel.
    """
    colors      = ["#2CA02C", "#D62728", "#4C72B0", "#E07B39"]
    d_list      = sorted(sigma_histories.keys())
    d_linestyle = {d: ls for d, ls in zip(d_list, ["-", "--", "-.", ":"])}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # ---- panels 0 & 1: loss and h(W) from single d ----
    for col, (data_key, ylabel, title) in enumerate([
        ("loss", r"$\mathcal{L}_{\mathrm{data}}(W)$", "Data fidelity loss"),
        ("h",    r"$h(W)$",                            "Acyclicity penalty"),
    ]):
        ax = axes[col]
        for idx, (label, hist) in enumerate(histories.items()):
            ax.plot(
                range(len(hist[data_key])),
                hist[data_key],
                color=colors[idx % len(colors)],
                linewidth=2.0,
                label=label,
            )
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # ---- panel 2: sigma schedule, adaptive variant only, multi-d ----
    ax_sigma = axes[2]
    for d in d_list:
        for idx, (label, hist) in enumerate(sigma_histories[d].items()):
            if "Fixed" in label:
                continue  # skip fixed; shown as reference lines instead
            ax_sigma.plot(
                range(len(hist["sigma"])),
                hist["sigma"],
                color=colors[idx % len(colors)],
                linestyle=d_linestyle[d],
                linewidth=2.0,
                label=f"{label} (d={d})",
            )

    if sigma_max is not None:
        ax_sigma.axhline(sigma_max, color="dimgray", linestyle="--", linewidth=1.5,
                         label=f"$\\sigma_{{\\max}}={sigma_max}$")
    if sigma_min is not None:
        ax_sigma.axhline(sigma_min, color="dimgray", linestyle=":", linewidth=1.5,
                         label=f"$\\sigma_{{\\min}}={sigma_min}$")

    ax_sigma.set_xlabel("Iteration")
    ax_sigma.set_ylabel(r"$\sigma_k$")
    ax_sigma.set_title("Noise level schedule")
    ax_sigma.legend(fontsize=8)
    ax_sigma.grid(True, alpha=0.3)

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
