# analysis/plot_entropy.py
# Person D — Analysis & Plotting
# Loads entropy_results.json and produces 3 publication-ready plots:
#
#   Plot 1 — Mean entropy per layer across all three conditions (main result)
#   Plot 2 — Entropy difference (delta) vs original per layer
#   Plot 3 — Per-layer box plots showing variance across sentences
#
# Run from the project root:
#   python analysis/plot_entropy.py
 
import json
import os
 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy import stats
 
 
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
 
RESULTS_PATH = "results/entropy_results.json"
OUTPUT_DIR   = "results/plots"
N_LAYERS     = 12
LAYER_LABELS = [str(i + 1) for i in range(N_LAYERS)]
 
# Condition display names and colours
CONDITIONS = {
    "original"     : {"label": "Original",          "color": "#4C72B0", "ls": "-"},
    "np_shuffled"  : {"label": "NP-shuffled",        "color": "#DD8452", "ls": "--"},
    "full_shuffled": {"label": "Full-sentence shuffle","color": "#C44E52", "ls": "-."},
}
 
# Shared plot styling
STYLE = {
    "figure.dpi"        : 150,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.grid"         : True,
    "grid.alpha"        : 0.3,
    "grid.linestyle"    : "--",
    "font.size"         : 11,
    "axes.titlesize"    : 13,
    "axes.labelsize"    : 11,
    "legend.fontsize"   : 10,
    "xtick.labelsize"   : 10,
    "ytick.labelsize"   : 10,
}
 
 
# ---------------------------------------------------------------------------
# Data loading & cleaning
# ---------------------------------------------------------------------------
 
def load_results(path=RESULTS_PATH):
    """
    Loads entropy_results.json and returns a dict:
    {
      condition_name: np.ndarray of shape (n_sentences, 12)
    }
    Rows with any None values (failed sentences) are dropped.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
 
    data = {}
    for condition, entropies in raw.items():
        arr = np.array(entropies, dtype=object)
 
        # Drop rows that contain None (failed sentences)
        mask = np.array([
            all(v is not None for v in row) for row in arr
        ])
        clean = arr[mask].astype(float)
 
        dropped = (~mask).sum()
        if dropped > 0:
            print(f"  [{condition}] Dropped {dropped} failed sentences.")
 
        data[condition] = clean
        print(f"  [{condition}] {clean.shape[0]} sentences x {clean.shape[1]} layers loaded.")
 
    return data
 
 
# ---------------------------------------------------------------------------
# Plot 1 — Mean entropy per layer (main result)
# ---------------------------------------------------------------------------
 
def plot_mean_entropy(data: dict, output_dir=OUTPUT_DIR):
    """
    Line plot of mean attention entropy per layer for each condition.
    This is your primary result figure.
 
    What to look for:
      - If np_shuffled diverges from original in early layers (1-4),
        that suggests early layers are sensitive to local NP structure.
      - If full_shuffled diverges more broadly across mid layers (3-8),
        that suggests mid layers process global syntactic order.
    """
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 4.5))
 
        for cond_key, meta in CONDITIONS.items():
            if cond_key not in data:
                continue
            arr = data[cond_key]               # (n_sentences, 12)
            means = arr.mean(axis=0)           # (12,)
            sems  = stats.sem(arr, axis=0)     # standard error of the mean
 
            x = np.arange(1, N_LAYERS + 1)
            ax.plot(x, means,
                    label=meta["label"],
                    color=meta["color"],
                    linestyle=meta["ls"],
                    linewidth=2,
                    marker="o",
                    markersize=4)
 
            # Shaded ±1 SEM band
            ax.fill_between(x,
                            means - sems,
                            means + sems,
                            color=meta["color"],
                            alpha=0.12)
 
        ax.set_xlabel("BERT layer")
        ax.set_ylabel("Mean attention entropy")
        ax.set_title("Mean attention entropy per layer — BERT-base-uncased")
        ax.set_xticks(np.arange(1, N_LAYERS + 1))
        ax.legend(frameon=False)
        ax.xaxis.set_minor_locator(ticker.NullLocator())
 
        fig.tight_layout()
        path = os.path.join(output_dir, "plot1_mean_entropy.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")
 
 
# ---------------------------------------------------------------------------
# Plot 2 — Entropy delta vs original
# ---------------------------------------------------------------------------
 
def plot_entropy_delta(data: dict, output_dir=OUTPUT_DIR):
    """
    Shows how much each corruption INCREASES entropy relative to the original,
    per layer. Positive = higher entropy (more diffuse attention) than baseline.
 
    This isolates the effect of each corruption type and makes layer-level
    differences much easier to see than Plot 1.
    """
    if "original" not in data:
        print("  [SKIP] plot_entropy_delta requires 'original' condition.")
        return
 
    original_means = data["original"].mean(axis=0)  # (12,)
 
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 4.5))
 
        x = np.arange(1, N_LAYERS + 1)
        ax.axhline(0, color="#4C72B0", linewidth=1.2, linestyle="-",
                   label="Original (baseline)", alpha=0.6)
 
        for cond_key in ["np_shuffled", "full_shuffled"]:
            if cond_key not in data:
                continue
            meta  = CONDITIONS[cond_key]
            means = data[cond_key].mean(axis=0)
            delta = means - original_means
 
            ax.plot(x, delta,
                    label=meta["label"],
                    color=meta["color"],
                    linestyle=meta["ls"],
                    linewidth=2,
                    marker="o",
                    markersize=4)
 
            # Shade area above/below baseline
            ax.fill_between(x, 0, delta,
                            color=meta["color"],
                            alpha=0.08)
 
        ax.set_xlabel("BERT layer")
        ax.set_ylabel("Δ entropy vs original")
        ax.set_title("Entropy increase relative to original — per layer")
        ax.set_xticks(np.arange(1, N_LAYERS + 1))
        ax.legend(frameon=False)
 
        fig.tight_layout()
        path = os.path.join(output_dir, "plot2_entropy_delta.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")
 
 
# ---------------------------------------------------------------------------
# Plot 3 — Box plots per layer (variance across sentences)
# ---------------------------------------------------------------------------
 
def plot_boxplots(data: dict, output_dir=OUTPUT_DIR):
    """
    For each condition, shows the distribution of per-sentence entropy
    at each layer as a box plot. Reveals whether the mean in Plot 1
    is stable across sentences or driven by outliers.
 
    Produces one subplot per condition (3 rows total).
    """
    conditions_present = [k for k in CONDITIONS if k in data]
    n = len(conditions_present)
 
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(n, 1, figsize=(10, 3.5 * n), sharex=True, sharey=True)
        if n == 1:
            axes = [axes]
 
        x = np.arange(1, N_LAYERS + 1)
 
        for ax, cond_key in zip(axes, conditions_present):
            meta = CONDITIONS[cond_key]
            arr  = data[cond_key]          # (n_sentences, 12)
 
            # Build list of arrays, one per layer
            layer_data = [arr[:, i] for i in range(N_LAYERS)]
 
            bp = ax.boxplot(layer_data,
                            positions=x,
                            widths=0.55,
                            patch_artist=True,
                            showfliers=False,   # hide extreme outliers for clarity
                            medianprops=dict(color="white", linewidth=1.5))
 
            for patch in bp["boxes"]:
                patch.set_facecolor(meta["color"])
                patch.set_alpha(0.6)
            for element in ["whiskers", "caps"]:
                for line in bp[element]:
                    line.set_color(meta["color"])
                    line.set_alpha(0.5)
 
            ax.set_title(meta["label"])
            ax.set_ylabel("Entropy")
            ax.set_xticks(x)
 
        axes[-1].set_xlabel("BERT layer")
        fig.suptitle("Per-sentence entropy distribution per layer", y=1.01, fontsize=13)
 
        fig.tight_layout()
        path = os.path.join(output_dir, "plot3_boxplots.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")
 
 
# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------
 
def run_significance_tests(data: dict):
    """
    Runs a paired t-test at each layer comparing each corruption condition
    to the original. Prints a table of p-values.
 
    A p-value < 0.05 at a given layer means the entropy difference at that
    layer is statistically significant — not just noise.
 
    Results are also saved to results/significance_tests.txt.
    """
    if "original" not in data:
        print("  [SKIP] significance tests require 'original' condition.")
        return
 
    original = data["original"]   # (n_sentences, 12)
    lines    = []
 
    header = f"{'Layer':<8}" + "".join(
        f"{CONDITIONS[c]['label']:<22}"
        for c in ["np_shuffled", "full_shuffled"] if c in data
    )
    lines.append(header)
    lines.append("-" * len(header))
 
    for layer_idx in range(N_LAYERS):
        row = f"{layer_idx + 1:<8}"
        for cond_key in ["np_shuffled", "full_shuffled"]:
            if cond_key not in data:
                continue
            corrupted = data[cond_key]
            t_stat, p_val = stats.ttest_rel(
                corrupted[:, layer_idx],
                original[:, layer_idx]
            )
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            row += f"p={p_val:.4f} {sig:<4}          "
        lines.append(row)
 
    output = "\n".join(lines)
    print("\n--- Paired t-test results (corruption vs original) ---")
    print(output)
 
    path = os.path.join(OUTPUT_DIR, "significance_tests.txt")
    with open(path, "w") as f:
        f.write("Paired t-test: corruption condition vs original\n")
        f.write("* p<0.05  ** p<0.01  *** p<0.001\n\n")
        f.write(output)
    print(f"\nSaved: {path}")
 
 
# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
 
    print("Loading results...")
    data = load_results()
 
    print("\nGenerating plots...")
    plot_mean_entropy(data)
    plot_entropy_delta(data)
    plot_boxplots(data)
 
    print("\nRunning significance tests...")
    run_significance_tests(data)
 
    print(f"\nAll outputs saved to {OUTPUT_DIR}/")