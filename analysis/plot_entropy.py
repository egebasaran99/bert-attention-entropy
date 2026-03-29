# analysis/plot_entropy.py
# Person D — Analysis & Plotting
# Loads entropy results JSON and produces:
#   Plot 1 — Mean entropy per layer across all three conditions
#   Plot 2 — Entropy difference (delta) vs original per layer
#   Plot 3 — Per-layer box plots showing variance across sentences
#   Plot 4 — Heatmap of delta vs original
#   significance_tests_vs_original.txt — paired t-tests per layer vs original
#   direct_comparison_tests.txt       — paired t-tests NP-shuffled vs full-shuffle
#   effect_sizes.csv                  — per-layer mean deltas + paired Cohen's d
#   layer_group_summary.csv           — early/middle/late summary table
#
# Example:
#   python analysis/plot_entropy.py \
#       --results_path results/entropy_results_sst2_2000.json \
#       --output_dir results/plots_sst2_2000

import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_RESULTS_PATH = "results/entropy_results.json"
DEFAULT_OUTPUT_DIR = "results/plots"
N_LAYERS = 12

CONDITIONS = {
    "original": {
        "label": "Original",
        "color": "#4C72B0",
        "ls": "-"
    },
    "np_shuffled": {
        "label": "NP-shuffled",
        "color": "#DD8452",
        "ls": "--"
    },
    "full_shuffled": {
        "label": "Full-sentence shuffle",
        "color": "#C44E52",
        "ls": "-."
    },
}

STYLE = {
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def layer_group(layer_idx_1based: int) -> str:
    if 1 <= layer_idx_1based <= 4:
        return "early"
    if 5 <= layer_idx_1based <= 8:
        return "middle"
    return "late"


def paired_cohens_d(x, y):
    """
    Cohen's d for paired samples:
        mean(diff) / std(diff)
    """
    diff = np.array(x) - np.array(y)
    sd = diff.std(ddof=1)
    if sd == 0:
        return 0.0
    return diff.mean() / sd


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(path):
    """
    Loads entropy results JSON and returns:
    {
      condition_name: np.ndarray of shape (n_sentences, 12)
    }
    Rows with any None values are dropped.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    data = {}
    for condition, entropies in raw.items():
        arr = np.array(entropies, dtype=object)

        mask = np.array([all(v is not None for v in row) for row in arr])
        clean = arr[mask].astype(float)

        dropped = (~mask).sum()
        if dropped > 0:
            print(f"  [{condition}] Dropped {dropped} failed sentences.")

        data[condition] = clean
        print(f"  [{condition}] {clean.shape[0]} sentences x {clean.shape[1]} layers loaded.")

    return data


def to_long_dataframe(data):
    rows = []
    for condition, arr in data.items():
        for sentence_id in range(arr.shape[0]):
            for layer_idx in range(arr.shape[1]):
                rows.append({
                    "sentence_id": sentence_id,
                    "condition": condition,
                    "layer": layer_idx + 1,
                    "layer_group": layer_group(layer_idx + 1),
                    "entropy": arr[sentence_id, layer_idx],
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot 1 — Mean entropy per layer
# ---------------------------------------------------------------------------

def plot_mean_entropy(data, output_dir):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 4.5))

        for cond_key, meta in CONDITIONS.items():
            if cond_key not in data:
                continue

            arr = data[cond_key]
            means = arr.mean(axis=0)
            sems = stats.sem(arr, axis=0)

            x = np.arange(1, N_LAYERS + 1)

            ax.plot(
                x, means,
                label=meta["label"],
                color=meta["color"],
                linestyle=meta["ls"],
                linewidth=2,
                marker="o",
                markersize=4,
            )

            ax.fill_between(
                x,
                means - sems,
                means + sems,
                color=meta["color"],
                alpha=0.12,
            )

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
# Plot 2 — Delta vs original
# ---------------------------------------------------------------------------

def plot_entropy_delta(data, output_dir):
    if "original" not in data:
        print("  [SKIP] plot_entropy_delta requires 'original' condition.")
        return

    original_means = data["original"].mean(axis=0)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 4.5))

        x = np.arange(1, N_LAYERS + 1)
        ax.axhline(
            0,
            color="#4C72B0",
            linewidth=1.2,
            linestyle="-",
            label="Original (baseline)",
            alpha=0.6,
        )

        for cond_key in ["np_shuffled", "full_shuffled"]:
            if cond_key not in data:
                continue

            meta = CONDITIONS[cond_key]
            means = data[cond_key].mean(axis=0)
            delta = means - original_means

            ax.plot(
                x, delta,
                label=meta["label"],
                color=meta["color"],
                linestyle=meta["ls"],
                linewidth=2,
                marker="o",
                markersize=4,
            )

            ax.fill_between(
                x, 0, delta,
                color=meta["color"],
                alpha=0.08,
            )

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
# Plot 3 — Boxplots
# ---------------------------------------------------------------------------

def plot_boxplots(data, output_dir):
    conditions_present = [k for k in CONDITIONS if k in data]
    n = len(conditions_present)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(n, 1, figsize=(10, 3.5 * n), sharex=True, sharey=True)
        if n == 1:
            axes = [axes]

        x = np.arange(1, N_LAYERS + 1)

        for ax, cond_key in zip(axes, conditions_present):
            meta = CONDITIONS[cond_key]
            arr = data[cond_key]
            layer_data = [arr[:, i] for i in range(N_LAYERS)]

            bp = ax.boxplot(
                layer_data,
                positions=x,
                widths=0.55,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(color="white", linewidth=1.5),
            )

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
# Plot 4 — Heatmap of delta vs original
# ---------------------------------------------------------------------------

def plot_delta_heatmap(data, output_dir):
    if "original" not in data:
        print("  [SKIP] plot_delta_heatmap requires 'original' condition.")
        return

    original_means = data["original"].mean(axis=0)

    rows = []
    labels = []

    for cond_key in ["np_shuffled", "full_shuffled"]:
        if cond_key not in data:
            continue
        delta = data[cond_key].mean(axis=0) - original_means
        rows.append(delta)
        labels.append(CONDITIONS[cond_key]["label"])

    if not rows:
        print("  [SKIP] No corruption conditions present for heatmap.")
        return

    heatmap = np.vstack(rows)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 2.8))
        im = ax.imshow(heatmap, aspect="auto", cmap="coolwarm", interpolation="nearest")

        ax.set_xticks(np.arange(N_LAYERS))
        ax.set_xticklabels(np.arange(1, N_LAYERS + 1))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("BERT layer")
        ax.set_title("Entropy increase relative to original — heatmap")

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Δ entropy vs original")

        fig.tight_layout()
        path = os.path.join(output_dir, "plot4_delta_heatmap.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Statistical tests and effect sizes
# ---------------------------------------------------------------------------

def run_significance_tests_vs_original(data, output_dir):
    if "original" not in data:
        print("  [SKIP] significance tests require 'original' condition.")
        return

    original = data["original"]
    lines = []

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
            _, p_val = stats.ttest_rel(corrupted[:, layer_idx], original[:, layer_idx])
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            row += f"p={p_val:.4f} {sig:<4}          "

        lines.append(row)

    output = "\n".join(lines)
    print("\n--- Paired t-test results (corruption vs original) ---")
    print(output)

    path = os.path.join(output_dir, "significance_tests_vs_original.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("Paired t-test: corruption condition vs original\n")
        f.write("* p<0.05  ** p<0.01  *** p<0.001\n\n")
        f.write(output)

    print(f"\nSaved: {path}")


def run_direct_comparison_tests(data, output_dir):
    if "np_shuffled" not in data or "full_shuffled" not in data:
        print("  [SKIP] direct comparison requires both np_shuffled and full_shuffled.")
        return

    np_arr = data["np_shuffled"]
    full_arr = data["full_shuffled"]

    lines = []
    lines.append(f"{'Layer':<8}{'NP vs Full-shuffle':<24}")
    lines.append("-" * 32)

    for layer_idx in range(N_LAYERS):
        _, p_val = stats.ttest_rel(np_arr[:, layer_idx], full_arr[:, layer_idx])
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        lines.append(f"{layer_idx + 1:<8}p={p_val:.4f} {sig:<4}")

    output = "\n".join(lines)
    print("\n--- Direct paired t-test results (NP-shuffled vs full-shuffle) ---")
    print(output)

    path = os.path.join(output_dir, "direct_comparison_tests.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("Paired t-test: NP-shuffled vs full-shuffle\n")
        f.write("* p<0.05  ** p<0.01  *** p<0.001\n\n")
        f.write(output)

    print(f"\nSaved: {path}")


def save_effect_sizes(data, output_dir):
    if "original" not in data:
        print("  [SKIP] effect size table requires 'original'.")
        return

    rows = []
    original = data["original"]

    for layer_idx in range(N_LAYERS):
        base = original[:, layer_idx]

        row = {
            "layer": layer_idx + 1,
            "original_mean": base.mean(),
        }

        for cond_key in ["np_shuffled", "full_shuffled"]:
            if cond_key not in data:
                continue

            arr = data[cond_key][:, layer_idx]
            diff = arr - base

            row[f"{cond_key}_mean"] = arr.mean()
            row[f"{cond_key}_delta_mean"] = diff.mean()
            row[f"{cond_key}_cohens_d_paired"] = paired_cohens_d(arr, base)

        if "np_shuffled" in data and "full_shuffled" in data:
            np_arr = data["np_shuffled"][:, layer_idx]
            full_arr = data["full_shuffled"][:, layer_idx]
            row["np_vs_full_delta_mean"] = (full_arr - np_arr).mean()
            row["np_vs_full_cohens_d_paired"] = paired_cohens_d(full_arr, np_arr)

        rows.append(row)

    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "effect_sizes.csv")
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def save_layer_group_summary(results_df, output_dir):
    summary = (
        results_df
        .groupby(["condition", "layer_group"], as_index=False)["entropy"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # flatten multi-index columns if needed
    summary.columns = [c if isinstance(c, str) else "_".join([x for x in c if x]) for c in summary.columns]

    path = os.path.join(output_dir, "layer_group_summary.csv")
    summary.to_csv(path, index=False)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot entropy results and run significance tests.")
    parser.add_argument(
        "--results_path",
        type=str,
        default=DEFAULT_RESULTS_PATH,
        help="Path to entropy results JSON file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save plots and tables."
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading results from: {args.results_path}")
    data = load_results(args.results_path)
    results_df = to_long_dataframe(data)

    print("\nGenerating plots...")
    plot_mean_entropy(data, args.output_dir)
    plot_entropy_delta(data, args.output_dir)
    plot_boxplots(data, args.output_dir)
    plot_delta_heatmap(data, args.output_dir)

    print("\nRunning significance tests...")
    run_significance_tests_vs_original(data, args.output_dir)
    run_direct_comparison_tests(data, args.output_dir)

    print("\nSaving summary tables...")
    save_effect_sizes(data, args.output_dir)
    save_layer_group_summary(results_df, args.output_dir)

    print(f"\nAll outputs saved to {args.output_dir}/")
