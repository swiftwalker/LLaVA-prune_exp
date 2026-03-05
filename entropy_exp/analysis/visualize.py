"""
Visualization for entropy experiment results.

Generates:
1. Per-sample entropy curves across layers (overlaid on one plot)
2. Mean entropy curve with confidence bands per dataset
3. Entropy delta (layer-to-layer change) curves
4. Cross-dataset comparison
5. Sample-level consistency analysis

Usage:
    python visualize.py --csv outputs/processed/entropy_per_sample_layer.csv --output outputs/figures/
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D


# Plot style
plt.rcParams.update({
    "figure.figsize": (12, 7),
    "font.size": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "savefig.dpi": 150,
    "savefig.bbox_inches": "tight",
})


def plot_entropy_curves_overlay(df: pd.DataFrame, dataset: str, metric: str,
                                 output_dir: str, max_curves: int = 200):
    """
    Plot individual sample entropy curves overlaid on the same figure.
    Answers Q2: are entropy curves consistent across samples?
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    samples = df["sample_id"].unique()
    if len(samples) > max_curves:
        rng = np.random.RandomState(42)
        samples = rng.choice(samples, max_curves, replace=False)

    for sample_id in samples:
        sdf = df[df["sample_id"] == sample_id].sort_values("layer")
        ax.plot(sdf["layer"], sdf[metric], alpha=0.15, color="steelblue", linewidth=0.8)

    # Overlay mean curve
    mean_curve = df.groupby("layer")[metric].mean()
    ax.plot(mean_curve.index, mean_curve.values, color="red", linewidth=2.5, label="Mean")

    ax.set_xlabel("Layer Index")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric.replace('_', ' ').title()} Across Layers — {dataset.upper()} "
                 f"({len(samples)} samples)")
    ax.legend()

    path = os.path.join(output_dir, f"{dataset}_{metric}_overlay.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_entropy_mean_with_band(df: pd.DataFrame, dataset: str, metric: str,
                                 output_dir: str):
    """
    Plot mean entropy curve with ±1σ shaded band.
    """
    fig, ax = plt.subplots()

    stats = df.groupby("layer")[metric].agg(["mean", "std"])
    layers = stats.index.values
    mean_vals = stats["mean"].values
    std_vals = stats["std"].values

    ax.plot(layers, mean_vals, color="steelblue", linewidth=2, label="Mean")
    ax.fill_between(layers, mean_vals - std_vals, mean_vals + std_vals,
                     alpha=0.2, color="steelblue", label="±1σ")

    ax.set_xlabel("Layer Index")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric.replace('_', ' ').title()} — {dataset.upper()} (mean ± σ)")
    ax.legend()

    path = os.path.join(output_dir, f"{dataset}_{metric}_mean_band.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_entropy_delta(df: pd.DataFrame, dataset: str, output_dir: str):
    """
    Plot entropy delta (change between consecutive layers).
    Answers Q1: where is the max entropy change / min entropy point?
    """
    delta_col = "shannon_delta"
    if delta_col not in df.columns:
        print(f"  Skipping delta plot: {delta_col} not found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: delta curves overlay
    ax = axes[0]
    samples = df["sample_id"].unique()
    max_show = min(len(samples), 200)
    rng = np.random.RandomState(42)
    show_samples = rng.choice(samples, max_show, replace=False)

    for sid in show_samples:
        sdf = df[df["sample_id"] == sid].sort_values("layer").dropna(subset=[delta_col])
        ax.plot(sdf["layer"], sdf[delta_col], alpha=0.12, color="steelblue", linewidth=0.8)

    mean_delta = df.dropna(subset=[delta_col]).groupby("layer")[delta_col].mean()
    ax.plot(mean_delta.index, mean_delta.values, color="red", linewidth=2.5, label="Mean Δ")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Shannon Entropy Δ")
    ax.set_title(f"Entropy Change (Layer-to-Layer) — {dataset.upper()}")
    ax.legend()

    # Right: heatmap of where max-delta and min-entropy occur
    ax = axes[1]
    # For each sample, find: layer of max |delta|, layer of min entropy
    records = []
    for sid in df["sample_id"].unique():
        sdf = df[df["sample_id"] == sid].sort_values("layer")
        min_entropy_layer = sdf.loc[sdf["shannon"].idxmin(), "layer"]
        delta_df = sdf.dropna(subset=[delta_col])
        if len(delta_df) > 0:
            max_delta_layer = delta_df.loc[delta_df[delta_col].abs().idxmax(), "layer"]
        else:
            max_delta_layer = np.nan
        records.append({"min_entropy_layer": min_entropy_layer, "max_delta_layer": max_delta_layer})

    rec_df = pd.DataFrame(records)
    bins = np.arange(-0.5, 32.5, 1)

    ax.hist(rec_df["min_entropy_layer"].dropna(), bins=bins, alpha=0.6,
            color="steelblue", label="Min Entropy Layer")
    ax.hist(rec_df["max_delta_layer"].dropna(), bins=bins, alpha=0.6,
            color="coral", label="Max |ΔH| Layer")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of Key Layers — {dataset.upper()}")
    ax.legend()

    fig.tight_layout()
    path = os.path.join(output_dir, f"{dataset}_entropy_delta.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_cross_dataset_comparison(df: pd.DataFrame, output_dir: str):
    """
    Compare entropy curves across datasets.
    Answers Q2: is entropy driven by image+model or model alone?
    """
    datasets = df["dataset"].unique()
    if len(datasets) < 2:
        print("  Skipping cross-dataset comparison: need >= 2 datasets")
        return

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    metrics = ["shannon", "renyi_2", "gini"]
    colors = cm.Set2(np.linspace(0, 1, len(datasets)))

    for ax, metric in zip(axes, metrics):
        for i, ds in enumerate(sorted(datasets)):
            ddf = df[df["dataset"] == ds]
            stats = ddf.groupby("layer")[metric].agg(["mean", "std"])
            layers = stats.index.values
            mean_vals = stats["mean"].values
            std_vals = stats["std"].values

            ax.plot(layers, mean_vals, color=colors[i], linewidth=2, label=ds.upper())
            ax.fill_between(layers, mean_vals - std_vals, mean_vals + std_vals,
                            alpha=0.1, color=colors[i])

        ax.set_xlabel("Layer Index")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(metric.replace("_", " ").title())
        ax.legend()

    fig.suptitle("Cross-Dataset Entropy Comparison (mean ± σ)", fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, "cross_dataset_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_variance_analysis(df: pd.DataFrame, dataset: str, output_dir: str):
    """
    Analyze inter-sample variance at each layer.
    High variance → image-dependent; Low variance → model-determined.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: coefficient of variation (CV = std / mean) per layer
    ax = axes[0]
    for metric, color in [("shannon", "steelblue"), ("gini", "coral")]:
        stats = df.groupby("layer")[metric].agg(["mean", "std"])
        cv = stats["std"] / stats["mean"].clip(lower=1e-8)
        ax.plot(cv.index, cv.values, color=color, linewidth=2, label=metric, marker='o', markersize=4)

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Coefficient of Variation (σ/μ)")
    ax.set_title(f"Inter-Sample Variability — {dataset.upper()}")
    ax.legend()

    # Right: entropy distribution boxplot at selected layers
    ax = axes[1]
    selected_layers = [0, 4, 8, 12, 16, 20, 24, 28, 31]
    selected_layers = [l for l in selected_layers if l in df["layer"].unique()]
    box_data = [df[df["layer"] == l]["shannon"].values for l in selected_layers]
    bp = ax.boxplot(box_data, positions=range(len(selected_layers)), widths=0.6)
    ax.set_xticks(range(len(selected_layers)))
    ax.set_xticklabels([str(l) for l in selected_layers])
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Shannon Entropy")
    ax.set_title(f"Entropy Distribution at Key Layers — {dataset.upper()}")

    fig.tight_layout()
    path = os.path.join(output_dir, f"{dataset}_variance_analysis.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize entropy experiment results")
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to entropy_per_sample_layer.csv")
    parser.add_argument("--output", type=str, default="entropy_exp/outputs/figures",
                        help="Output directory for figures")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading: {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"  {len(df)} rows, {df['sample_id'].nunique()} samples, "
          f"datasets: {df['dataset'].unique().tolist()}")

    # Per-dataset plots
    for dataset_name, ddf in df.groupby("dataset"):
        print(f"\n--- {dataset_name.upper()} ---")
        ds_dir = os.path.join(args.output, dataset_name)
        os.makedirs(ds_dir, exist_ok=True)

        plot_entropy_curves_overlay(ddf, dataset_name, "shannon", ds_dir)
        plot_entropy_curves_overlay(ddf, dataset_name, "gini", ds_dir)
        plot_entropy_mean_with_band(ddf, dataset_name, "shannon", ds_dir)
        plot_entropy_mean_with_band(ddf, dataset_name, "renyi_2", ds_dir)
        plot_entropy_mean_with_band(ddf, dataset_name, "gini", ds_dir)
        plot_entropy_delta(ddf, dataset_name, ds_dir)
        plot_variance_analysis(ddf, dataset_name, ds_dir)

    # Cross-dataset comparison
    if df["dataset"].nunique() >= 2:
        print(f"\n--- Cross-Dataset ---")
        plot_cross_dataset_comparison(df, args.output)

    print("\nDone!")


if __name__ == "__main__":
    main()
