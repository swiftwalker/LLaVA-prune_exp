"""
Offline entropy analysis: reads HDF5 attention data, computes metrics, outputs CSV.

Usage:
    python entropy_analysis.py --h5 outputs/raw/gqa_*.h5 --output outputs/processed/
"""

import argparse
import os
import sys
import glob
import json
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add src to path
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

from metrics import (
    normalize_to_distribution,
    shannon_entropy,
    renyi_entropy,
    gini_coefficient,
    topk_concentration,
    compute_all_metrics,
)


def analyze_h5_file(h5_path: str, topk_values: list = None) -> pd.DataFrame:
    """
    Read one HDF5 file and compute entropy metrics for all samples and layers.

    Returns:
        DataFrame with columns:
            sample_id, question_id, image_file, layer, shannon, renyi_2, gini,
            topk_32, topk_64, topk_128, prune_scores_mean, prune_scores_std,
            prune_scores_max, prune_scores_min
    """
    if topk_values is None:
        topk_values = [32, 64, 128]

    rows = []

    with h5py.File(h5_path, 'r') as f:
        sample_keys = [k for k in f.keys() if k.startswith("sample_")]
        print(f"  Found {len(sample_keys)} samples in {os.path.basename(h5_path)}")

        for sample_key in tqdm(sample_keys, desc="  Computing metrics"):
            grp = f[sample_key]
            question_id = grp.attrs.get("question_id", "")
            image_file = grp.attrs.get("image_file", "")

            layer_keys = sorted(
                [k for k in grp.keys() if k.startswith("layer_")],
                key=lambda x: int(x.split("_")[1])
            )

            for layer_key in layer_keys:
                layer_idx = int(layer_key.split("_")[1])
                layer_grp = grp[layer_key]

                # Use pre-computed prune_scores if available, else compute from tv_attn
                if "prune_scores" in layer_grp:
                    prune_scores = layer_grp["prune_scores"][:]
                elif "tv_attn" in layer_grp:
                    tv_attn = layer_grp["tv_attn"][:]
                    if tv_attn.ndim == 3:
                        tv_attn = tv_attn.mean(axis=0)  # head mean
                    prune_scores = tv_attn.mean(axis=0)  # text mean → [L_v]
                else:
                    continue

                metrics = compute_all_metrics(prune_scores, topk_values)

                row = {
                    "sample_id": sample_key,
                    "question_id": str(question_id),
                    "image_file": str(image_file),
                    "layer": layer_idx,
                    "prune_scores_mean": float(prune_scores.mean()),
                    "prune_scores_std": float(prune_scores.std()),
                    "prune_scores_max": float(prune_scores.max()),
                    "prune_scores_min": float(prune_scores.min()),
                }
                row.update(metrics)
                rows.append(row)

    return pd.DataFrame(rows)


def compute_entropy_delta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute layer-to-layer entropy change (delta) for each sample.
    Adds columns: shannon_delta, renyi_2_delta, gini_delta.
    """
    df = df.sort_values(["sample_id", "layer"]).copy()
    for metric in ["shannon", "renyi_2", "gini"]:
        df[f"{metric}_delta"] = df.groupby("sample_id")[metric].diff()
    return df


def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-layer summary statistics across all samples.

    Returns:
        DataFrame indexed by layer with mean/std/median/min/max for each metric.
    """
    metrics = ["shannon", "renyi_2", "gini", "topk_32", "topk_64", "topk_128",
               "shannon_delta", "renyi_2_delta", "gini_delta"]
    available_metrics = [m for m in metrics if m in df.columns]

    agg_funcs = ["mean", "std", "median", "min", "max"]
    summary = df.groupby("layer")[available_metrics].agg(agg_funcs)
    summary.columns = ['_'.join(col) for col in summary.columns]
    return summary


def main():
    parser = argparse.ArgumentParser(description="Offline entropy analysis from HDF5 attention data")
    parser.add_argument("--h5", type=str, nargs="+", required=True,
                        help="Path(s) to HDF5 files (supports glob patterns)")
    parser.add_argument("--output", type=str, default="entropy_exp/outputs/processed",
                        help="Output directory for CSV results")
    parser.add_argument("--topk", type=int, nargs="+", default=[32, 64, 128],
                        help="Top-K values for concentration metric")
    args = parser.parse_args()

    # Expand glob patterns
    h5_files = []
    for pattern in args.h5:
        expanded = glob.glob(pattern)
        if expanded:
            h5_files.extend(expanded)
        elif os.path.exists(pattern):
            h5_files.append(pattern)
    h5_files = sorted(set(h5_files))

    if not h5_files:
        print("No HDF5 files found!")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    all_dfs = []
    for h5_path in h5_files:
        print(f"\nProcessing: {h5_path}")
        df = analyze_h5_file(h5_path, args.topk)
        # Tag with source file
        df["source_file"] = os.path.basename(h5_path)
        # Extract dataset name from filename (e.g., gqa_20260305_120000.h5 → gqa)
        basename = os.path.basename(h5_path)
        dataset_name = basename.split("_")[0] if "_" in basename else basename.replace(".h5", "")
        df["dataset"] = dataset_name
        all_dfs.append(df)

    # Merge all
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = compute_entropy_delta(combined)

    # Save per-sample detailed results
    detail_path = os.path.join(args.output, "entropy_per_sample_layer.csv")
    combined.to_csv(detail_path, index=False)
    print(f"\nDetailed results: {detail_path}")
    print(f"  {len(combined)} rows ({combined['sample_id'].nunique()} samples × {combined['layer'].nunique()} layers)")

    # Save per-layer summary (per dataset)
    for dataset_name, ddf in combined.groupby("dataset"):
        summary = compute_summary_stats(ddf)
        summary_path = os.path.join(args.output, f"entropy_summary_{dataset_name}.csv")
        summary.to_csv(summary_path)
        print(f"  Summary ({dataset_name}): {summary_path}")

    # Save cross-dataset summary
    summary_all = compute_summary_stats(combined)
    summary_all_path = os.path.join(args.output, "entropy_summary_all.csv")
    summary_all.to_csv(summary_all_path)
    print(f"  Summary (all): {summary_all_path}")


if __name__ == "__main__":
    main()
