"""
Secondary workflow: offline entropy analysis for retained Phase 1 captures.

Reads HDF5 attention data, computes metrics, and writes CSV summaries for the
capture/analysis path that is no longer the repository's primary workflow.
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

    def _find_sample_groups(group, prefix=""):
        """Recursively find groups that contain layer_* children (the actual sample data)."""
        results = []
        has_layers = any(k.startswith("layer_") for k in group.keys())
        if has_layers:
            results.append((prefix, group))
        else:
            for k in group.keys():
                child = group[k]
                if isinstance(child, h5py.Group):
                    child_path = f"{prefix}/{k}" if prefix else k
                    results.extend(_find_sample_groups(child, child_path))
        return results

    with h5py.File(h5_path, 'r') as f:
        # Find all sample groups, handling both flat (sample_xxx/layer_*)
        # and nested (sample_xxx/subpath/layer_*) HDF5 structures
        sample_entries = []
        for k in f.keys():
            obj = f[k]
            if isinstance(obj, h5py.Group):
                sample_entries.extend(_find_sample_groups(obj, k))
        print(f"  Found {len(sample_entries)} samples in {os.path.basename(h5_path)}")

        for sample_key, grp in tqdm(sample_entries, desc="  Computing metrics"):
            question_id = grp.attrs.get("question_id", "")
            image_file = grp.attrs.get("image_file", "")

            layer_keys = sorted(
                [k for k in grp.keys() if k.startswith("layer_")],
                key=lambda x: int(x.split("_")[1])
            )

            for layer_key in layer_keys:
                layer_idx = int(layer_key.split("_")[1])
                layer_grp = grp[layer_key]

                # Load tv_attn matrix for rank computation
                tv_attn_2d = None
                if "tv_attn" in layer_grp:
                    tv_attn = layer_grp["tv_attn"][:]
                    if tv_attn.ndim == 3:
                        tv_attn_2d = tv_attn.mean(axis=0)  # head mean → [L_t, L_v]
                    else:
                        tv_attn_2d = tv_attn  # already [L_t, L_v]

                # Use pre-computed prune_scores if available, else compute from tv_attn
                if "prune_scores" in layer_grp:
                    prune_scores = layer_grp["prune_scores"][:]
                elif tv_attn_2d is not None:
                    prune_scores = tv_attn_2d.mean(axis=0)  # text mean → [L_v]
                else:
                    continue

                metrics = compute_all_metrics(prune_scores, topk_values, tv_attn=tv_attn_2d)

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
    group_key = "sample_uid" if "sample_uid" in df.columns else "sample_id"
    df = df.sort_values([group_key, "layer"]).copy()
    for metric in ["shannon", "renyi_2", "gini", "attn_rank"]:
        if metric in df.columns:
            df[f"{metric}_delta"] = df.groupby(group_key)[metric].diff()
    return df


def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-layer summary statistics across all samples.

    Returns:
        DataFrame indexed by layer with mean/std/median/min/max for each metric.
    """
    metrics = ["shannon", "renyi_2", "gini", "topk_32", "topk_64", "topk_128",
               "attn_rank", "attn_rank_ratio", "attn_sv_top1_ratio",
               "shannon_delta", "renyi_2_delta", "gini_delta", "attn_rank_delta"]
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
        basename = os.path.basename(h5_path)

        # Create per-file subdirectory: processed/<h5_stem>/
        file_stem = basename.replace(".h5", "")
        file_output_dir = os.path.join(args.output, file_stem)
        os.makedirs(file_output_dir, exist_ok=True)

        if df.empty:
            detail_path = os.path.join(file_output_dir, "per_sample_layer.csv")
            summary_path = os.path.join(file_output_dir, "summary.csv")
            df.to_csv(detail_path, index=False)
            pd.DataFrame().to_csv(summary_path, index=False)
            print(f"  No analyzable samples found, wrote empty outputs to: {file_output_dir}")
            continue

        # Tag with source file
        df["source_file"] = basename
        # Unique sample identifier across multiple capture runs/files
        df["sample_uid"] = df["source_file"].astype(str) + "::" + df["sample_id"].astype(str)
        # Extract dataset name from filename (e.g., gqa_20260305_120000.h5 → gqa)
        dataset_name = basename.split("_")[0] if "_" in basename else basename.replace(".h5", "")
        df["dataset"] = dataset_name

        # Compute delta per file (groupby sample_id within this file)
        df = compute_entropy_delta(df)

        # Save per-file detailed results
        detail_path = os.path.join(file_output_dir, "per_sample_layer.csv")
        df.to_csv(detail_path, index=False)
        print(
            f"  Detail: {detail_path}  "
            f"({len(df)} rows, {df['sample_uid'].nunique()} samples × {df['layer'].nunique()} layers)"
        )

        # Save per-file summary
        summary = compute_summary_stats(df)
        summary_path = os.path.join(file_output_dir, "summary.csv")
        summary.to_csv(summary_path)
        print(f"  Summary: {summary_path}")

        all_dfs.append(df)

    if not all_dfs:
        print("\nNo non-empty HDF5 files processed, skipping cross-file aggregation.")
        return

    # Cross-file aggregation (only when multiple files)
    if len(all_dfs) > 1:
        combined = pd.concat(all_dfs, ignore_index=True)

        # Save merged results under processed/merged/
        merged_dir = os.path.join(args.output, "merged")
        os.makedirs(merged_dir, exist_ok=True)

        detail_all_path = os.path.join(merged_dir, "per_sample_layer.csv")
        combined.to_csv(detail_all_path, index=False)
        print(f"\nMerged detail: {detail_all_path}")
        print(
            f"  {len(combined)} rows "
            f"({combined['sample_uid'].nunique()} samples × {combined['layer'].nunique()} layers)"
        )

        # Save per-dataset summary
        for ds_name, ddf in combined.groupby("dataset"):
            ds_dir = os.path.join(merged_dir, ds_name)
            os.makedirs(ds_dir, exist_ok=True)
            summary = compute_summary_stats(ddf)
            summary_path = os.path.join(ds_dir, "summary.csv")
            summary.to_csv(summary_path)
            print(f"  Summary ({ds_name}): {summary_path}")

        # Save cross-dataset summary
        summary_all = compute_summary_stats(combined)
        summary_all_path = os.path.join(merged_dir, "summary.csv")
        summary_all.to_csv(summary_all_path)
        print(f"  Summary (all): {summary_all_path}")
    else:
        print("\nSingle file processed, skipping cross-file aggregation.")


if __name__ == "__main__":
    main()
