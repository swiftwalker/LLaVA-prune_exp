from pathlib import Path

from report import save_batch_flops_bundle
from utils.io import load_yaml, make_run_dir


def _iter_run_dirs(runs_root: str | Path) -> list[Path]:
    root = Path(runs_root)
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "config.yaml").is_file()
    )


def _filter_run_dirs(run_dirs: list[Path], dataset: str | None = None) -> list[Path]:
    if not dataset:
        return run_dirs
    prefix = f"{dataset}_"
    return [path for path in run_dirs if path.name.startswith(prefix)]


def compute_batch_flops(
    config_path: str,
    runs_root: str,
    compute_llm_flops_fn,
    output_base_dir: str,
    scope: str = "runs",
    dataset: str | None = None,
    stats_json: str | None = None,
    fixed_text_len: float | None = None,
    length_stat: str = "mean",
    overrides: list[str] | None = None,
):
    if not stats_json and fixed_text_len is None:
        raise ValueError("Batch FLOPs requires either stats_json or fixed_text_len")

    _ = load_yaml(config_path)
    run_dirs = _filter_run_dirs(_iter_run_dirs(runs_root), dataset=dataset)
    if not run_dirs:
        raise ValueError(f"No run directories matched under {runs_root!r} for dataset={dataset!r}")

    per_run_rows = []
    generated_run_dirs = []
    for run_dir in run_dirs:
        output_run_dir, report = compute_llm_flops_fn(
            config_path=config_path,
            run_dir=str(run_dir),
            fixed_text_len=fixed_text_len,
            stats_json=stats_json,
            length_stat=length_stat,
            overrides=overrides,
        )
        generated_run_dirs.append(str(output_run_dir))
        per_run_rows.append(
            {
                "run_name": report["resolved_prune_spec"]["source_run_name"],
                "dataset": report["resolved_prune_spec"]["source_dataset"],
                "run_mode": report["resolved_prune_spec"]["run_mode"],
                "strategy": report["resolved_prune_spec"]["strategy_name"],
                "source_config": report["resolved_prune_spec"]["source_path"],
                "text_length_mode": report["text_length_source"]["mode"],
                "text_token_len": report["text_length_source"]["text_token_len"],
                "prefill_seq_len": report["prefill_seq_len"],
                "baseline_total_llm_flops": report["baseline"]["total_llm_flops"],
                "pruned_total_llm_flops": report["pruned"]["total_llm_flops"],
                "reduction_ratio": report["reduction_ratio"],
                "output_run_dir": str(output_run_dir),
            }
        )

    summary = {
        "scope": scope,
        "dataset_filter": dataset,
        "runs_root": str(runs_root),
        "matched_runs": len(per_run_rows),
        "text_length_mode": "fixed_text_length" if fixed_text_len is not None else "stats_based",
        "length_stat": length_stat if stats_json else None,
        "stats_json": stats_json,
        "fixed_text_len": fixed_text_len,
        "mean_reduction_ratio": (
            sum(row["reduction_ratio"] for row in per_run_rows) / len(per_run_rows)
            if per_run_rows else 0.0
        ),
    }

    batch_name = dataset or scope
    batch_run_dir = make_run_dir(Path(output_base_dir), "batch_flops", batch_name)

    save_batch_flops_bundle(
        run_dir=batch_run_dir,
        config_snapshot={
            "config_path": config_path,
            "runs_root": runs_root,
            "scope": scope,
            "dataset": dataset,
            "stats_json": stats_json,
            "fixed_text_len": fixed_text_len,
            "length_stat": length_stat,
            "overrides": overrides or [],
        },
        summary=summary,
        per_run_rows=per_run_rows,
    )
    return batch_run_dir, summary, per_run_rows
