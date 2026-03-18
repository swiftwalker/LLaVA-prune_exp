from pathlib import Path

from utils.io import dump_yaml, ensure_dir, write_csv, write_json


def save_text_stats_bundle(run_dir: str | Path, config_snapshot: dict, summary: dict, per_sample: list[dict]) -> None:
    run_dir = ensure_dir(run_dir)
    dump_yaml(run_dir / "config.yaml", config_snapshot)
    write_json(run_dir / "summary.json", summary)
    write_csv(run_dir / "summary.csv", [summary], fieldnames=list(summary.keys()))
    if per_sample:
        write_csv(run_dir / "per_sample.csv", per_sample, fieldnames=list(per_sample[0].keys()))


def save_flops_bundle(
    run_dir: str | Path,
    config_snapshot: dict,
    resolved_method: dict,
    report: dict,
    notes: list[str],
) -> None:
    run_dir = ensure_dir(run_dir)
    dump_yaml(run_dir / "config.yaml", config_snapshot)
    write_json(run_dir / "resolved_method.json", resolved_method)
    write_json(run_dir / "report.json", report)
    rows = []
    baseline = report["baseline"]
    pruned = report["pruned"]
    for key in ["self_attention_flops", "mlp_flops", "prune_method_flops", "total_llm_flops"]:
        rows.append(
            {
                "metric": key,
                "baseline": baseline.get(key, 0.0),
                "pruned": pruned.get(key, 0.0),
            }
        )
    rows.append(
        {
            "metric": "reduction_ratio",
            "baseline": "",
            "pruned": report["reduction_ratio"],
        }
    )
    write_csv(run_dir / "report.csv", rows, fieldnames=["metric", "baseline", "pruned"])
    (run_dir / "notes.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")


def save_batch_flops_bundle(
    run_dir: str | Path,
    config_snapshot: dict,
    summary: dict,
    per_run_rows: list[dict],
) -> None:
    run_dir = ensure_dir(run_dir)
    dump_yaml(run_dir / "config.yaml", config_snapshot)
    write_json(run_dir / "summary.json", summary)
    write_csv(run_dir / "summary.csv", [summary], fieldnames=list(summary.keys()))
    if per_run_rows:
        write_csv(run_dir / "per_run.csv", per_run_rows, fieldnames=list(per_run_rows[0].keys()))

