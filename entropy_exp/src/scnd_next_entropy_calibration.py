"""Prepare and summarize architecture-level SCND entropy calibration runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import yaml

from strategies.scnd_entropy_calibration import ENTROPY_DEFINITION, model_config_fingerprint


def iter_top_level_json_object(path: Path, chunk_size: int = 1 << 20) -> Iterator[tuple[str, dict[str, Any]]]:
    """Stream key/value pairs from a large top-level JSON object."""
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    started = False
    finished = False

    with path.open("r", encoding="utf-8") as handle:
        while not finished:
            chunk = handle.read(chunk_size)
            if chunk:
                buffer = buffer[position:] + chunk
                position = 0
            elif position >= len(buffer):
                break

            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if not started:
                    if position >= len(buffer):
                        break
                    if buffer[position] != "{":
                        raise ValueError(f"Expected a top-level JSON object in {path}")
                    position += 1
                    started = True
                    continue

                while position < len(buffer) and (buffer[position].isspace() or buffer[position] == ","):
                    position += 1
                if position >= len(buffer):
                    break
                if buffer[position] == "}":
                    position += 1
                    finished = True
                    break

                try:
                    key, key_end = decoder.raw_decode(buffer, position)
                    cursor = key_end
                    while cursor < len(buffer) and buffer[cursor].isspace():
                        cursor += 1
                    if cursor >= len(buffer):
                        break
                    if buffer[cursor] != ":":
                        raise ValueError(f"Expected ':' after key {key!r} in {path}")
                    cursor += 1
                    while cursor < len(buffer) and buffer[cursor].isspace():
                        cursor += 1
                    value, value_end = decoder.raw_decode(buffer, cursor)
                except json.JSONDecodeError:
                    break

                if not isinstance(key, str) or not isinstance(value, dict):
                    raise ValueError("GQA source must map string question ids to JSON objects")
                yield key, value
                position = value_end

            if not chunk and not finished:
                raise ValueError(f"Unexpected end of JSON input: {path}")

    if not finished:
        raise ValueError(f"Top-level JSON object was not terminated: {path}")


def _stable_priority(seed: int, question_id: str) -> str:
    return hashlib.sha256(f"{seed}:{question_id}".encode("utf-8")).hexdigest()


def _question_stratum(question: dict[str, Any]) -> str:
    types = question.get("types") or {}
    structural = str(types.get("structural", "unknown"))
    semantic = str(types.get("semantic", "unknown"))
    return f"{structural}|{semantic}"


def prepare_gqa_calibration_subset(
    source: Path,
    image_folder: Path,
    output: Path,
    manifest_path: Path,
    target_images: int,
    seed: int,
    shard_count: int = 1,
) -> dict[str, Any]:
    if shard_count < 1:
        raise ValueError("shard_count must be at least 1")
    by_image: dict[str, tuple[str, str, dict[str, Any]]] = {}
    available_image_ids = {path.stem for path in image_folder.glob("*.jpg")}
    source_questions = 0
    missing_images = 0
    for question_id, question in iter_top_level_json_object(source):
        source_questions += 1
        image_id = str(question.get("imageId", ""))
        if not image_id or image_id not in available_image_ids:
            missing_images += 1
            continue
        priority = _stable_priority(seed, question_id)
        previous = by_image.get(image_id)
        if previous is None or priority < previous[0]:
            by_image[image_id] = (priority, question_id, question)

    strata: dict[str, list[tuple[str, str, str, dict[str, Any]]]] = defaultdict(list)
    for image_id, (priority, question_id, question) in by_image.items():
        strata[_question_stratum(question)].append((priority, image_id, question_id, question))
    for candidates in strata.values():
        candidates.sort(key=lambda item: (item[0], item[2]))

    selected: list[tuple[str, str, str, dict[str, Any]]] = []
    offsets = {stratum: 0 for stratum in strata}
    ordered_strata = sorted(strata)
    while len(selected) < target_images:
        made_progress = False
        for stratum in ordered_strata:
            offset = offsets[stratum]
            if offset >= len(strata[stratum]):
                continue
            selected.append(strata[stratum][offset])
            offsets[stratum] = offset + 1
            made_progress = True
            if len(selected) == target_images:
                break
        if not made_progress:
            break
    if len(selected) != target_images:
        raise ValueError(f"Requested {target_images} unique images, but only selected {len(selected)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    selected_strata: dict[str, int] = defaultdict(int)
    payloads: list[dict[str, Any]] = []
    for _priority, image_id, question_id, question in selected:
        stratum = _question_stratum(question)
        selected_strata[stratum] += 1
        payload = {
            "question_id": question_id,
            "image": f"{image_id}.jpg",
            "text": str(question["question"]),
            "answer": question.get("answer"),
            "metadata": {
                "calibration_split": "gqa_train_balanced",
                "question_stratum": stratum,
                "types": question.get("types", {}),
            },
        }
        payloads.append(payload)

    with output.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    shard_records: list[dict[str, Any]] = []
    if shard_count > 1:
        shard_width = len(str(shard_count - 1))
        for shard_index in range(shard_count):
            shard_path = output.with_name(
                f"{output.stem}_shard{shard_index:0{shard_width}d}of{shard_count}{output.suffix}"
            )
            shard_payloads = payloads[shard_index::shard_count]
            with shard_path.open("w", encoding="utf-8") as handle:
                for payload in shard_payloads:
                    handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
            shard_records.append(
                {
                    "index": shard_index,
                    "path": str(shard_path.resolve()),
                    "sample_count": len(shard_payloads),
                }
            )

    manifest = {
        "schema_version": 1,
        "source": str(source.resolve()),
        "image_folder": str(image_folder.resolve()),
        "output": str(output.resolve()),
        "seed": seed,
        "target_unique_images": target_images,
        "selected_unique_images": len(selected),
        "source_question_count": source_questions,
        "source_unique_image_count": len(by_image),
        "missing_image_question_count": missing_images,
        "stratification": "types.structural|types.semantic round-robin",
        "selected_stratum_counts": dict(sorted(selected_strata.items())),
        "shard_count": shard_count,
        "shards": shard_records,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _load_stats(run_dir: Path, layer: int) -> tuple[list[dict[str, Any]], list[float]]:
    stats_path = run_dir / "stats.jsonl"
    if not stats_path.is_file():
        raise FileNotFoundError(f"Missing stats file: {stats_path}")
    rows: list[dict[str, Any]] = []
    values: list[float] = []
    entropy_key = f"layer_{layer}_saliency_entropy_norm"
    with stats_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            value = row.get(entropy_key)
            if value is None:
                continue
            value = float(value)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Out-of-range {entropy_key}={value} in {stats_path}")
            rows.append(row)
            values.append(value)
    if not values:
        raise ValueError(f"No {entropy_key} values found in {stats_path}")
    return rows, values


def _summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "p05": float(np.quantile(array, 0.05, method="linear")),
        "p25": float(np.quantile(array, 0.25, method="linear")),
        "median": float(np.quantile(array, 0.50, method="linear")),
        "p75": float(np.quantile(array, 0.75, method="linear")),
        "p95": float(np.quantile(array, 0.95, method="linear")),
        "max": float(array.max()),
    }


def build_calibration_artifact(
    run_dir: Path | list[Path],
    output: Path,
    layer: int,
    q_low_probability: float,
    q_high_probability: float,
) -> dict[str, Any]:
    run_dirs = [run_dir] if isinstance(run_dir, Path) else list(run_dir)
    if not run_dirs:
        raise ValueError("At least one calibration run directory is required")

    values: list[float] = []
    question_ids: set[str] = set()
    source_runs: list[dict[str, Any]] = []
    reference_model_name: str | None = None
    reference_fingerprint: str | None = None
    reference_dataset: str | None = None
    for current_run_dir in run_dirs:
        rows, run_values = _load_stats(current_run_dir, layer)
        config_path = current_run_dir / "config.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing run config: {config_path}")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        metadata = config.get("model_config_metadata") or {}
        model_name = str(config["model"]["name"])
        fingerprint = model_config_fingerprint(metadata)
        dataset = str(config.get("_run_meta", {}).get("dataset", "unknown"))
        if reference_model_name is None:
            reference_model_name = model_name
            reference_fingerprint = fingerprint
            reference_dataset = dataset
        elif (model_name, fingerprint, dataset) != (
            reference_model_name,
            reference_fingerprint,
            reference_dataset,
        ):
            raise ValueError("Calibration shards must use the same model fingerprint and dataset")

        for row in rows:
            if row.get("question_id") is None:
                raise ValueError(f"Missing question_id in calibration stats: {current_run_dir}")
            question_id = str(row["question_id"])
            if question_id in question_ids:
                raise ValueError(f"Duplicate calibration question_id across shards: {question_id}")
            question_ids.add(question_id)
        values.extend(run_values)
        source_runs.append(
            {
                "run_dir": str(current_run_dir.resolve()),
                "sample_count": len(run_values),
            }
        )

    assert reference_model_name is not None
    assert reference_fingerprint is not None
    value_array = np.asarray(values, dtype=np.float64)
    q_low = float(np.quantile(value_array, q_low_probability, method="linear"))
    q_high = float(np.quantile(value_array, q_high_probability, method="linear"))
    if q_high - q_low < 1e-4:
        raise ValueError(f"Calibration entropy interval is degenerate: [{q_low}, {q_high}]")

    artifact = {
        "schema_version": 1,
        "mode": "quantile_affine",
        "entropy_definition": ENTROPY_DEFINITION,
        "model_name": reference_model_name,
        "model_config_fingerprint": reference_fingerprint,
        "capture_layer": int(layer),
        "quantiles": {
            "low_probability": float(q_low_probability),
            "high_probability": float(q_high_probability),
            "low_value": q_low,
            "high_value": q_high,
            "method": "numpy_linear",
        },
        "calibration_source": {
            "run_dirs": [source["run_dir"] for source in source_runs],
            "runs": source_runs,
            "dataset": reference_dataset,
            "sample_count": len(values),
            "unique_question_count": len(question_ids),
            "raw_entropy_summary": _summary(values),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return artifact


def summarize_runs(run_dirs: list[Path], output_dir: Path, layer: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    value_rows: list[dict[str, Any]] = []
    role_rows: list[dict[str, Any]] = []
    manifest_runs: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        rows, values = _load_stats(run_dir, layer)
        config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        dataset = str(config.get("_run_meta", {}).get("dataset", "unknown"))
        summary_rows.append({"dataset": dataset, **_summary(values)})
        for row, value in zip(rows, values):
            value_rows.append(
                {
                    "dataset": dataset,
                    "question_id": row.get("question_id"),
                    "H_norm_raw": value,
                    "H_norm_control": row.get(f"layer_{layer}_saliency_entropy_norm_control"),
                    "patch_only_H_norm": row.get(f"layer_{layer}_patch_only_saliency_entropy_norm"),
                }
            )
        role_keys = (
            "base_token_keep_rate",
            "local_patch_token_keep_rate",
            "newline_token_keep_rate",
            "newline_saliency_mass_ratio",
        )
        role_summary: dict[str, Any] = {"dataset": dataset, "count": len(rows)}
        for key in role_keys:
            collected = [row.get(f"layer_{layer}_{key}") for row in rows]
            collected = [float(value) for value in collected if value is not None]
            role_summary[f"{key}_mean"] = float(np.mean(collected)) if collected else None
            role_summary[f"{key}_std"] = float(np.std(collected)) if collected else None
        role_rows.append(role_summary)
        manifest_runs.append({"dataset": dataset, "run_dir": str(run_dir.resolve()), "samples": len(rows)})

    summary_rows.sort(key=lambda row: row["dataset"])
    role_rows.sort(key=lambda row: row["dataset"])
    for filename, rows in (
        ("entropy_summary.csv", summary_rows),
        ("entropy_values_long.csv", value_rows),
        ("token_role_summary.csv", role_rows),
    ):
        with (output_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    manifest = {
        "schema_version": 1,
        "entropy_definition": ENTROPY_DEFINITION,
        "capture_layer": int(layer),
        "runs": manifest_runs,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subset = subparsers.add_parser("prepare-gqa-subset")
    subset.add_argument("--source", type=Path, required=True)
    subset.add_argument("--image-folder", type=Path, required=True)
    subset.add_argument("--output", type=Path, required=True)
    subset.add_argument("--manifest", type=Path, required=True)
    subset.add_argument("--target-images", type=int, default=4096)
    subset.add_argument("--seed", type=int, default=42)
    subset.add_argument("--shard-count", type=int, default=1)

    artifact = subparsers.add_parser("build-artifact")
    artifact.add_argument("--run-dir", action="append", type=Path, required=True)
    artifact.add_argument("--output", type=Path, required=True)
    artifact.add_argument("--layer", type=int, default=2)
    artifact.add_argument("--q-low", type=float, default=0.05)
    artifact.add_argument("--q-high", type=float, default=0.95)

    summary = subparsers.add_parser("summarize")
    summary.add_argument("--run-dir", action="append", type=Path, required=True)
    summary.add_argument("--output-dir", type=Path, required=True)
    summary.add_argument("--layer", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare-gqa-subset":
        result = prepare_gqa_calibration_subset(
            source=args.source,
            image_folder=args.image_folder,
            output=args.output,
            manifest_path=args.manifest,
            target_images=args.target_images,
            seed=args.seed,
            shard_count=args.shard_count,
        )
    elif args.command == "build-artifact":
        if not 0.0 <= args.q_low < args.q_high <= 1.0:
            raise ValueError("Quantile probabilities must satisfy 0 <= q_low < q_high <= 1")
        result = build_calibration_artifact(
            run_dir=args.run_dir,
            output=args.output,
            layer=args.layer,
            q_low_probability=args.q_low,
            q_high_probability=args.q_high,
        )
    else:
        result = summarize_runs(args.run_dir, args.output_dir, args.layer)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
