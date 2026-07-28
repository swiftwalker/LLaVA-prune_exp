#!/usr/bin/env python3
"""Analyze and enforce the staged SCND next-layer routing gates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval_datasets import (  # noqa: E402
    load_textvqa_evaluator,
    normalize_open_answer,
    normalize_pope_answer,
    parse_scienceqa_answer,
    textvqa_prompt_processor,
)
from strategies.scnd_visual_roles import visual_token_role_names  # noqa: E402


DATASETS = ("gqa", "textvqa", "pope", "mme", "scienceqa")
CANDIDATES = ("legacy_scnd", "sparsevlm_topk", "native_maxmin")
CF_SUFFIX = {
    "legacy_scnd": "legacy",
    "sparsevlm_topk": "sparsevlm",
    "native_maxmin": "maxmin",
}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: Path, title: str, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(f"# {title}\n\nNo rows.\n", encoding="utf-8")
        return
    fields = list(rows[0])
    lines = [f"# {title}", "", "| " + " | ".join(fields) + " |", "|" + "---|" * len(fields)]
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                value = f"{value:.6f}"
            values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mean(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    return float(statistics.fmean(materialized)) if materialized else 0.0


def _config_value(run_dir: Path, keys: Sequence[str], default: Any = None) -> Any:
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def discover_runs(root: Path, *, require_eval: bool = False) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {dataset: [] for dataset in DATASETS}
    for answers_path in sorted(root.glob("runs/*/*/*/answers.jsonl")):
        run_dir = answers_path.parent
        if not (run_dir / "config.yaml").is_file() or not (run_dir / "stats.jsonl").is_file():
            continue
        if require_eval and not (run_dir / "eval" / "summary.json").is_file():
            continue
        dataset = str(_config_value(run_dir, ("_run_meta", "dataset"), answers_path.parents[1].name))
        if dataset in found:
            found[dataset].append(run_dir.resolve())
    return found


def one_run(root: Path, dataset: str, *, name_contains: str = "", require_eval: bool = False) -> Path:
    candidates = discover_runs(root, require_eval=require_eval)[dataset]
    if name_contains:
        candidates = [path for path in candidates if name_contains in path.name]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one run for dataset={dataset!r}, contains={name_contains!r} under {root}, "
            f"found {len(candidates)}: {[path.name for path in candidates]}"
        )
    return candidates[0]


def validate_aligned_rows(*groups: Sequence[Mapping[str, Any]], label: str) -> None:
    lengths = {len(group) for group in groups}
    if len(lengths) != 1:
        raise ValueError(f"{label}: row counts differ: {sorted(lengths)}")
    for index, rows in enumerate(zip(*groups)):
        question_ids = {str(row.get("question_id")) for row in rows}
        sample_indices = {int(row.get("sample_idx", index)) for row in rows}
        prompts = {str(row["prompt"]) for row in rows if "prompt" in row}
        if len(question_ids) != 1 or len(sample_indices) != 1 or len(prompts) > 1:
            raise ValueError(
                f"{label}: sample alignment mismatch at row {index}: "
                f"question_ids={question_ids}, sample_indices={sample_indices}, prompts={prompts}"
            )


def compare_answers_exact(reference: Path, candidate: Path) -> dict[str, Any]:
    reference_rows = read_jsonl(reference)
    candidate_rows = read_jsonl(candidate)
    validate_aligned_rows(reference_rows, candidate_rows, label="answer comparison")
    mismatches = []
    for index, (expected, actual) in enumerate(zip(reference_rows, candidate_rows)):
        if expected != actual:
            mismatches.append(
                {
                    "sample_idx": index,
                    "question_id": str(expected.get("question_id")),
                    "expected": expected,
                    "actual": actual,
                }
            )
    return {
        "count": len(reference_rows),
        "mismatch_count": len(mismatches),
        "exact": not mismatches,
        "mismatches": mismatches,
    }


def _index_value_map(row: Mapping[str, Any], layer: int, value_field: str) -> dict[int, float]:
    indices = row.get(f"layer_{layer}_current_patch_indices")
    values = row.get(f"layer_{layer}_{value_field}")
    if indices is None or values is None:
        return {}
    if len(indices) != len(values):
        raise ValueError(
            f"layer {layer}: {value_field} length {len(values)} != current index length {len(indices)}"
        )
    return {int(index): float(value) for index, value in zip(indices, values)}


def _role_counts(indices: Sequence[int], layout: Mapping[str, Any]) -> Counter[str]:
    if not indices:
        return Counter()
    return Counter(visual_token_role_names([int(index) for index in indices], dict(layout)))


def forensic_report(
    *,
    capture_root: Path,
    reference_root: Path,
    legacy_root: Path | None,
    output_dir: Path,
) -> dict[str, Any]:
    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    transition_summary: list[dict[str, Any]] = []
    answer_checks: dict[str, Any] = {}

    for dataset in DATASETS:
        capture_run = one_run(capture_root, dataset)
        reference_run = one_run(reference_root, dataset)
        answer_check = compare_answers_exact(reference_run / "answers.jsonl", capture_run / "answers.jsonl")
        answer_checks[dataset] = {
            key: value for key, value in answer_check.items() if key != "mismatches"
        }
        if answer_check["mismatches"]:
            write_json(output_dir / f"{dataset}_answer_mismatches.json", answer_check["mismatches"])

        if legacy_root is not None:
            legacy_run = one_run(legacy_root, dataset, name_contains="audit")
            legacy_answers = read_jsonl(legacy_run / "answers.jsonl")
            herc_answers = read_jsonl(capture_run / "answers.jsonl")
            validate_aligned_rows(legacy_answers, herc_answers, label=f"{dataset} HERC transition")
            legacy_utility = score_answers(dataset, legacy_answers)
            herc_utility = score_answers(dataset, herc_answers)
            transitions: Counter[str] = Counter()
            for index, (legacy_answer, herc_answer, legacy_score, herc_score) in enumerate(
                zip(legacy_answers, herc_answers, legacy_utility, herc_utility)
            ):
                if herc_score > legacy_score:
                    transition = "improved"
                elif herc_score < legacy_score:
                    transition = "regressed"
                else:
                    transition = "unchanged"
                transitions[transition] += 1
                transition_rows.append(
                    {
                        "dataset": dataset,
                        "sample_idx": index,
                        "question_id": legacy_answer.get("question_id"),
                        "legacy_answer": legacy_answer.get("text"),
                        "herc_answer": herc_answer.get("text"),
                        "legacy_utility": legacy_score,
                        "herc_utility": herc_score,
                        "utility_delta": herc_score - legacy_score,
                        "transition": transition,
                    }
                )
            transition_summary.append(
                {
                    "dataset": dataset,
                    "n_samples": len(legacy_answers),
                    "legacy_utility": mean(legacy_utility),
                    "herc_utility": mean(herc_utility),
                    "utility_delta_pp": (mean(herc_utility) - mean(legacy_utility)) * 100.0,
                    "improved": transitions["improved"],
                    "regressed": transitions["regressed"],
                    "unchanged": transitions["unchanged"],
                }
            )

        stats = read_jsonl(capture_run / "stats.jsonl")
        for layer in (2, 6, 16):
            aggregate: dict[str, list[float]] = defaultdict(list)
            role_in: Counter[str] = Counter()
            role_out: Counter[str] = Counter()
            for row in stats:
                incoming = [int(value) for value in row.get(f"layer_{layer}_evidence_reconcile_in_patch_indices", [])]
                outgoing = [int(value) for value in row.get(f"layer_{layer}_evidence_reconcile_out_patch_indices", [])]
                final_keep = {int(value) for value in row.get(f"layer_{layer}_keep_patch_indices", [])}
                if len(incoming) != len(outgoing):
                    raise ValueError(
                        f"{dataset} sample {row.get('sample_idx')} layer {layer}: unequal swap arrays"
                    )
                legacy_keep = (final_keep - set(incoming)) | set(outgoing)
                if final_keep and len(legacy_keep) != len(final_keep):
                    raise ValueError(
                        f"{dataset} sample {row.get('sample_idx')} layer {layer}: reconstructed keep count changed"
                    )

                raw_map = _index_value_map(row, layer, "importance")
                rank_map = _index_value_map(row, layer, "current_rank_score")
                layout = row.get("visual_token_layout", {}) or {}
                role_in.update(_role_counts(incoming, layout))
                role_out.update(_role_counts(outgoing, layout))
                aggregate["swap_count"].append(float(len(incoming)))
                aggregate["set_difference"].append(float(len(final_keep.symmetric_difference(legacy_keep))))
                aggregate["jaccard"].append(
                    float(len(final_keep & legacy_keep) / len(final_keep | legacy_keep))
                    if final_keep or legacy_keep
                    else 1.0
                )
                aggregate["incoming_raw"].extend(raw_map[index] for index in incoming if index in raw_map)
                aggregate["outgoing_raw"].extend(raw_map[index] for index in outgoing if index in raw_map)
                aggregate["incoming_rank"].extend(rank_map[index] for index in incoming if index in rank_map)
                aggregate["outgoing_rank"].extend(rank_map[index] for index in outgoing if index in rank_map)
                for field in (
                    "evidence_c_reference_survival_mass_mean",
                    "evidence_c_reference_survival_mass_min",
                    "evidence_c_reference_survival_mass_max",
                ):
                    value = row.get(f"layer_{layer}_{field}")
                    if value is not None:
                        aggregate[field].append(float(value))

                detail_rows.append(
                    {
                        "dataset": dataset,
                        "sample_idx": row.get("sample_idx"),
                        "question_id": row.get("question_id"),
                        "layer": layer,
                        "swap_count": len(incoming),
                        "legacy_final_jaccard": aggregate["jaccard"][-1],
                        "incoming_indices": json.dumps(incoming),
                        "outgoing_indices": json.dumps(outgoing),
                    }
                )

            summary_rows.append(
                {
                    "dataset": dataset,
                    "layer": layer,
                    "n_samples": len(stats),
                    "mean_swap_count": mean(aggregate["swap_count"]),
                    "mean_set_difference": mean(aggregate["set_difference"]),
                    "mean_jaccard": mean(aggregate["jaccard"]),
                    "mean_incoming_raw": mean(aggregate["incoming_raw"]),
                    "mean_outgoing_raw": mean(aggregate["outgoing_raw"]),
                    "mean_incoming_rank": mean(aggregate["incoming_rank"]),
                    "mean_outgoing_rank": mean(aggregate["outgoing_rank"]),
                    "incoming_roles": json.dumps(dict(sorted(role_in.items())), sort_keys=True),
                    "outgoing_roles": json.dumps(dict(sorted(role_out.items())), sort_keys=True),
                    "survival_mass_mean": mean(aggregate["evidence_c_reference_survival_mass_mean"]),
                    "survival_mass_min_mean": mean(aggregate["evidence_c_reference_survival_mass_min"]),
                    "survival_mass_max_mean": mean(aggregate["evidence_c_reference_survival_mass_max"]),
                }
            )

    gate = {
        "name": "forensic_behavior_equivalence",
        "passed": all(check["exact"] for check in answer_checks.values()),
        "answer_checks": answer_checks,
        "criteria": "all answer JSON rows exactly match the existing HERC v3 prefix runs",
    }
    write_csv(output_dir / "forensic_summary.csv", summary_rows)
    write_csv(output_dir / "forensic_sample_details.csv", detail_rows)
    if transition_rows:
        write_csv(output_dir / "correctness_transitions.csv", transition_rows)
        write_csv(output_dir / "correctness_transition_summary.csv", transition_summary)
    write_json(output_dir / "gate.json", gate)
    write_markdown_table(output_dir / "forensic_summary.md", "NLCR Stage 0 Forensic Summary", summary_rows)
    return gate


class UtilityScorer:
    """Per-sample utility under the protocol declared by the NLCR plan."""

    def __init__(self, dataset: str) -> None:
        self.dataset = dataset
        if dataset == "gqa":
            self.data = read_json(REPO_ROOT / "entropy_exp/datasets/gqa/testdev_balanced_questions.json")
        elif dataset == "textvqa":
            annotations = read_json(
                REPO_ROOT / "entropy_exp/eval_questions/textvqa/TextVQA_0.5.1_val.json"
            )["data"]
            self.data = {
                (str(item["image_id"]), str(item["question"]).lower()): item
                for item in annotations
            }
            self.evaluator = load_textvqa_evaluator()
        elif dataset == "pope":
            questions = read_jsonl(REPO_ROOT / "entropy_exp/eval_questions/pope/llava_pope_test.jsonl")
            self.data = {}
            labels_by_category: dict[str, list[int]] = {}
            for path in sorted((REPO_ROOT / "entropy_exp/datasets/pope/coco").glob("coco_pope_*.json")):
                category = path.name[10:-5]
                labels_by_category[category] = [
                    1 if item["label"] == "yes" else 0 for item in read_jsonl(path)
                ]
            category_offsets: Counter[str] = Counter()
            for item in questions:
                qid = str(item["question_id"])
                category = str(item["category"])
                offset = category_offsets[category]
                category_offsets[category] += 1
                self.data[qid] = labels_by_category[category][offset]
        elif dataset == "scienceqa":
            self.data = read_json(REPO_ROOT / "entropy_exp/eval_questions/scienceqa/problems.json")
        elif dataset == "mme":
            self.data = self._load_mme_ground_truth()
        else:
            raise ValueError(f"Unsupported utility dataset: {dataset}")

    @staticmethod
    def _load_mme_ground_truth() -> dict[tuple[str, str, str], str]:
        root = REPO_ROOT / "entropy_exp/datasets/MME_Benchmark_release_version"
        ground_truth: dict[tuple[str, str, str], str] = {}
        for category_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            qa_dir = (
                category_dir / "questions_answers_YN"
                if (category_dir / "questions_answers_YN").is_dir()
                else category_dir
            )
            for path in sorted(qa_dir.glob("*.txt")):
                for line in path.read_text(encoding="utf-8").splitlines():
                    question, answer = line.strip().split("\t")
                    ground_truth[(category_dir.name, path.name, question)] = answer.strip().lower()
        return ground_truth

    @staticmethod
    def _mme_prompt_key(answer: Mapping[str, Any]) -> tuple[str, str, str]:
        qid = str(answer["question_id"])
        category = qid.split("/", 1)[0]
        filename = Path(qid.split("/", 1)[-1]).stem + ".txt"
        prompt = str(answer.get("prompt", ""))
        prompt = prompt.replace("Answer the question using a single word or phrase.", "").strip()
        if "Please answer yes or no." not in prompt:
            prompt += " Please answer yes or no."
        return category, filename, prompt

    @staticmethod
    def _normalize_yes_no(value: str) -> str:
        lowered = str(value).strip().lower()
        if lowered.startswith("yes"):
            return "yes"
        if lowered.startswith("no"):
            return "no"
        return "other"

    def score(self, answer: Mapping[str, Any]) -> float:
        qid = str(answer["question_id"])
        prediction = str(answer.get("text", ""))
        if self.dataset == "gqa":
            return float(
                normalize_open_answer(prediction)
                == normalize_open_answer(self.data[qid].get("answer", ""))
            )
        if self.dataset == "textvqa":
            question = textvqa_prompt_processor(str(answer["prompt"]))
            annotation = self.data[(qid, question)]
            scores = self.evaluator._compute_answer_scores(annotation["answers"])
            parsed = self.evaluator.answer_processor(prediction)
            return float(scores.get(parsed, 0.0))
        if self.dataset == "pope":
            return float(normalize_pope_answer(prediction) == self.data[qid])
        if self.dataset == "scienceqa":
            options = ["A", "B", "C", "D", "E"]
            parsed = parse_scienceqa_answer(prediction, options)
            return float(parsed == options[int(self.data[qid]["answer"])])
        if self.dataset == "mme":
            key = self._mme_prompt_key(answer)
            if key not in self.data:
                spaced_key = (key[0], key[1], key[2].replace(" Please answer", "  Please answer"))
                key = spaced_key
            return float(self._normalize_yes_no(prediction) == self.data[key])
        raise AssertionError(self.dataset)


def score_answers(dataset: str, answers: Sequence[Mapping[str, Any]]) -> list[float]:
    scorer = UtilityScorer(dataset)
    return [scorer.score(answer) for answer in answers]


def _metric_from_eval(run_dir: Path) -> tuple[str, float]:
    metrics = read_json(run_dir / "eval" / "summary.json")["metrics"]
    if _config_value(run_dir, ("_run_meta", "dataset")) in {"gqa", "textvqa", "scienceqa"}:
        return "accuracy", float(metrics["accuracy"])
    if _config_value(run_dir, ("_run_meta", "dataset")) == "pope":
        return "macro_f1", float(metrics["macro_f1"])
    return "overall_total_score", float(metrics["overall_total_score"])


def full_metric_map(full_reference_root: Path) -> dict[str, float]:
    """Read each dataset's raw denominator from evaluated Full runs."""
    metrics: dict[str, float] = {}
    for dataset in DATASETS:
        run_dir = one_run(full_reference_root, dataset, require_eval=True)
        _, value = _metric_from_eval(run_dir)
        if value <= 0.0:
            raise ValueError(f"{dataset}: Full baseline metric must be positive, got {value}")
        metrics[dataset] = value
    return metrics


def _timing_rows(run_dir: Path, warmup: int) -> list[dict[str, Any]]:
    rows = read_jsonl(run_dir / "stats.jsonl")
    measured = [
        row for index, row in enumerate(rows)
        if not bool(row.get("benchmark_is_warmup", index < warmup))
    ]
    return measured


def smoke_gate(
    *,
    legacy_root: Path,
    audit_root: Path,
    output_dir: Path,
    warmup: int,
    max_prefill_increase: float,
    max_peak_memory_gb: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    checks: list[bool] = []
    for dataset in ("pope", "textvqa", "mme"):
        legacy_run = one_run(legacy_root, dataset, name_contains="legacy")
        audit_run = one_run(audit_root, dataset, name_contains="audit")
        answers = compare_answers_exact(legacy_run / "answers.jsonl", audit_run / "answers.jsonl")
        legacy_stats = _timing_rows(legacy_run, warmup)
        audit_stats = _timing_rows(audit_run, warmup)
        legacy_prefill = mean(row["prefill_time_ms"] for row in legacy_stats)
        audit_prefill = mean(row["prefill_time_ms"] for row in audit_stats)
        legacy_peak = mean(row["peak_memory_gb"] for row in legacy_stats)
        audit_peak = mean(row["peak_memory_gb"] for row in audit_stats)
        prefill_increase = audit_prefill / legacy_prefill - 1.0 if legacy_prefill else math.inf
        peak_delta = audit_peak - legacy_peak
        cf_rows = [row for row in audit_stats if row.get("layer_2_cf_mode") == "audit_only"]
        passed = (
            answers["exact"]
            and len(cf_rows) == len(audit_stats)
            and all(not row.get("layer_2_cf_bypassed", False) for row in cf_rows)
            and prefill_increase <= max_prefill_increase
            and peak_delta <= max_peak_memory_gb
        )
        checks.append(passed)
        rows.append(
            {
                "dataset": dataset,
                "samples": len(audit_stats),
                "answers_exact": answers["exact"],
                "legacy_prefill_ms": legacy_prefill,
                "audit_prefill_ms": audit_prefill,
                "prefill_increase_pct": prefill_increase * 100.0,
                "legacy_peak_gb": legacy_peak,
                "audit_peak_gb": audit_peak,
                "peak_delta_gb": peak_delta,
                "mean_cf_ms": mean(row.get("layer_2_cf_total_time_ms", 0.0) for row in cf_rows),
                "passed": passed,
            }
        )
    gate = {
        "name": "audit_smoke",
        "passed": all(checks),
        "criteria": {
            "answers_exact": True,
            "max_prefill_increase": max_prefill_increase,
            "max_peak_memory_delta_gb": max_peak_memory_gb,
        },
        "datasets": rows,
    }
    write_csv(output_dir / "smoke_summary.csv", rows)
    write_json(output_dir / "gate.json", gate)
    write_markdown_table(output_dir / "smoke_summary.md", "NLCR Audit Smoke", rows)
    return gate


def high_bypass_gate(*, root: Path, output_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    checks: list[bool] = []
    for dataset in ("pope", "textvqa", "mme"):
        legacy_run = one_run(root, dataset, name_contains="legacy")
        route_run = one_run(root, dataset, name_contains="route")
        answers = compare_answers_exact(legacy_run / "answers.jsonl", route_run / "answers.jsonl")
        legacy_stats = read_jsonl(legacy_run / "stats.jsonl")
        route_stats = read_jsonl(route_run / "stats.jsonl")
        validate_aligned_rows(legacy_stats, route_stats, label=f"{dataset} high stats")
        keep_exact = all(
            legacy.get("layer_2_keep_patch_indices") == route.get("layer_2_keep_patch_indices")
            for legacy, route in zip(legacy_stats, route_stats)
        )
        bypass_exact = all(
            route.get("layer_2_cf_bypassed") is True
            and route.get("layer_2_cf_bypass_reason") == "high_budget_profile"
            and float(route.get("layer_2_cf_total_time_ms", 0.0)) == 0.0
            for route in route_stats
        )
        passed = answers["exact"] and keep_exact and bypass_exact
        checks.append(passed)
        rows.append(
            {
                "dataset": dataset,
                "samples": len(route_stats),
                "answers_exact": answers["exact"],
                "keep_indices_exact": keep_exact,
                "zero_cost_bypass": bypass_exact,
                "passed": passed,
            }
        )
    gate = {"name": "high_exact_bypass", "passed": all(checks), "datasets": rows}
    write_csv(output_dir / "high_bypass_summary.csv", rows)
    write_json(output_dir / "gate.json", gate)
    return gate


def _candidate_runs(root: Path, dataset: str) -> dict[str, Path]:
    return {
        "legacy_scnd": one_run(root, dataset, name_contains="audit"),
        "sparsevlm_topk": one_run(root, dataset, name_contains="force_topk"),
        "native_maxmin": one_run(root, dataset, name_contains="force_maxmin"),
    }


def headroom_gate(
    *,
    root: Path,
    output_dir: Path,
    full_reference_root: Path | None,
    oracle_gain_pp: float,
    comparator_accuracy: float,
) -> dict[str, Any]:
    sample_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    comparator_correct = 0
    comparator_pairs = 0
    nonnegative_tasks = 0
    routed_total_gain = 0.0
    legacy_means: dict[str, float] = {}
    oracle_means: dict[str, float] = {}
    routed_means: dict[str, float] = {}
    full_means: dict[str, float] = {}

    for dataset in DATASETS:
        dataset_comparator_correct = 0
        dataset_comparator_pairs = 0
        runs = _candidate_runs(root, dataset)
        answers = {candidate: read_jsonl(run / "answers.jsonl") for candidate, run in runs.items()}
        stats = read_jsonl(runs["legacy_scnd"] / "stats.jsonl")
        validate_aligned_rows(*answers.values(), stats, label=f"{dataset} candidate alignment")
        utilities = {
            candidate: score_answers(dataset, candidate_answers)
            for candidate, candidate_answers in answers.items()
        }
        if full_reference_root is not None:
            full_run = one_run(full_reference_root, dataset)
            full_answers = read_jsonl(full_run / "answers.jsonl")[: len(stats)]
            validate_aligned_rows(answers["legacy_scnd"], full_answers, label=f"{dataset} full alignment")
            full_means[dataset] = mean(score_answers(dataset, full_answers))

        oracle_values: list[float] = []
        routed_values: list[float] = []
        route_counts: Counter[str] = Counter()
        for index, row in enumerate(stats):
            route = str(row.get("layer_2_cf_routed_candidate", "legacy_scnd"))
            if route not in CANDIDATES:
                raise ValueError(f"{dataset} row {index}: invalid routed candidate {route!r}")
            route_counts[route] += 1
            values = {candidate: utilities[candidate][index] for candidate in CANDIDATES}
            oracle = max(values.values())
            routed = values[route]
            oracle_values.append(oracle)
            routed_values.append(routed)

            has_counterfactual_errors = all(
                f"layer_2_cf_error_max_{CF_SUFFIX[candidate]}" in row
                and f"layer_2_cf_error_sum_{CF_SUFFIX[candidate]}" in row
                for candidate in CANDIDATES
            )
            errors = (
                {
                    candidate: (
                        float(row[f"layer_2_cf_error_max_{CF_SUFFIX[candidate]}"]),
                        float(row[f"layer_2_cf_error_sum_{CF_SUFFIX[candidate]}"]),
                        CANDIDATES.index(candidate),
                    )
                    for candidate in CANDIDATES
                }
                if has_counterfactual_errors
                else {}
            )
            if not has_counterfactual_errors and len(set(values.values())) != 1:
                raise ValueError(
                    f"{dataset} row {index}: candidate utility differs without visual counterfactual errors"
                )
            for left_index, left in enumerate(CANDIDATES):
                for right in CANDIDATES[left_index + 1 :]:
                    if values[left] == values[right]:
                        continue
                    predicted = min((left, right), key=lambda candidate: errors[candidate])
                    actual = left if values[left] > values[right] else right
                    comparator_pairs += 1
                    comparator_correct += int(predicted == actual)
                    dataset_comparator_pairs += 1
                    dataset_comparator_correct += int(predicted == actual)

            sample_rows.append(
                {
                    "dataset": dataset,
                    "sample_idx": index,
                    "question_id": row.get("question_id"),
                    "legacy_utility": values["legacy_scnd"],
                    "topk_utility": values["sparsevlm_topk"],
                    "maxmin_utility": values["native_maxmin"],
                    "oracle_utility": oracle,
                    "routed_candidate": route,
                    "routed_utility": routed,
                    "routed_gain": routed - values["legacy_scnd"],
                }
            )

        legacy_mean = mean(utilities["legacy_scnd"])
        oracle_mean = mean(oracle_values)
        routed_mean = mean(routed_values)
        legacy_means[dataset] = legacy_mean
        oracle_means[dataset] = oracle_mean
        routed_means[dataset] = routed_mean
        routed_gain = routed_mean - legacy_mean
        full_mean = full_means.get(dataset)
        routed_total_gain += routed_gain
        nonnegative_tasks += int(routed_gain >= 0.0)
        dataset_rows.append(
            {
                "dataset": dataset,
                "n_samples": len(stats),
                "legacy_utility": legacy_mean,
                "topk_utility": mean(utilities["sparsevlm_topk"]),
                "maxmin_utility": mean(utilities["native_maxmin"]),
                "oracle_utility": oracle_mean,
                "oracle_gain_pp": (oracle_mean - legacy_mean) * 100.0,
                "routed_utility": routed_mean,
                "routed_gain_pp": routed_gain * 100.0,
                "full_utility": full_mean,
                "oracle_ret_gain_pp": (
                    None
                    if full_mean is None
                    else (oracle_mean - legacy_mean) / full_mean * 100.0
                ),
                "routed_ret_gain_pp": (
                    None if full_mean is None else routed_gain / full_mean * 100.0
                ),
                "comparator_correct_pairs": dataset_comparator_correct,
                "comparator_utility_different_pairs": dataset_comparator_pairs,
                "comparator_accuracy": (
                    dataset_comparator_correct / dataset_comparator_pairs
                    if dataset_comparator_pairs
                    else 0.0
                ),
                "route_legacy": route_counts["legacy_scnd"],
                "route_topk": route_counts["sparsevlm_topk"],
                "route_maxmin": route_counts["native_maxmin"],
            }
        )

    if full_reference_root is None:
        oracle_mean_ret_gain = None
    else:
        if any(value <= 0.0 for value in full_means.values()):
            raise ValueError(f"Full baseline utility contains a non-positive denominator: {full_means}")
        oracle_mean_ret_gain = mean(
            (oracle_means[dataset] - legacy_means[dataset]) / full_means[dataset] * 100.0
            for dataset in DATASETS
        )
    comparator_rate = comparator_correct / comparator_pairs if comparator_pairs else 0.0
    aggregate_routed_gain = routed_total_gain / len(DATASETS)
    gate = {
        "name": "candidate_headroom",
        "passed": (
            oracle_mean_ret_gain is not None
            and oracle_mean_ret_gain >= oracle_gain_pp
            and comparator_rate > comparator_accuracy
            and aggregate_routed_gain > 0.0
            and nonnegative_tasks >= 4
        ),
        "oracle_mean_ret_gain_pp": oracle_mean_ret_gain,
        "comparator_accuracy": comparator_rate,
        "comparator_correct_pairs": comparator_correct,
        "comparator_utility_different_pairs": comparator_pairs,
        "simulated_route_aggregate_utility_gain": aggregate_routed_gain,
        "nonnegative_tasks": nonnegative_tasks,
        "criteria": {
            "oracle_mean_ret_gain_pp_min": oracle_gain_pp,
            "comparator_accuracy_strictly_greater_than": comparator_accuracy,
            "simulated_route_aggregate_utility_gain_strictly_positive": True,
            "minimum_nonnegative_tasks": 4,
            "full_reference_required": True,
        },
        "full_reference_available": full_reference_root is not None,
    }
    write_csv(output_dir / "headroom_dataset_summary.csv", dataset_rows)
    write_csv(output_dir / "headroom_sample_details.csv", sample_rows)
    write_json(output_dir / "gate.json", gate)
    write_markdown_table(output_dir / "headroom_summary.md", "NLCR Candidate Headroom", dataset_rows)
    return gate


def route_gate(
    *,
    legacy_root: Path,
    route_root: Path,
    output_dir: Path,
    full_metric_json: Path | None,
    full_reference_root: Path | None,
    mean_ret_gain_pp: float,
) -> dict[str, Any]:
    if (full_metric_json is None) == (full_reference_root is None):
        raise ValueError("Exactly one of full_metric_json and full_reference_root is required")
    full_metrics = (
        read_json(full_metric_json)
        if full_metric_json is not None
        else full_metric_map(full_reference_root)
    )
    rows: list[dict[str, Any]] = []
    all_raw_nonnegative = True
    ret_gains: list[float] = []
    all_keep_counts = True
    for dataset in DATASETS:
        legacy_run = one_run(legacy_root, dataset, name_contains="audit", require_eval=True)
        route_run = one_run(route_root, dataset, require_eval=True)
        legacy_metric_name, legacy_metric = _metric_from_eval(legacy_run)
        route_metric_name, route_metric = _metric_from_eval(route_run)
        if legacy_metric_name != route_metric_name:
            raise ValueError(f"{dataset}: metric mismatch {legacy_metric_name} vs {route_metric_name}")
        denominator = float(full_metrics[dataset])
        raw_gain = route_metric - legacy_metric
        ret_gain = raw_gain / denominator * 100.0
        ret_gains.append(ret_gain)
        all_raw_nonnegative &= raw_gain >= 0.0

        legacy_stats = read_jsonl(legacy_run / "stats.jsonl")
        route_stats = read_jsonl(route_run / "stats.jsonl")
        validate_aligned_rows(legacy_stats, route_stats, label=f"{dataset} route stats")
        keep_counts_equal = all(
            all(
                int(left[f"layer_{layer}_after"]) == int(right[f"layer_{layer}_after"])
                for layer in (2, 6, 16)
            )
            for left, right in zip(legacy_stats, route_stats)
        )
        all_keep_counts &= keep_counts_equal
        route_counts = Counter(
            str(row.get("layer_2_cf_selected_candidate", "legacy_scnd")) for row in route_stats
        )
        rows.append(
            {
                "dataset": dataset,
                "metric": legacy_metric_name,
                "legacy_raw": legacy_metric,
                "route_raw": route_metric,
                "raw_gain": raw_gain,
                "ret_gain_pp": ret_gain,
                "keep_counts_equal": keep_counts_equal,
                "route_legacy": route_counts["legacy_scnd"],
                "route_topk": route_counts["sparsevlm_topk"],
                "route_maxmin": route_counts["native_maxmin"],
            }
        )
    average_ret_gain = mean(ret_gains)
    gate = {
        "name": "prefix_route",
        "passed": all_raw_nonnegative and all_keep_counts and average_ret_gain >= mean_ret_gain_pp,
        "all_raw_metrics_nonnegative": all_raw_nonnegative,
        "all_keep_counts_equal": all_keep_counts,
        "mean_ret_gain_pp": average_ret_gain,
        "criteria": {
            "all_raw_metrics_nonnegative": True,
            "all_keep_counts_equal": True,
            "mean_ret_gain_pp_min": mean_ret_gain_pp,
        },
    }
    write_csv(output_dir / "route_dataset_summary.csv", rows)
    write_json(output_dir / "gate.json", gate)
    write_markdown_table(output_dir / "route_summary.md", "NLCR Prefix Route Gate", rows)
    return gate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    forensic = subparsers.add_parser("forensic")
    forensic.add_argument("--capture-root", type=Path, required=True)
    forensic.add_argument("--reference-root", type=Path, required=True)
    forensic.add_argument("--legacy-root", type=Path)
    forensic.add_argument("--output-dir", type=Path, required=True)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--legacy-root", type=Path, required=True)
    smoke.add_argument("--audit-root", type=Path, required=True)
    smoke.add_argument("--output-dir", type=Path, required=True)
    smoke.add_argument("--warmup", type=int, default=2)
    smoke.add_argument("--max-prefill-increase", type=float, default=0.15)
    smoke.add_argument("--max-peak-memory-gb", type=float, default=1.0)

    high = subparsers.add_parser("high-bypass")
    high.add_argument("--root", type=Path, required=True)
    high.add_argument("--output-dir", type=Path, required=True)

    headroom = subparsers.add_parser("headroom")
    headroom.add_argument("--root", type=Path, required=True)
    headroom.add_argument("--output-dir", type=Path, required=True)
    headroom.add_argument("--full-reference-root", type=Path)
    headroom.add_argument("--oracle-gain-pp", type=float, default=0.30)
    headroom.add_argument("--comparator-accuracy", type=float, default=0.55)

    route = subparsers.add_parser("route-gate")
    route.add_argument("--legacy-root", type=Path, required=True)
    route.add_argument("--route-root", type=Path, required=True)
    route.add_argument("--output-dir", type=Path, required=True)
    denominator = route.add_mutually_exclusive_group(required=True)
    denominator.add_argument("--full-metric-json", type=Path)
    denominator.add_argument("--full-reference-root", type=Path)
    route.add_argument("--mean-ret-gain-pp", type=float, default=0.30)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "forensic":
        gate = forensic_report(
            capture_root=args.capture_root,
            reference_root=args.reference_root,
            legacy_root=args.legacy_root,
            output_dir=args.output_dir,
        )
    elif args.command == "smoke":
        gate = smoke_gate(
            legacy_root=args.legacy_root,
            audit_root=args.audit_root,
            output_dir=args.output_dir,
            warmup=args.warmup,
            max_prefill_increase=args.max_prefill_increase,
            max_peak_memory_gb=args.max_peak_memory_gb,
        )
    elif args.command == "high-bypass":
        gate = high_bypass_gate(root=args.root, output_dir=args.output_dir)
    elif args.command == "headroom":
        gate = headroom_gate(
            root=args.root,
            output_dir=args.output_dir,
            full_reference_root=args.full_reference_root,
            oracle_gain_pp=args.oracle_gain_pp,
            comparator_accuracy=args.comparator_accuracy,
        )
    elif args.command == "route-gate":
        gate = route_gate(
            legacy_root=args.legacy_root,
            route_root=args.route_root,
            output_dir=args.output_dir,
            full_metric_json=args.full_metric_json,
            full_reference_root=args.full_reference_root,
            mean_ret_gain_pp=args.mean_ret_gain_pp,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(gate, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if gate["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
