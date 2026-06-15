"""Small benchmark helpers for local official-method wrappers."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch


def add_benchmark_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--benchmark", action="store_true", help="Write per-sample benchmark_stats.jsonl")
    parser.add_argument("--benchmark-stats-file", default=None, help="Optional benchmark stats JSONL path")
    parser.add_argument("--benchmark-warmup-samples", type=int, default=0, help="Samples to mark as warmup")
    parser.add_argument("--benchmark-run-label", default=None, help="Optional label stored in benchmark stats")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples for smoke/timing subsets")


def generated_sequences(output_ids: Any) -> torch.Tensor:
    if hasattr(output_ids, "sequences"):
        return output_ids.sequences
    if isinstance(output_ids, dict):
        return output_ids["sequences"]
    return output_ids


def continuation_sequences(output_ids: Any, input_ids: torch.Tensor) -> torch.Tensor:
    sequences = generated_sequences(output_ids)
    if sequences.shape[1] > input_ids.shape[1]:
        return sequences[:, input_ids.shape[1] :]
    return sequences


class BenchmarkRecorder:
    def __init__(
        self,
        *,
        enabled: bool,
        stats_file: Path,
        warmup_samples: int,
        method: str,
        dataset: str,
        run_label: str | None = None,
    ) -> None:
        self.enabled = enabled
        self.stats_file = stats_file
        self.warmup_samples = max(0, int(warmup_samples))
        self.method = method
        self.dataset = dataset
        self.run_label = run_label
        if self.enabled:
            self.stats_file.parent.mkdir(parents=True, exist_ok=True)
            self.stats_file.write_text("", encoding="utf-8")

    @classmethod
    def from_args(
        cls,
        args: argparse.Namespace,
        *,
        answers_file: Path,
        method: str,
        dataset: str,
    ) -> "BenchmarkRecorder":
        stats_file_value = getattr(args, "benchmark_stats_file", None)
        stats_file = Path(stats_file_value).expanduser() if stats_file_value else answers_file.with_name("benchmark_stats.jsonl")
        return cls(
            enabled=bool(getattr(args, "benchmark", False)),
            stats_file=stats_file,
            warmup_samples=int(getattr(args, "benchmark_warmup_samples", 0) or 0),
            method=method,
            dataset=dataset,
            run_label=getattr(args, "benchmark_run_label", None),
        )

    @staticmethod
    def _sync_cuda() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @staticmethod
    def _reset_peak_memory() -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @staticmethod
    def _peak_memory_gb() -> float | None:
        if not torch.cuda.is_available():
            return None
        return float(torch.cuda.max_memory_allocated()) / float(1024**3)

    @staticmethod
    def _allocated_memory_gb() -> float | None:
        if not torch.cuda.is_available():
            return None
        return float(torch.cuda.memory_allocated()) / float(1024**3)

    def start_sample(self) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        self._sync_cuda()
        self._reset_peak_memory()
        self._sync_cuda()
        return {"start_time": time.perf_counter()}

    def finish_sample(
        self,
        token: dict[str, Any] | None,
        *,
        sample_idx: int,
        question_id: Any,
        input_token_count: int,
        has_image: bool,
        num_generated_tokens: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled or token is None:
            return
        self._sync_cuda()
        elapsed_s = time.perf_counter() - float(token["start_time"])
        elapsed_ms = elapsed_s * 1000.0
        decode_tps = float(num_generated_tokens) / elapsed_s if elapsed_s > 0 else None
        record = {
            "sample_idx": sample_idx,
            "question_id": question_id,
            "dataset": self.dataset,
            "method": self.method,
            "run_label": self.run_label,
            "benchmark_enabled": True,
            "benchmark_is_warmup": sample_idx < self.warmup_samples,
            "benchmark_warmup_samples": self.warmup_samples,
            "input_token_count": input_token_count,
            "has_image": bool(has_image),
            "num_generated_tokens": int(num_generated_tokens),
            "prefill_time_ms": None,
            "decode_time_ms": None,
            "total_time_ms": elapsed_ms,
            "end_to_end_time_ms": elapsed_ms,
            "end_to_end_time": elapsed_s,
            "decode_tokens_per_second": decode_tps,
            "peak_memory_gb": self._peak_memory_gb(),
            "allocated_memory_gb": self._allocated_memory_gb(),
            "metadata": metadata or {},
        }
        with self.stats_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
