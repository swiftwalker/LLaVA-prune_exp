from __future__ import annotations

import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path


SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

from adaptive_hparam_report import main


class AdaptiveHparamReportTests(unittest.TestCase):
    def _write_csv(self, path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_report_outputs_best_tables_and_schedule_candidates(self):
        fieldnames = [
            "dataset",
            "strategy",
            "run_mode",
            "run_name",
            "run_dir",
            "effective_prune_layers",
            "prune_ratio",
            "primary_metric_value",
            "adaptive_grid_size",
            "adaptive_high_ratio",
            "pruning_config_json",
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            adaptive_summary = root / "adaptive.csv"
            gqa_references = root / "gqa_refs.csv"
            textvqa_references = root / "textvqa_refs.csv"
            output_dir = root / "report"

            adaptive_rows = [
                {
                    "dataset": "gqa",
                    "strategy": "sparsevlm_adaptive_stratified",
                    "run_mode": "prune",
                    "run_name": "gqa_adaptive_l1_r0p5_g6_a0p55",
                    "run_dir": "/tmp/gqa_adaptive_l1_r0p5_g6_a0p55",
                    "effective_prune_layers": "[1]",
                    "prune_ratio": "[0.5]",
                    "primary_metric_value": 61.20,
                    "adaptive_grid_size": 6,
                    "adaptive_high_ratio": 0.55,
                    "pruning_config_json": "",
                },
                {
                    "dataset": "gqa",
                    "strategy": "sparsevlm_adaptive_stratified",
                    "run_mode": "prune",
                    "run_name": "gqa_adaptive_l1_r0p6_g6_a0p4",
                    "run_dir": "/tmp/gqa_adaptive_l1_r0p6_g6_a0p4",
                    "effective_prune_layers": "[1]",
                    "prune_ratio": "[0.6]",
                    "primary_metric_value": 61.05,
                    "adaptive_grid_size": 6,
                    "adaptive_high_ratio": 0.4,
                    "pruning_config_json": "",
                },
                {
                    "dataset": "gqa",
                    "strategy": "sparsevlm_adaptive_stratified",
                    "run_mode": "prune",
                    "run_name": "gqa_adaptive_l1_r0p7_g8_a0p85",
                    "run_dir": "/tmp/gqa_adaptive_l1_r0p7_g8_a0p85",
                    "effective_prune_layers": "[1]",
                    "prune_ratio": "[0.7]",
                    "primary_metric_value": 60.80,
                    "adaptive_grid_size": 8,
                    "adaptive_high_ratio": 0.85,
                    "pruning_config_json": "",
                },
                {
                    "dataset": "textvqa",
                    "strategy": "sparsevlm_adaptive_stratified",
                    "run_mode": "prune",
                    "run_name": "textvqa_adaptive_l3_r0p2_g4_a0p7",
                    "run_dir": "/tmp/textvqa_adaptive_l3_r0p2_g4_a0p7",
                    "effective_prune_layers": "[3]",
                    "prune_ratio": "[0.2]",
                    "primary_metric_value": 58.31,
                    "adaptive_grid_size": 4,
                    "adaptive_high_ratio": 0.7,
                    "pruning_config_json": "",
                },
            ]
            self._write_csv(adaptive_summary, fieldnames, adaptive_rows)

            gqa_reference_rows = [
                {
                    "dataset": "gqa",
                    "strategy": "attn_score",
                    "run_mode": "baseline",
                    "run_name": "gqa_baseline_demo",
                    "run_dir": "/tmp/gqa_baseline_demo",
                    "effective_prune_layers": "",
                    "prune_ratio": 0.0,
                    "primary_metric_value": 60.50,
                    "adaptive_grid_size": "",
                    "adaptive_high_ratio": "",
                    "pruning_config_json": "",
                },
                {
                    "dataset": "gqa",
                    "strategy": "random",
                    "run_mode": "prune",
                    "run_name": "gqa_random_l1_r0p5",
                    "run_dir": "/tmp/gqa_random_l1_r0p5",
                    "effective_prune_layers": "[1]",
                    "prune_ratio": "[0.5]",
                    "primary_metric_value": 58.10,
                    "adaptive_grid_size": "",
                    "adaptive_high_ratio": "",
                    "pruning_config_json": "",
                },
                {
                    "dataset": "gqa",
                    "strategy": "random",
                    "run_mode": "prune",
                    "run_name": "gqa_random_l1_r0p6",
                    "run_dir": "/tmp/gqa_random_l1_r0p6",
                    "effective_prune_layers": "[1]",
                    "prune_ratio": "[0.6]",
                    "primary_metric_value": 57.60,
                    "adaptive_grid_size": "",
                    "adaptive_high_ratio": "",
                    "pruning_config_json": "",
                },
                {
                    "dataset": "gqa",
                    "strategy": "random",
                    "run_mode": "prune",
                    "run_name": "gqa_random_l1_r0p7",
                    "run_dir": "/tmp/gqa_random_l1_r0p7",
                    "effective_prune_layers": "[1]",
                    "prune_ratio": "[0.7]",
                    "primary_metric_value": 57.20,
                    "adaptive_grid_size": "",
                    "adaptive_high_ratio": "",
                    "pruning_config_json": "",
                },
                {
                    "dataset": "gqa",
                    "strategy": "sparsevlm",
                    "run_mode": "prune",
                    "run_name": "gqa_sparsevlm_l1_r0p5",
                    "run_dir": "/tmp/gqa_sparsevlm_l1_r0p5",
                    "effective_prune_layers": "[1]",
                    "prune_ratio": "[0.5]",
                    "primary_metric_value": 60.90,
                    "adaptive_grid_size": "",
                    "adaptive_high_ratio": "",
                    "pruning_config_json": "",
                },
                {
                    "dataset": "gqa",
                    "strategy": "sparsevlm",
                    "run_mode": "prune",
                    "run_name": "gqa_sparsevlm_l1_r0p6",
                    "run_dir": "/tmp/gqa_sparsevlm_l1_r0p6",
                    "effective_prune_layers": "[1]",
                    "prune_ratio": "[0.6]",
                    "primary_metric_value": 60.60,
                    "adaptive_grid_size": "",
                    "adaptive_high_ratio": "",
                    "pruning_config_json": "",
                },
                {
                    "dataset": "gqa",
                    "strategy": "sparsevlm",
                    "run_mode": "prune",
                    "run_name": "gqa_sparsevlm_l1_r0p7",
                    "run_dir": "/tmp/gqa_sparsevlm_l1_r0p7",
                    "effective_prune_layers": "[1]",
                    "prune_ratio": "[0.7]",
                    "primary_metric_value": 60.40,
                    "adaptive_grid_size": "",
                    "adaptive_high_ratio": "",
                    "pruning_config_json": "",
                },
            ]
            self._write_csv(gqa_references, fieldnames, gqa_reference_rows)

            textvqa_reference_rows = [
                {
                    "dataset": "textvqa",
                    "strategy": "baseline",
                    "run_mode": "baseline",
                    "run_name": "textvqa_baseline_demo",
                    "run_dir": "/tmp/textvqa_baseline_demo",
                    "effective_prune_layers": "",
                    "prune_ratio": 0.0,
                    "primary_metric_value": 58.20,
                    "adaptive_grid_size": "",
                    "adaptive_high_ratio": "",
                    "pruning_config_json": "",
                },
                {
                    "dataset": "textvqa",
                    "strategy": "random",
                    "run_mode": "prune",
                    "run_name": "textvqa_random_l3_r0p2",
                    "run_dir": "/tmp/textvqa_random_l3_r0p2",
                    "effective_prune_layers": "[3]",
                    "prune_ratio": "[0.2]",
                    "primary_metric_value": 57.10,
                    "adaptive_grid_size": "",
                    "adaptive_high_ratio": "",
                    "pruning_config_json": "",
                },
                {
                    "dataset": "textvqa",
                    "strategy": "sparsevlm",
                    "run_mode": "prune",
                    "run_name": "textvqa_sparsevlm_l3_r0p2",
                    "run_dir": "/tmp/textvqa_sparsevlm_l3_r0p2",
                    "effective_prune_layers": "[3]",
                    "prune_ratio": "[0.2]",
                    "primary_metric_value": 58.18,
                    "adaptive_grid_size": "",
                    "adaptive_high_ratio": "",
                    "pruning_config_json": "",
                },
            ]
            self._write_csv(textvqa_references, fieldnames, textvqa_reference_rows)

            rc = main(
                [
                    "--adaptive-summary",
                    str(adaptive_summary),
                    "--reference-summary",
                    str(gqa_references),
                    str(textvqa_references),
                    "--output-dir",
                    str(output_dir),
                ]
            )
            self.assertEqual(rc, 0)

            best_overall = output_dir / "best_overall_by_dataset.csv"
            best_by_ratio = output_dir / "best_by_ratio.csv"
            high_ratio = output_dir / "high_ratio_slice_report.csv"
            schedule = output_dir / "alpha_schedule_candidates.csv"
            report = output_dir / "report.md"

            for path in (best_overall, best_by_ratio, high_ratio, schedule, report):
                self.assertTrue(path.is_file(), path)

            with best_overall.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["dataset"] for row in rows}, {"gqa", "textvqa"})
            gqa_row = next(row for row in rows if row["dataset"] == "gqa")
            self.assertEqual(gqa_row["best_grid_size"], "6")
            self.assertEqual(gqa_row["best_high_ratio"], "0.55")

            with schedule.open("r", encoding="utf-8", newline="") as handle:
                schedule_rows = list(csv.DictReader(handle))
            self.assertEqual(len(schedule_rows), 1)
            self.assertEqual(schedule_rows[0]["dataset"], "gqa")
            self.assertEqual(schedule_rows[0]["layer"], "1")
            self.assertEqual(schedule_rows[0]["recommend_ratio_conditioned_alpha_schedule"], "True")

            report_text = report.read_text(encoding="utf-8")
            self.assertIn("best overall = layer 1, ratio 0.5, G=6, alpha=0.55", report_text)
            self.assertIn("`gqa` layer 1: recommend", report_text)


if __name__ == "__main__":
    unittest.main()
