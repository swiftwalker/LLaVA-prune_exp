import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

from summarize_results import build_csv_rows, extract_record, primary_metric


class SummarizeResultsPopeTests(unittest.TestCase):
    def test_primary_metric_uses_macro_f1(self):
        name, value = primary_metric("pope", {"macro_f1": 0.853})
        self.assertEqual(name, "macro_f1")
        self.assertAlmostEqual(value, 0.853)

    def test_primary_metric_falls_back_to_legacy_weighted_average(self):
        name, value = primary_metric("pope", {"weighted_average": {"f1_score": 0.812}})
        self.assertEqual(name, "macro_f1")
        self.assertAlmostEqual(value, 0.812)

    def test_primary_metric_uses_accuracy_for_textvqa_and_scienceqa(self):
        textvqa_name, textvqa_value = primary_metric("textvqa", {"accuracy": 58.7})
        scienceqa_name, scienceqa_value = primary_metric("scienceqa", {"accuracy": 77.2})
        self.assertEqual(textvqa_name, "accuracy")
        self.assertAlmostEqual(textvqa_value, 58.7)
        self.assertEqual(scienceqa_name, "accuracy")
        self.assertAlmostEqual(scienceqa_value, 77.2)

    def test_extract_record_and_csv_rows_use_pope_macro_f1(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "pope_masking_attn_score_l1_r0p6__demo"
            eval_dir = run_dir / "eval"
            eval_dir.mkdir(parents=True)

            config = {
                "_run_meta": {"dataset": "pope", "run_mode": "prune", "timestamp": "demo"},
                "pruning": {
                    "strategy": "masking_attn_score",
                    "layer_selection": "fixed",
                    "prune_layers": [1],
                    "prune_ratio": [0.6],
                    "v_token_num": 576,
                    "max_samples": None,
                    "entropy": {},
                },
            }
            metrics = {
                "popular": {"samples": 3000, "f1_score": 0.85},
                "adversarial": {"samples": 3000, "f1_score": 0.84},
                "random": {"samples": 2910, "f1_score": 0.87},
                "macro_f1": 0.8533333333333334,
            }

            (run_dir / "config.yaml").write_text(
                yaml.safe_dump(config, sort_keys=False),
                encoding="utf-8",
            )
            (eval_dir / "summary.json").write_text(
                json.dumps({"dataset": "pope", "metrics": metrics}, indent=2),
                encoding="utf-8",
            )

            record, skipped = extract_record(run_dir)
            self.assertIsNone(skipped)
            self.assertEqual(record["primary_metric_name"], "macro_f1")
            self.assertAlmostEqual(record["primary_metric_value"], metrics["macro_f1"])
            self.assertAlmostEqual(record["pope_macro_f1"], metrics["macro_f1"])

            rows = build_csv_rows([record])
            self.assertEqual(rows[0]["primary_metric_name"], "macro_f1")
            self.assertAlmostEqual(rows[0]["primary_metric_value"], metrics["macro_f1"])
            self.assertAlmostEqual(rows[0]["pope_macro_f1"], metrics["macro_f1"])

    def test_extract_record_preserves_tail_masking_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "gqa_tail_masking_attn_score_l3_r0p2__demo"
            eval_dir = run_dir / "eval"
            eval_dir.mkdir(parents=True)

            config = {
                "_run_meta": {"dataset": "gqa", "run_mode": "prune", "timestamp": "demo"},
                "pruning": {
                    "strategy": "tail_masking_attn_score",
                    "layer_selection": "fixed",
                    "prune_layers": [3],
                    "effective_prune_layers": [3, 4, 5],
                    "tail_start_layer": 3,
                    "prune_ratio": [0.2],
                    "v_token_num": 576,
                    "max_samples": None,
                    "entropy": {},
                },
            }
            metrics = {"accuracy": 62.5}

            (run_dir / "config.yaml").write_text(
                yaml.safe_dump(config, sort_keys=False),
                encoding="utf-8",
            )
            (eval_dir / "summary.json").write_text(
                json.dumps({"dataset": "gqa", "metrics": metrics}, indent=2),
                encoding="utf-8",
            )

            record, skipped = extract_record(run_dir)
            self.assertIsNone(skipped)
            self.assertEqual(record["effective_prune_layers"], [3, 4, 5])
            self.assertEqual(record["tail_start_layer"], 3)

            rows = build_csv_rows([record])
            self.assertEqual(rows[0]["effective_prune_layers"], "[3, 4, 5]")
            self.assertEqual(rows[0]["tail_start_layer"], 3)

    def test_extract_record_and_csv_rows_include_textvqa_and_scienceqa_accuracy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "textvqa_random_l1_r0p2__demo"
            eval_dir = run_dir / "eval"
            eval_dir.mkdir(parents=True)

            config = {
                "_run_meta": {"dataset": "textvqa", "run_mode": "prune", "timestamp": "demo"},
                "pruning": {
                    "strategy": "random",
                    "layer_selection": "fixed",
                    "prune_layers": [1],
                    "prune_ratio": [0.2],
                    "v_token_num": 576,
                    "max_samples": None,
                    "entropy": {},
                },
            }
            metrics = {"accuracy": 58.7}

            (run_dir / "config.yaml").write_text(
                yaml.safe_dump(config, sort_keys=False),
                encoding="utf-8",
            )
            (eval_dir / "summary.json").write_text(
                json.dumps({"dataset": "textvqa", "metrics": metrics}, indent=2),
                encoding="utf-8",
            )

            record, skipped = extract_record(run_dir)
            self.assertIsNone(skipped)
            self.assertEqual(record["primary_metric_name"], "accuracy")
            self.assertAlmostEqual(record["primary_metric_value"], 58.7)
            self.assertAlmostEqual(record["textvqa_accuracy"], 58.7)

            rows = build_csv_rows([record])
            self.assertAlmostEqual(rows[0]["textvqa_accuracy"], 58.7)
            self.assertEqual(rows[0]["scienceqa_accuracy"], "")

    def test_extract_record_prefers_run_meta_strategy_for_baseline_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "textvqa_baseline_l1_r0__demo"
            eval_dir = run_dir / "eval"
            eval_dir.mkdir(parents=True)

            config = {
                "_run_meta": {
                    "dataset": "textvqa",
                    "run_mode": "baseline",
                    "timestamp": "demo",
                    "strategy": "baseline",
                },
                "pruning": {
                    "strategy": "attn_score",
                    "layer_selection": "fixed",
                    "prune_layers": [1],
                    "prune_ratio": 0.0,
                    "v_token_num": 576,
                    "max_samples": None,
                    "entropy": {},
                },
            }
            metrics = {"accuracy": 58.2}

            (run_dir / "config.yaml").write_text(
                yaml.safe_dump(config, sort_keys=False),
                encoding="utf-8",
            )
            (eval_dir / "summary.json").write_text(
                json.dumps({"dataset": "textvqa", "metrics": metrics}, indent=2),
                encoding="utf-8",
            )

            record, skipped = extract_record(run_dir)
            self.assertIsNone(skipped)
            self.assertEqual(record["strategy"], "baseline")

            rows = build_csv_rows([record])
            self.assertEqual(rows[0]["strategy"], "baseline")

    def test_extract_record_and_csv_rows_include_adaptive_hparams(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "gqa_sparsevlm_adaptive_stratified_l1_r0p2__demo"
            eval_dir = run_dir / "eval"
            eval_dir.mkdir(parents=True)

            config = {
                "_run_meta": {
                    "dataset": "gqa",
                    "run_mode": "prune",
                    "timestamp": "demo",
                    "strategy": "sparsevlm_adaptive_stratified",
                },
                "pruning": {
                    "strategy": "sparsevlm_adaptive_stratified",
                    "layer_selection": "fixed",
                    "prune_layers": [1],
                    "prune_ratio": [0.2],
                    "v_token_num": 576,
                    "max_samples": None,
                    "entropy": {},
                    "sparsevlm_adaptive_stratified": {
                        "fallback_topk": 4,
                        "exclude_special_tokens": True,
                        "min_visual_tokens_after_prune": 16,
                        "grid_size": 8,
                        "high_ratio": 0.55,
                        "patch_per_row": 24,
                        "intra_stratum_mode": "random",
                    },
                },
            }
            metrics = {"accuracy": 61.2}

            (run_dir / "config.yaml").write_text(
                yaml.safe_dump(config, sort_keys=False),
                encoding="utf-8",
            )
            (eval_dir / "summary.json").write_text(
                json.dumps({"dataset": "gqa", "metrics": metrics}, indent=2),
                encoding="utf-8",
            )

            record, skipped = extract_record(run_dir)
            self.assertIsNone(skipped)
            self.assertEqual(record["adaptive_grid_size"], 8)
            self.assertAlmostEqual(record["adaptive_high_ratio"], 0.55)

            rows = build_csv_rows([record])
            self.assertEqual(rows[0]["adaptive_grid_size"], 8)
            self.assertAlmostEqual(rows[0]["adaptive_high_ratio"], 0.55)


if __name__ == "__main__":
    unittest.main()
