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


if __name__ == "__main__":
    unittest.main()
