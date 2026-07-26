from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "algo_compare" / "scripts" / "run_official_batch.py"
SPEC = importlib.util.spec_from_file_location("run_official_batch", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
batch = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = batch
SPEC.loader.exec_module(batch)


class OfficialBatchPlanTests(unittest.TestCase):
    def test_max_samples_limits_expected_count_and_command(self):
        plan = {
            "python_bin": "/tmp/python",
            "model_path": "/tmp/model",
            "model_name": "llava-v1.6-vicuna-7b",
            "output_root": "/tmp/output",
            "max_samples": 3,
            "methods": {
                "divprune": {"variant": "divprune_next_dynamic"},
                "cdpruner": {
                    "variant": "cdpruner_next_per_crop_dpp",
                    "llava_next_compat": "off",
                    "padding_diagnostics": True,
                },
            },
            "budgets": {
                "high": {
                    "one_shot_tokens": 126,
                    "divprune_subset_ratio": 0.21875,
                }
            },
            "datasets": [{"name": "gqa", "samples": 12578}],
        }

        jobs = batch.build_jobs(plan)

        self.assertEqual(len(jobs), 2)
        self.assertTrue(all(job.expected_samples == 3 for job in jobs))
        self.assertTrue(all("--max-samples" in job.command for job in jobs))
        self.assertTrue(all(job.command[job.command.index("--max-samples") + 1] == "3" for job in jobs))
        self.assertTrue(all(job.command[job.command.index("--conv-mode") + 1] == "vicuna_v1" for job in jobs))
        cdpruner_job = next(job for job in jobs if job.method == "cdpruner")
        self.assertEqual(
            cdpruner_job.command[cdpruner_job.command.index("--cdpruner-llava-next-compat") + 1],
            "off",
        )
        self.assertIn("--cdpruner-padding-diagnostics", cdpruner_job.command)

    def test_non_positive_max_samples_is_rejected(self):
        plan = {
            "python_bin": "/tmp/python",
            "model_path": "/tmp/model",
            "model_name": "model",
            "output_root": "/tmp/output",
            "max_samples": 0,
            "methods": {"cdpruner": {"variant": "cd"}},
            "budgets": {"high": {"one_shot_tokens": 126}},
            "datasets": [{"name": "gqa", "samples": 2}],
        }
        with self.assertRaisesRegex(ValueError, "max_samples must be positive"):
            batch.build_jobs(plan)


if __name__ == "__main__":
    unittest.main()
