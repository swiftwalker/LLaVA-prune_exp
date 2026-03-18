import json
import os
import sys
import tempfile
import unittest

import yaml

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

from interface import compute_batch_llm_flops, compute_llm_flops


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(REPO_ROOT, "entropy_exp", "models", "llava-v1.5-7b")


class BatchFlopsTests(unittest.TestCase):
    def _make_temp_config(self, tmpdir: str) -> str:
        path = os.path.join(tmpdir, "default.yaml")
        payload = {
            "model": {
                "model_path": MODEL_PATH,
                "tokenizer_path": None,
                "conv_mode": "vicuna_v1",
            },
            "input": {
                "image_token_len": 576,
            },
            "output": {
                "base_dir": os.path.join(tmpdir, "outputs"),
            },
            "evaluation": {
                "default_length_stat": "mean",
                "include_decode": False,
            },
        }
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle)
        return path

    def _make_stats_json(self, tmpdir: str) -> str:
        path = os.path.join(tmpdir, "summary.json")
        payload = {
            "raw_text_token_len_mean": 20.0,
            "raw_text_token_len_median": 20.0,
            "raw_text_token_len_p25": 18.0,
            "raw_text_token_len_p75": 22.0,
            "template_overhead_tokens": 35,
            "image_token_len": 576,
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return path

    def _make_run_dir(self, runs_root: str, run_name: str, run_mode: str, strategy: str) -> str:
        run_dir = os.path.join(runs_root, run_name)
        os.makedirs(run_dir, exist_ok=True)
        payload = {
            "_run_meta": {
                "run_mode": run_mode,
                "dataset": run_name.split("_")[0],
                "timestamp": "20260317_000000",
                "run_dir": run_dir,
            },
            "pruning": {
                "strategy": strategy,
                "layer_selection": "fixed",
                "prune_layers": [2],
                "prune_ratio": [0.2],
                "entropy": {"dynamic_ratio": False, "dynamic_scale": 0.5, "max_prune_ratio": 0.9},
            },
        }
        with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle)
        return run_dir

    def test_compute_llm_flops_requires_run_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._make_temp_config(tmpdir)
            stats_json = self._make_stats_json(tmpdir)
            with self.assertRaises(ValueError):
                compute_llm_flops(config_path=config_path, stats_json=stats_json)

    def test_compute_batch_llm_flops_filters_dataset_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._make_temp_config(tmpdir)
            stats_json = self._make_stats_json(tmpdir)
            runs_root = os.path.join(tmpdir, "runs")
            os.makedirs(runs_root, exist_ok=True)
            self._make_run_dir(runs_root, "mme_attn_score_20260317_1", "prune", "attn_score")
            self._make_run_dir(runs_root, "gqa_attn_score_20260317_1", "prune", "attn_score")

            _, summary, per_run_rows = compute_batch_llm_flops(
                config_path=config_path,
                runs_root=runs_root,
                dataset="mme",
                stats_json=stats_json,
            )

            self.assertEqual(summary["matched_runs"], 1)
            self.assertEqual(len(per_run_rows), 1)
            self.assertEqual(per_run_rows[0]["dataset"], "mme")

    def test_compute_batch_llm_flops_scope_runs_processes_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._make_temp_config(tmpdir)
            stats_json = self._make_stats_json(tmpdir)
            runs_root = os.path.join(tmpdir, "runs")
            os.makedirs(runs_root, exist_ok=True)
            self._make_run_dir(runs_root, "mme_attn_score_20260317_1", "prune", "attn_score")
            self._make_run_dir(runs_root, "gqa_baseline_20260317_1", "baseline", "attn_score")

            _, summary, per_run_rows = compute_batch_llm_flops(
                config_path=config_path,
                runs_root=runs_root,
                scope="runs",
                stats_json=stats_json,
            )

            self.assertEqual(summary["matched_runs"], 2)
            self.assertEqual(len(per_run_rows), 2)


if __name__ == "__main__":
    unittest.main()
