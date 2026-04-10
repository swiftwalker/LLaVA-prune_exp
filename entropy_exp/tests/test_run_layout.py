from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

from run_layout import build_run_dir, find_run_dirs, read_run_metadata


class RunLayoutTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.runs_dir = Path(self.temp_dir.name) / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_run(self, relative_dir: str, config: dict | None = None, with_answers: bool = True, with_capture: bool = False):
        run_dir = self.runs_dir / relative_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        if config is not None:
            (run_dir / "config.yaml").write_text(
                yaml.safe_dump(config, sort_keys=False),
                encoding="utf-8",
            )
        if with_answers:
            (run_dir / "answers.jsonl").write_text('{"ok": true}\n', encoding="utf-8")
        if with_capture:
            (run_dir / "captures.h5").write_text("", encoding="utf-8")
        return run_dir

    def test_find_run_dirs_supports_flat_and_nested_layouts(self):
        flat_name = "gqa_random_l1_r0p2__20260322_235451_349889"
        nested_name = "pope_baseline_l2-3_r0__20260322_134126_631672"
        self._write_run(flat_name, config={"_run_meta": {"dataset": "gqa", "run_mode": "prune"}, "pruning": {"strategy": "random"}})
        self._write_run(
            f"baseline/pope/{nested_name}",
            config={"_run_meta": {"dataset": "pope", "run_mode": "baseline"}, "pruning": {"strategy": "attn_score"}},
            with_capture=True,
        )

        all_runs = find_run_dirs(self.runs_dir)
        self.assertCountEqual([run.name for run in all_runs], [flat_name, nested_name])

        capture_runs = find_run_dirs(self.runs_dir, required_files=["captures.h5"])
        self.assertEqual([run.name for run in capture_runs], [nested_name])

        prefixed = find_run_dirs(self.runs_dir, name_prefix="gqa_random_")
        self.assertEqual([run.name for run in prefixed], [flat_name])

    def test_read_run_metadata_uses_baseline_branch_and_fallbacks(self):
        baseline_name = "gqa_baseline_20260320_205031"
        baseline_dir = self._write_run(
            baseline_name,
            config={
                "_run_meta": {"dataset": "gqa", "run_mode": "baseline", "timestamp": "20260320_205031"},
                "pruning": {"strategy": "attn_score"},
            },
        )
        metadata = read_run_metadata(baseline_dir)
        self.assertEqual(metadata.dataset, "gqa")
        self.assertEqual(metadata.strategy, "baseline")
        self.assertEqual(metadata.run_mode, "baseline")

        fallback_name = "pope_random_l1_r0p2__20260323_122015_787997"
        fallback_dir = self._write_run(fallback_name, config=None)
        fallback_metadata = read_run_metadata(fallback_dir)
        self.assertEqual(fallback_metadata.dataset, "pope")
        self.assertEqual(fallback_metadata.strategy, "random")

        adaptive_name = "mme_sparsevlm_adaptive_stratified_l2_r0p4__20260410_101010_000001"
        adaptive_dir = self._write_run(adaptive_name, config=None)
        adaptive_metadata = read_run_metadata(adaptive_dir)
        self.assertEqual(adaptive_metadata.dataset, "mme")
        self.assertEqual(adaptive_metadata.strategy, "sparsevlm_adaptive_stratified")

    def test_build_run_dir_creates_strategy_dataset_parent(self):
        output_base_dir = Path(self.temp_dir.name) / "entropy_exp" / "outputs"
        run_dir = build_run_dir(
            output_base_dir,
            strategy="random",
            dataset="gqa",
            run_name="gqa_random_l1_r0p2__20260322_235451_349889",
        )
        self.assertEqual(
            run_dir,
            output_base_dir / "runs" / "random" / "gqa" / "gqa_random_l1_r0p2__20260322_235451_349889",
        )


if __name__ == "__main__":
    unittest.main()
