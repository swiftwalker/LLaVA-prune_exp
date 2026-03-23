from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from entropy_exp.src.scheduler import (
    build_progress_payload,
    build_run_command,
    compute_retry_budget,
    expand_jobs,
    finalize_attempt_result,
    format_progress_line,
    load_scheduler_plan,
    GPUConfig,
    select_gpu_for_dispatch,
)


class SchedulerPlanTests(unittest.TestCase):
    def _write_plan(self, plan: dict) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "plan.yaml"
        path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
        return path

    def test_new_plan_format_expands_jobs(self):
        plan_path = self._write_plan(
            {
                "version": 1,
                "label": "demo",
                "pool_size": 2,
                "gpu": {
                    "min_free_gib": 16,
                    "selection": "max_free",
                    "sample_seconds": 3,
                    "poll_interval_seconds": 15,
                },
                "retry": {"budget_ratio": 0.1, "rounding": "ceil"},
                "tmux": {"session_name": "sched_demo", "log_dir": "entropy_exp/outputs/logs/tmux"},
                "environment": {
                    "conda_sh": "/data/liuyu/anaconda3/etc/profile.d/conda.sh",
                    "conda_env": "llava",
                },
                "defaults": {"max_samples": None, "extra_sets": ["inference.seed=42"]},
                "experiments": [
                    {
                        "name": "gqa_l1_r0.2",
                        "dataset": "gqa",
                        "strategies": ["random", "entropy"],
                        "extra_sets": [
                            "pruning.layer_selection=fixed",
                            "pruning.prune_layers=[1]",
                            "pruning.prune_ratio=[0.2]",
                        ],
                    }
                ],
            }
        )
        plan = load_scheduler_plan(plan_path)
        jobs = expand_jobs(plan)
        self.assertEqual(len(jobs), 2)
        self.assertEqual(jobs[0].job_id, "job_0001")
        self.assertEqual(jobs[0].run_prefix, "gqa_random_l1_r0p2__")
        self.assertIn("--no-auto-gpu", build_run_command(jobs[0]))

    def test_baseline_prefix_is_broad_session_safe_prefix(self):
        plan_path = self._write_plan(
            {
                "version": 1,
                "label": "demo-baseline",
                "pool_size": 1,
                "gpu": {
                    "min_free_gib": 16,
                    "selection": "max_free",
                    "sample_seconds": 1,
                    "poll_interval_seconds": 5,
                },
                "retry": {"budget_ratio": 0.1, "rounding": "ceil"},
                "tmux": {"session_name": "sched_demo", "log_dir": "entropy_exp/outputs/logs/tmux"},
                "environment": {
                    "conda_sh": "/data/liuyu/anaconda3/etc/profile.d/conda.sh",
                    "conda_env": "llava",
                },
                "defaults": {"max_samples": 5, "extra_sets": []},
                "experiments": [
                    {
                        "name": "gqa_baseline_smoke",
                        "dataset": "gqa",
                        "strategies": ["baseline"],
                        "extra_sets": [
                            "pruning.layer_selection=fixed",
                            "pruning.prune_layers=[1]",
                            "pruning.prune_ratio=[0.2]",
                        ],
                    }
                ],
            }
        )
        plan = load_scheduler_plan(plan_path)
        jobs = expand_jobs(plan)
        self.assertEqual(jobs[0].run_prefix, "gqa_baseline_")

    def test_retry_budget_and_progress_format(self):
        self.assertEqual(compute_retry_budget(324, 0.1, "ceil"), 33)
        state = {
            "total_jobs": 324,
            "retry_budget_total": 33,
            "retry_budget_remaining": 31,
            "pending": ["job_0005"] * 304,
            "running": {
                "job_0001": {"gpu": 0},
                "job_0002": {"gpu": 2},
                "job_0003": {"gpu": 4},
                "job_0004": {"gpu": 5},
                "job_0005": {"gpu": 2},
                "job_0006": {"gpu": 0},
            },
            "completed": [f"job_{idx:04d}" for idx in range(1, 13)],
            "failed_final": ["job_0323", "job_0324"],
        }
        progress = build_progress_payload(state)
        line = format_progress_line(progress)
        self.assertIn("12/324 completed", line)
        self.assertIn("running=6", line)
        self.assertIn("retry=31/33", line)
        self.assertIn("active_gpus=0,2,4,5", line)


class SchedulerFinalizeTests(unittest.TestCase):
    def test_finalize_attempt_marks_completed_for_valid_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            (repo_root / "entropy_exp" / "outputs" / "runs" / "random" / "gqa").mkdir(parents=True)
            (repo_root / "entropy_exp" / "datasets").mkdir(parents=True)
            questions_path = repo_root / "entropy_exp" / "datasets" / "questions.jsonl"
            questions_path.write_text("{}\n{}\n", encoding="utf-8")

            run_dir = repo_root / "entropy_exp" / "outputs" / "runs" / "random" / "gqa" / "gqa_random_l1_r0p2__20260323_120000_000001"
            run_dir.mkdir(parents=True)
            (run_dir / "config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "_run_meta": {"dataset": "gqa", "max_samples": 2},
                        "datasets": {"gqa": {"question_file": "entropy_exp/datasets/questions.jsonl"}},
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            (run_dir / "answers.jsonl").write_text('{"ok": true}\n{"ok": true}\n', encoding="utf-8")

            state_dir = repo_root / "state"
            state_dir.mkdir(parents=True)
            result_path = finalize_attempt_result(
                repo_root=repo_root,
                state_dir=state_dir,
                job_id="job_0001",
                attempt=1,
                run_prefix="gqa_random_l1_r0p2__",
                dataset="gqa",
                started_at=run_dir.stat().st_mtime,
                exit_code=0,
                log_path="/tmp/demo.log",
                gpu=0,
                tmux_session="sched_demo",
                tmux_window="gqa_random_l1_r0p2_try1",
            )
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(payload["validation"]["expected_answers"], 2)
            self.assertEqual(payload["validation"]["valid_answers"], 2)

    def test_finalize_attempt_uses_max_samples_for_expected_answers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            (repo_root / "entropy_exp" / "outputs" / "runs" / "random" / "gqa").mkdir(parents=True)
            (repo_root / "entropy_exp" / "datasets").mkdir(parents=True)
            questions_path = repo_root / "entropy_exp" / "datasets" / "questions.jsonl"
            questions_path.write_text("{}\n{}\n{}\n{}\n{}\n", encoding="utf-8")

            run_dir = repo_root / "entropy_exp" / "outputs" / "runs" / "random" / "gqa" / "gqa_random_l1_r0p2__20260323_120000_000001"
            run_dir.mkdir(parents=True)
            (run_dir / "config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "_run_meta": {"dataset": "gqa", "max_samples": 3},
                        "datasets": {"gqa": {"question_file": "entropy_exp/datasets/questions.jsonl"}},
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            (run_dir / "answers.jsonl").write_text('{"ok": true}\n{"ok": true}\n{"ok": true}\n', encoding="utf-8")

            state_dir = repo_root / "state"
            state_dir.mkdir(parents=True)
            result_path = finalize_attempt_result(
                repo_root=repo_root,
                state_dir=state_dir,
                job_id="job_0001",
                attempt=1,
                run_prefix="gqa_random_l1_r0p2__",
                dataset="gqa",
                started_at=run_dir.stat().st_mtime,
                exit_code=0,
                log_path="/tmp/demo.log",
                gpu=0,
                tmux_session="sched_demo",
                tmux_window="gqa_random_l1_r0p2_try1",
            )
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(payload["validation"]["expected_answers"], 3)
            self.assertEqual(payload["validation"]["valid_answers"], 3)


class SchedulerGpuSelectionTests(unittest.TestCase):
    def test_running_jobs_do_not_reduce_observed_free_memory_again(self):
        gpu_cfg = GPUConfig(
            min_free_gib=16,
            selection="max_free",
            sample_seconds=3,
            poll_interval_seconds=15,
        )
        observed = {
            0: 41405,
            1: 41287,
            2: 41407,
            3: 41389,
            4: 26300,
            5: 41387,
            6: 41405,
            7: 41287,
        }

        from unittest import mock

        with mock.patch(
            "entropy_exp.src.scheduler.collect_average_gpu_free_mib",
            return_value=observed,
        ):
            selected_gpu, observed_free, projected_free = select_gpu_for_dispatch(
                gpu_cfg,
                provisional_reservations_by_gpu={},
            )

        self.assertEqual(observed_free, observed)
        self.assertEqual(projected_free, observed)
        self.assertEqual(selected_gpu, 2)

    def test_provisional_reservation_blocks_second_launch_on_same_gpu_in_one_pass(self):
        gpu_cfg = GPUConfig(
            min_free_gib=16,
            selection="max_free",
            sample_seconds=3,
            poll_interval_seconds=15,
        )
        observed = {0: 20000, 1: 18000}
        reserve_per_job_mib = 16 * 1024

        from unittest import mock

        with mock.patch(
            "entropy_exp.src.scheduler.collect_average_gpu_free_mib",
            return_value=observed,
        ):
            selected_gpu, observed_free, projected_free = select_gpu_for_dispatch(
                gpu_cfg,
                provisional_reservations_by_gpu={},
            )
            self.assertEqual(selected_gpu, 0)
            self.assertEqual(observed_free, observed)
            self.assertEqual(projected_free[0], 20000)

            selected_gpu_2, _, projected_free_2 = select_gpu_for_dispatch(
                gpu_cfg,
                provisional_reservations_by_gpu={0: reserve_per_job_mib},
            )

        self.assertEqual(projected_free_2[0], 20000 - reserve_per_job_mib)
        self.assertEqual(selected_gpu_2, 1)

    def test_tie_break_prefers_lower_gpu_index(self):
        gpu_cfg = GPUConfig(
            min_free_gib=16,
            selection="max_free",
            sample_seconds=3,
            poll_interval_seconds=15,
        )
        observed = {0: 20000, 1: 20000}

        from unittest import mock

        with mock.patch(
            "entropy_exp.src.scheduler.collect_average_gpu_free_mib",
            return_value=observed,
        ):
            selected_gpu, _, projected_free = select_gpu_for_dispatch(
                gpu_cfg,
                provisional_reservations_by_gpu={},
            )

        self.assertEqual(projected_free, observed)
        self.assertEqual(selected_gpu, 0)


if __name__ == "__main__":
    unittest.main()
