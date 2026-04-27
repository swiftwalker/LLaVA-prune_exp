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
    build_run_prefix,
    build_window_base_name,
    compute_retry_budget,
    DEFAULT_CONDA_SH,
    expand_jobs,
    finalize_attempt_result,
    format_progress_line,
    load_scheduler_plan,
    GPUConfig,
    resolve_conda_activate_target,
    select_gpu_for_dispatch,
    validate_answers_file,
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
                    "conda_sh": "~/miniconda3/etc/profile.d/conda.sh",
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
                    "conda_sh": "~/miniconda3/etc/profile.d/conda.sh",
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

    def test_scheduler_defaults_conda_sh_to_home_miniconda(self):
        plan_path = self._write_plan(
            {
                "version": 1,
                "label": "demo-default-conda",
                "pool_size": 1,
                "gpu": {
                    "min_free_gib": 16,
                    "selection": "max_free",
                    "sample_seconds": 1,
                    "poll_interval_seconds": 5,
                },
                "retry": {"budget_ratio": 0.1, "rounding": "ceil"},
                "tmux": {"session_name": "sched_demo", "log_dir": "entropy_exp/outputs/logs/tmux"},
                "environment": {"conda_env": "llava"},
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
        self.assertEqual(plan.environment.conda_sh, str(DEFAULT_CONDA_SH))
        self.assertEqual(plan.environment.extra_env, [])

    def test_scheduler_accepts_environment_extra_env_assignments(self):
        plan_path = self._write_plan(
            {
                "version": 1,
                "label": "demo-extra-env",
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
                    "conda_sh": "~/miniconda3/etc/profile.d/conda.sh",
                    "conda_env": "llava",
                    "extra_env": ["HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1"],
                },
                "defaults": {"max_samples": 2, "extra_sets": []},
                "experiments": [
                    {
                        "name": "gqa_adaptive_demo",
                        "dataset": "gqa",
                        "strategies": ["sparsevlm_adaptive_stratified"],
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
        self.assertEqual(plan.environment.extra_env, ["HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1"])

    def test_scheduler_accepts_sparsevlm_adaptive_stratified_strategy(self):
        plan_path = self._write_plan(
            {
                "version": 1,
                "label": "demo-adaptive",
                "pool_size": 1,
                "gpu": {
                    "min_free_gib": 16,
                    "selection": "max_free",
                    "sample_seconds": 1,
                    "poll_interval_seconds": 5,
                },
                "retry": {"budget_ratio": 0.1, "rounding": "ceil"},
                "tmux": {"session_name": "sched_demo", "log_dir": "entropy_exp/outputs/logs/tmux"},
                "environment": {"conda_env": "llava"},
                "defaults": {"max_samples": 2, "extra_sets": []},
                "experiments": [
                    {
                        "name": "mme_adaptive",
                        "dataset": "mme",
                        "strategies": ["sparsevlm_adaptive_stratified"],
                        "extra_sets": [
                            "pruning.layer_selection=fixed",
                            "pruning.prune_layers=[2]",
                            "pruning.prune_ratio=[0.4]",
                        ],
                    }
                ],
            }
        )

        plan = load_scheduler_plan(plan_path)
        jobs = expand_jobs(plan)
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].strategy, "sparsevlm_adaptive_stratified")
        self.assertEqual(jobs[0].run_prefix, "mme_sparsevlm_adaptive_stratified_l2_r0p4__")

    def test_scheduler_accepts_sparsevlm_entropy_alpha_strategy(self):
        plan_path = self._write_plan(
            {
                "version": 1,
                "label": "demo-entropy-alpha",
                "pool_size": 1,
                "gpu": {
                    "min_free_gib": 16,
                    "selection": "max_free",
                    "sample_seconds": 1,
                    "poll_interval_seconds": 5,
                },
                "retry": {"budget_ratio": 0.1, "rounding": "ceil"},
                "tmux": {"session_name": "sched_demo", "log_dir": "entropy_exp/outputs/logs/tmux"},
                "environment": {"conda_env": "llava"},
                "defaults": {"max_samples": 2, "extra_sets": []},
                "experiments": [
                    {
                        "name": "mme_entropy_alpha",
                        "dataset": "mme",
                        "strategies": ["sparsevlm_entropy_alpha"],
                        "extra_sets": [
                            "pruning.layer_selection=fixed",
                            "pruning.prune_layers=[2]",
                            "pruning.prune_ratio=[0.4]",
                        ],
                    }
                ],
            }
        )

        plan = load_scheduler_plan(plan_path)
        jobs = expand_jobs(plan)
        self.assertEqual(jobs[0].strategy, "sparsevlm_entropy_alpha")
        self.assertEqual(jobs[0].run_prefix, "mme_sparsevlm_entropy_alpha_l2_r0p4__")

    def test_scheduler_accepts_new_inference_datasets(self):
        plan_path = self._write_plan(
            {
                "version": 1,
                "label": "demo-new-datasets",
                "pool_size": 2,
                "gpu": {
                    "min_free_gib": 16,
                    "selection": "max_free",
                    "sample_seconds": 1,
                    "poll_interval_seconds": 5,
                },
                "retry": {"budget_ratio": 0.1, "rounding": "ceil"},
                "tmux": {"session_name": "sched_demo", "log_dir": "entropy_exp/outputs/logs/tmux"},
                "environment": {"conda_env": "llava"},
                "defaults": {"max_samples": 1, "extra_sets": ["pruning.prune_layers=[1]", "pruning.prune_ratio=[0.2]"]},
                "experiments": [
                    {"name": "textvqa_smoke", "dataset": "textvqa", "strategies": ["random"]},
                    {"name": "scienceqa_smoke", "dataset": "scienceqa", "strategies": ["random"]},
                    {"name": "mmbench_smoke", "dataset": "mmbench", "strategies": ["random"]},
                ],
            }
        )
        plan = load_scheduler_plan(plan_path)
        jobs = expand_jobs(plan)
        self.assertEqual([job.dataset for job in jobs], ["textvqa", "scienceqa", "mmbench"])

    def test_real_adaptive_full_matrix_plan_has_expected_grid(self):
        plan_path = ROOT_DIR / "entropy_exp" / "plans" / "keep_position_ids_sparsevlm_adaptive_stratified_full_matrix.yaml"
        plan = load_scheduler_plan(plan_path)
        jobs = expand_jobs(plan)

        self.assertEqual(plan.pool_size, 12)
        self.assertEqual(plan.environment.conda_sh, "/data/liuyu/anaconda3/etc/profile.d/conda.sh")
        self.assertEqual(len(plan.experiments), 54)
        self.assertEqual(len(jobs), 54)
        self.assertTrue(all(job.strategy == "sparsevlm_adaptive_stratified" for job in jobs))
        self.assertTrue(all("--no-auto-gpu" in build_run_command(job) for job in jobs))
        self.assertEqual({job.dataset for job in jobs}, {"gqa", "mme", "pope"})

        layers = set()
        ratios = set()
        for job in jobs:
            for override in job.extra_sets:
                if override.startswith("pruning.prune_layers="):
                    layers.update(yaml.safe_load(override.split("=", 1)[1]))
                if override.startswith("pruning.prune_ratio="):
                    ratios.update(round(value, 1) for value in yaml.safe_load(override.split("=", 1)[1]))

        self.assertEqual(layers, {1, 2, 3})
        self.assertEqual(ratios, {0.2, 0.3, 0.4, 0.5, 0.6, 0.7})
        self.assertEqual(jobs[0].run_prefix, "gqa_sparsevlm_adaptive_stratified_l1_r0p2__")
        self.assertEqual(jobs[-1].run_prefix, "pope_sparsevlm_adaptive_stratified_l3_r0p7__")

    def test_real_patch_distribution_full_matrix_plan_has_expected_grid(self):
        plan_path = ROOT_DIR / "entropy_exp" / "plans" / "gqa_textvqa_patch_distribution_full_matrix.yaml"
        plan = load_scheduler_plan(plan_path)
        jobs = expand_jobs(plan)

        self.assertEqual(plan.pool_size, 12)
        self.assertEqual(plan.tmux.session_name, "sched_gqa_tv_patchdist")
        self.assertEqual(len(plan.experiments), 110)
        self.assertEqual(len(jobs), 110)
        self.assertEqual({job.dataset for job in jobs}, {"gqa", "textvqa"})
        self.assertTrue(all("--no-auto-gpu" in build_run_command(job) for job in jobs))

        baseline_jobs = [job for job in jobs if job.strategy == "baseline"]
        self.assertEqual(len(baseline_jobs), 2)
        self.assertEqual(baseline_jobs[0].run_prefix, "gqa_baseline_")
        self.assertEqual(baseline_jobs[1].run_prefix, "textvqa_baseline_")

        overrides = {override for job in jobs for override in job.extra_sets}
        self.assertIn("capture.save_attention=false", overrides)
        self.assertIn("capture.save_importance_scores=false", overrides)
        self.assertIn("capture.save_keep_indices=true", overrides)
        self.assertIn("pruning.sparsevlm_adaptive_stratified.patch_per_row=24", overrides)

        pruning_jobs = [job for job in jobs if job.strategy != "baseline"]
        layers = set()
        ratios = set()
        strategies = {job.strategy for job in pruning_jobs}
        for job in pruning_jobs:
            for override in job.extra_sets:
                if override.startswith("pruning.prune_layers="):
                    layers.update(yaml.safe_load(override.split("=", 1)[1]))
                if override.startswith("pruning.prune_ratio="):
                    value = yaml.safe_load(override.split("=", 1)[1])
                    if isinstance(value, list):
                        ratios.update(round(entry, 1) for entry in value)
                    else:
                        ratios.add(round(value, 1))

        self.assertEqual(strategies, {"random", "sparsevlm", "sparsevlm_adaptive_stratified"})
        self.assertEqual(layers, {1, 2, 3})
        self.assertEqual(ratios, {0.2, 0.3, 0.4, 0.5, 0.6, 0.7})
        self.assertEqual(jobs[-1].run_prefix, "textvqa_sparsevlm_adaptive_stratified_l3_r0p7__")

    def test_real_adaptive_hparam_round1_remote_plan_has_expected_grid(self):
        plan_path = ROOT_DIR / "entropy_exp" / "plans" / "adaptive_hparam_round1_remote.yaml"
        plan = load_scheduler_plan(plan_path)
        jobs = expand_jobs(plan)

        self.assertEqual(plan.pool_size, 24)
        self.assertEqual(plan.tmux.session_name, "sched_adaptive_hp_r1_remote")
        self.assertEqual(plan.environment.conda_sh, "/home/liuyu/miniconda3/etc/profile.d/conda.sh")
        self.assertEqual(plan.environment.extra_env, ["HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1"])
        self.assertEqual(len(plan.experiments), 360)
        self.assertEqual(len(jobs), 360)
        self.assertEqual({job.dataset for job in jobs}, {"gqa", "textvqa"})
        self.assertTrue(all(job.strategy == "sparsevlm_adaptive_stratified" for job in jobs))
        self.assertTrue(all("--no-auto-gpu" in build_run_command(job) for job in jobs))

        layers = set()
        ratios = set()
        grid_sizes = set()
        high_ratios = set()
        run_tag_suffixes = set()
        for job in jobs:
            for override in job.extra_sets:
                if override.startswith("pruning.prune_layers="):
                    layers.update(yaml.safe_load(override.split("=", 1)[1]))
                if override.startswith("pruning.prune_ratio="):
                    ratios.update(round(value, 2) for value in yaml.safe_load(override.split("=", 1)[1]))
                if override.startswith("pruning.sparsevlm_adaptive_stratified.grid_size="):
                    grid_sizes.add(int(yaml.safe_load(override.split("=", 1)[1])))
                if override.startswith("pruning.sparsevlm_adaptive_stratified.high_ratio="):
                    high_ratios.add(round(float(yaml.safe_load(override.split("=", 1)[1])), 2))
                if override.startswith("output.run_tag_suffix="):
                    run_tag_suffixes.add(override.split("=", 1)[1])

        self.assertEqual(layers, {1, 3})
        self.assertEqual(ratios, {0.2, 0.3, 0.4, 0.5, 0.6, 0.7})
        self.assertEqual(grid_sizes, {4, 6, 8})
        self.assertEqual(high_ratios, {0.4, 0.55, 0.7, 0.85, 1.0})
        self.assertEqual(len(run_tag_suffixes), 15)
        self.assertEqual(jobs[0].run_prefix, "gqa_sparsevlm_adaptive_stratified_g4_a0p4_l1_r0p2__")
        self.assertEqual(jobs[-1].run_prefix, "textvqa_sparsevlm_adaptive_stratified_g8_a1p0_l3_r0p7__")

    def test_resolve_conda_activate_target_uses_same_conda_root(self):
        target = resolve_conda_activate_target(
            "/home/liuyu/miniconda3/etc/profile.d/conda.sh",
            "llava",
        )
        self.assertEqual(target, "/home/liuyu/miniconda3/envs/llava")

    def test_build_run_prefix_and_window_name_include_run_tag_suffix(self):
        extra_sets = [
            "pruning.prune_layers=[1]",
            "pruning.prune_ratio=[0.2]",
            "output.run_tag_suffix=g4_a0p4",
        ]
        job_prefix = build_run_prefix("gqa", "sparsevlm_adaptive_stratified", extra_sets)
        window_name = build_window_base_name("gqa", "sparsevlm_adaptive_stratified", extra_sets)

        self.assertEqual(
            job_prefix,
            "gqa_sparsevlm_adaptive_stratified_g4_a0p4_l1_r0p2__",
        )
        self.assertEqual(
            window_name,
            "gqa_sparsevlm_adaptive_stratified_g4_a0p4_l1_r0p2",
        )

    def test_build_run_prefix_and_window_name_support_sparsevlm_entropy_alpha(self):
        extra_sets = [
            "pruning.prune_layers=[3]",
            "pruning.prune_ratio=[0.6]",
        ]
        job_prefix = build_run_prefix("textvqa", "sparsevlm_entropy_alpha", extra_sets)
        window_name = build_window_base_name("textvqa", "sparsevlm_entropy_alpha", extra_sets)

        self.assertEqual(
            job_prefix,
            "textvqa_sparsevlm_entropy_alpha_l3_r0p6__",
        )
        self.assertEqual(
            window_name,
            "textvqa_sparsevlm_entropy_alpha_l3_r0p6",
        )

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

    def test_validate_answers_file_supports_scienceqa_json_list(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            run_dir = repo_root / "entropy_exp" / "outputs" / "runs" / "random" / "scienceqa" / "scienceqa_random_l1_r0p2__demo"
            eval_dir = repo_root / "entropy_exp" / "eval_questions" / "scienceqa"
            run_dir.mkdir(parents=True)
            eval_dir.mkdir(parents=True)

            question_path = eval_dir / "llava_test_CQM-A.json"
            question_path.write_text(
                json.dumps(
                    [
                        {"id": "4", "conversations": [{"from": "human", "value": "Question only"}]},
                        {"id": "5", "image": "5/image.png", "conversations": [{"from": "human", "value": "<image>\nQuestion with image"}]},
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (run_dir / "config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "_run_meta": {"dataset": "scienceqa", "max_samples": 2},
                        "datasets": {"scienceqa": {"question_file": "entropy_exp/eval_questions/scienceqa/llava_test_CQM-A.json"}},
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            (run_dir / "answers.jsonl").write_text('{"ok": true}\n{"ok": true}\n', encoding="utf-8")

            validation = validate_answers_file(run_dir, "scienceqa", repo_root)
            self.assertTrue(validation["ok"])
            self.assertEqual(validation["expected_answers"], 2)
            self.assertEqual(validation["valid_answers"], 2)

    def test_validate_answers_file_supports_mmbench_tsv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            run_dir = repo_root / "entropy_exp" / "outputs" / "runs" / "random" / "mmbench" / "mmbench_random_l1_r0p2__demo"
            eval_dir = repo_root / "entropy_exp" / "eval_questions" / "mmbench"
            run_dir.mkdir(parents=True)
            eval_dir.mkdir(parents=True)

            question_path = eval_dir / "mmbench_dev_20230712.tsv"
            question_path.write_text(
                "\t".join(["index", "question", "hint", "A", "B", "C", "D", "image"]) + "\n"
                + "\t".join(["1", "What is shown?", "", "cat", "dog", "", "", "ZmFrZQ=="]) + "\n"
                + "\t".join(["2", "Pick one", "hint text", "A1", "B1", "C1", "D1", "ZmFrZTI="]) + "\n",
                encoding="utf-8",
            )
            (run_dir / "config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "_run_meta": {"dataset": "mmbench", "max_samples": 2},
                        "datasets": {"mmbench": {"question_file": "entropy_exp/eval_questions/mmbench/mmbench_dev_20230712.tsv"}},
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            (run_dir / "answers.jsonl").write_text('{"ok": true}\n{"ok": true}\n', encoding="utf-8")

            validation = validate_answers_file(run_dir, "mmbench", repo_root)
            self.assertTrue(validation["ok"])
            self.assertEqual(validation["expected_answers"], 2)
            self.assertEqual(validation["valid_answers"], 2)


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

    def test_visible_gpu_env_filters_scheduler_candidates(self):
        gpu_cfg = GPUConfig(
            min_free_gib=16,
            selection="max_free",
            sample_seconds=3,
            poll_interval_seconds=15,
        )
        observed = {0: 90000, 1: 89000, 4: 95000, 5: 94000}

        from unittest import mock

        with mock.patch(
            "entropy_exp.src.scheduler.collect_average_gpu_free_mib",
            return_value=observed,
        ), mock.patch.dict(
            "os.environ",
            {"LLAVA_SCHEDULER_VISIBLE_GPUS": "4,5"},
            clear=False,
        ):
            selected_gpu, observed_free, projected_free = select_gpu_for_dispatch(
                gpu_cfg,
                provisional_reservations_by_gpu={},
            )

        self.assertEqual(observed_free, {4: 95000, 5: 94000})
        self.assertEqual(projected_free, {4: 95000, 5: 94000})
        self.assertEqual(selected_gpu, 4)


if __name__ == "__main__":
    unittest.main()
