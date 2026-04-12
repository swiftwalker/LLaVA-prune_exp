import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

from patch_distribution_report import main  # noqa: E402


def _write_run(
    root: Path,
    *,
    dataset: str,
    strategy: str,
    layer: int,
    ratio: float,
    keep_indices: list[int],
    sample_count: int = 2,
    include_adaptive_fields: bool = False,
) -> Path:
    run_dir = root / f"{dataset}_{strategy}_l{layer}_r{str(ratio).replace('.', 'p')}__demo"
    run_dir.mkdir(parents=True)
    config = {
        "_run_meta": {"dataset": dataset, "strategy": strategy},
        "pruning": {"v_token_num": 576},
    }
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    stats_lines = []
    keep_set = set(keep_indices)
    pruned_indices = [index for index in range(576) if index not in keep_set]
    for sample_idx in range(sample_count):
        line = {
            "sample_idx": sample_idx,
            f"layer_{layer}_ratio": ratio,
            f"layer_{layer}_before": 576,
            f"layer_{layer}_after": len(keep_indices),
            f"layer_{layer}_pruned": len(pruned_indices),
            f"layer_{layer}_keep_indices": keep_indices,
            f"layer_{layer}_pruned_indices": pruned_indices,
        }
        if include_adaptive_fields:
            line[f"layer_{layer}_high_keep_patch_indices"] = keep_indices[: len(keep_indices) // 2]
            line[f"layer_{layer}_low_keep_patch_indices"] = keep_indices[len(keep_indices) // 2 :]
            line[f"layer_{layer}_stratum_quotas"] = [1] * 36
        stats_lines.append(json.dumps(line))
    (run_dir / "stats.jsonl").write_text("\n".join(stats_lines) + "\n", encoding="utf-8")
    return run_dir


class PatchDistributionReportTests(unittest.TestCase):
    def test_report_generates_tables_and_heatmaps_from_state_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            runs_root = temp_root / "runs"
            state_dir = temp_root / "scheduler" / "demo"
            attempts_dir = state_dir / "attempts"
            attempts_dir.mkdir(parents=True)

            baseline_run = _write_run(
                runs_root,
                dataset="gqa",
                strategy="baseline",
                layer=1,
                ratio=0.0,
                keep_indices=list(range(576)),
            )
            random_run = _write_run(
                runs_root,
                dataset="gqa",
                strategy="random",
                layer=1,
                ratio=0.2,
                keep_indices=list(range(400)),
            )
            sparsevlm_run = _write_run(
                runs_root,
                dataset="gqa",
                strategy="sparsevlm",
                layer=1,
                ratio=0.2,
                keep_indices=list(range(420)),
            )
            adaptive_run = _write_run(
                runs_root,
                dataset="gqa",
                strategy="sparsevlm_adaptive_stratified",
                layer=1,
                ratio=0.2,
                keep_indices=list(range(430)),
                include_adaptive_fields=True,
            )

            for index, run_dir in enumerate(
                [baseline_run, random_run, sparsevlm_run, adaptive_run],
                start=1,
            ):
                (attempts_dir / f"attempt_{index:04d}.json").write_text(
                    json.dumps({"status": "completed", "run_dir": str(run_dir)}),
                    encoding="utf-8",
                )

            output_dir = temp_root / "report"
            rc = main(["--state-dir", str(state_dir), "--output-dir", str(output_dir)])
            self.assertEqual(rc, 0)
            self.assertTrue((output_dir / "manifest" / "completed_run_dirs.json").is_file())
            self.assertTrue((output_dir / "tables" / "per_config_patch_summary.csv").is_file())
            self.assertTrue((output_dir / "heatmaps" / "gqa" / "baseline" / "reference_grid.png").is_file())
            self.assertTrue((output_dir / "heatmaps" / "gqa" / "random" / "keep_rate_grid.png").is_file())
            self.assertTrue(
                (output_dir / "heatmaps" / "gqa" / "sparsevlm_adaptive_stratified" / "high_keep_rate_grid.png").is_file()
            )
            self.assertTrue(
                (output_dir / "compare" / "gqa" / "layer_1_ratio_0p2_strategies.png").is_file()
            )
            self.assertTrue((output_dir / "compare" / "gqa" / "strategy_trends.png").is_file())

    def test_report_supports_run_dir_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            run_dir = _write_run(
                temp_root / "runs",
                dataset="textvqa",
                strategy="random",
                layer=2,
                ratio=0.3,
                keep_indices=list(range(350)),
            )
            output_dir = temp_root / "report"
            rc = main(["--run-dir", str(run_dir), "--output-dir", str(output_dir)])
            self.assertEqual(rc, 0)
            summary_json = output_dir / "tables" / "per_config_patch_summary.json"
            self.assertTrue(summary_json.is_file())
            payload = json.loads(summary_json.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["records"]), 1)
            self.assertEqual(payload["records"][0]["dataset"], "textvqa")
            self.assertEqual(payload["records"][0]["layer"], 2)
            self.assertEqual(payload["records"][0]["ratio"], 0.3)


if __name__ == "__main__":
    unittest.main()
