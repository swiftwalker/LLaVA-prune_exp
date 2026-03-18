import os
import sys
import tempfile
import unittest

import yaml

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

from prune_config_resolver import resolve_prune_spec


class PruneConfigResolverTests(unittest.TestCase):
    def _write_run_config(self, run_mode: str, strategy: str = "attn_score", prune_layers=None, prune_ratio=None):
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        run_dir = os.path.join(tmpdir.name, "mme_attn_score_20260312_141401")
        os.makedirs(run_dir, exist_ok=True)
        config_path = os.path.join(run_dir, "config.yaml")
        payload = {
            "_run_meta": {
                "run_mode": run_mode,
                "dataset": "mme",
                "timestamp": "20260312_141401",
                "run_dir": run_dir,
            },
            "pruning": {
                "strategy": strategy,
                "layer_selection": "fixed",
                "prune_layers": prune_layers if prune_layers is not None else [2, 3],
                "prune_ratio": prune_ratio if prune_ratio is not None else [0.5, 0.5],
                "entropy": {"dynamic_ratio": False, "dynamic_scale": 0.5, "max_prune_ratio": 0.9},
            },
        }
        with open(config_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle)
        return run_dir, config_path

    def test_resolve_prune_run_from_run_dir(self):
        run_dir, _ = self._write_run_config("prune", prune_layers=[3], prune_ratio=[0.7])
        spec, _ = resolve_prune_spec(run_dir=run_dir)
        self.assertEqual(spec.strategy_name, "attn_score")
        self.assertEqual(spec.prune_ratio_map, {3: 0.7})
        self.assertEqual(spec.source_dataset, "mme")
        self.assertEqual(spec.source_run_name, "mme_attn_score_20260312_141401")

    def test_resolve_baseline_from_run_metadata(self):
        run_dir, _ = self._write_run_config("baseline", prune_ratio=0.0)
        spec, _ = resolve_prune_spec(run_dir=run_dir)
        self.assertEqual(spec.strategy_name, "baseline")
        self.assertEqual(spec.prune_layers, [])

    def test_resolve_entropy_dynamic_ratio_marks_estimated(self):
        run_dir, config_path = self._write_run_config("prune", strategy="entropy", prune_layers=[2], prune_ratio=[0.4])
        with open(config_path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        payload["pruning"]["entropy"]["dynamic_ratio"] = True
        with open(config_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle)

        spec, _ = resolve_prune_spec(run_dir=run_dir)
        self.assertEqual(spec.ratio_mode, "dynamic_estimated")

    def test_resolve_dynamic_layer_selection_requires_explicit_plan(self):
        run_dir, config_path = self._write_run_config("prune")
        with open(config_path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        payload["pruning"]["layer_selection"] = "dynamic"
        with open(config_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle)

        with self.assertRaises(ValueError):
            resolve_prune_spec(run_dir=run_dir)


if __name__ == "__main__":
    unittest.main()
