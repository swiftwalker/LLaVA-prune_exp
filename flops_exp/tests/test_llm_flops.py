import os
import sys
import unittest

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

from llm_flops import compute_flops_report
from model_profile import ModelProfile
from prune_config_resolver import ResolvedPruneSpec


def _spec(strategy_name: str):
    if strategy_name == "baseline":
        return ResolvedPruneSpec(
            strategy_name="baseline",
            layer_selection="fixed",
            prune_layers=[],
            prune_ratio_map={},
            ratio_mode="static",
            scoring_mode="none",
            run_mode="baseline",
            config_source="run_config",
            source_path="/tmp/baseline/config.yaml",
            source_dataset="mme",
            source_run_name="mme_baseline_20260312_180039",
            notes=[],
        )
    return ResolvedPruneSpec(
        strategy_name=strategy_name,
        layer_selection="fixed",
        prune_layers=[2, 3],
        prune_ratio_map={2: 0.5, 3: 0.5},
        ratio_mode="static",
        scoring_mode=strategy_name,
        run_mode="prune",
        config_source="run_config",
        source_path=f"/tmp/{strategy_name}/config.yaml",
        source_dataset="mme",
        source_run_name=f"mme_{strategy_name}_20260312_141401",
        notes=[],
    )


class LlmFlopsTests(unittest.TestCase):
    def test_pruned_flops_are_smaller_than_baseline(self):
        profile = ModelProfile("toy", 32, 4096, 11008, 32)
        report = compute_flops_report(profile, _spec("attn_score"), 24.0, 20, 576)
        self.assertLess(report["pruned"]["total_llm_flops"], report["baseline"]["total_llm_flops"])

    def test_entropy_method_cost_exceeds_attn_score_cost(self):
        profile = ModelProfile("toy", 32, 4096, 11008, 32)
        attn_report = compute_flops_report(profile, _spec("attn_score"), 24.0, 20, 576)
        entropy_report = compute_flops_report(profile, _spec("entropy"), 24.0, 20, 576)
        self.assertGreaterEqual(
            entropy_report["pruned"]["prune_method_flops"],
            attn_report["pruned"]["prune_method_flops"],
        )

    def test_longer_text_increases_baseline_flops(self):
        profile = ModelProfile("toy", 32, 4096, 11008, 32)
        short_report = compute_flops_report(profile, _spec("baseline"), 12.0, 20, 576)
        long_report = compute_flops_report(profile, _spec("baseline"), 48.0, 20, 576)
        self.assertGreater(
            long_report["baseline"]["total_llm_flops"],
            short_report["baseline"]["total_llm_flops"],
        )

    def test_current_ratio_schedule_matches_576_to_288_to_144(self):
        profile = ModelProfile("toy", 32, 4096, 11008, 32)
        report = compute_flops_report(profile, _spec("attn_score"), 24.0, 20, 576)
        visual_tokens = [segment["visual_tokens"] for segment in report["pruned_schedule"]]
        self.assertEqual(visual_tokens, [576, 288, 144])


if __name__ == "__main__":
    unittest.main()
