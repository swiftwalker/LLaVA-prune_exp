import importlib.util
import pathlib
import unittest


MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "analysis" / "compare_masking_branch_paths.py"
SPEC = importlib.util.spec_from_file_location("compare_masking_branch_paths", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CompareMaskingBranchPathsTests(unittest.TestCase):
    def test_parse_config_labels(self):
        self.assertEqual(MODULE.parse_config_labels("l1_r0p2,l3_r0p7"), ["l1_r0p2", "l3_r0p7"])
        with self.assertRaises(ValueError):
            MODULE.parse_config_labels("unknown")

    def test_position_id_helpers(self):
        self.assertTrue(MODULE.is_dense_contiguous([0, 1, 2, 3]))
        self.assertTrue(MODULE.is_dense_contiguous([4, 5, 6]))
        self.assertFalse(MODULE.is_dense_contiguous([0, 2, 3]))
        self.assertEqual(MODULE.position_ids_max_plus_one([0, 1, 2]), 3)
        self.assertEqual(MODULE.build_drop_indices(5, [0, 2, 4]), [1, 3])

    def test_summarize_tv_attn(self):
        tv_attn = [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]]
        summary = MODULE.summarize_tv_attn(tv_attn, keep_indices=[0, 2], drop_indices=[1])
        self.assertEqual(summary["shape"], [1, 2, 3])
        self.assertAlmostEqual(summary["mean"], 0.35)
        self.assertAlmostEqual(summary["kept_mean"], (0.1 + 0.3 + 0.4 + 0.6) / 4)
        self.assertAlmostEqual(summary["dropped_mean"], (0.2 + 0.5) / 2)
        self.assertAlmostEqual(summary["dropped_max"], 0.5)

    def test_compare_worker_outputs_detects_no_divergence(self):
        sample_record = {
            "question_id": "q1",
            "prepare_position_ids": None,
            "effective_initial_position_ids": [0, 1, 2],
            "target_layer_position_ids": [0, 1, 2],
            "seq_len": 3,
            "position_ids_max_plus_one": 3,
            "position_ids_dense_contiguous": True,
            "rotary_effective_span": 3,
            "target_layer_keep_indices": [0, 2],
            "target_layer_pre_mask_tv_attn": [[[0.2, 0.3, 0.5]]],
            "target_layer_post_mask_tv_attn": [[[0.4, 0.0, 0.6]]],
            "target_layer_pre_mask_tv_attn_summary": {
                "shape": [1, 1, 3],
                "mean": 1.0 / 3.0,
                "min": 0.2,
                "max": 0.5,
                "kept_mean": 0.35,
                "dropped_mean": 0.3,
                "dropped_max": 0.3,
            },
            "target_layer_post_mask_tv_attn_summary": {
                "shape": [1, 1, 3],
                "mean": 1.0 / 3.0,
                "min": 0.0,
                "max": 0.6,
                "kept_mean": 0.5,
                "dropped_mean": 0.0,
                "dropped_max": 0.0,
            },
            "answer": "yes",
        }
        worker_result = {
            "run_dir": "/tmp/run",
            "trace_file": "/tmp/trace.jsonl",
            "eval_summary": {"metrics": {"accuracy": 1.0}},
            "compat_patch_enabled": True,
            "target_layer_compat_wrapped": True,
            "records": [sample_record],
        }
        summary, first_divergence = MODULE.compare_worker_outputs(worker_result, worker_result)
        self.assertIsNone(first_divergence)
        self.assertTrue(summary["eval_summary_check"]["match"])
        self.assertTrue(summary["sample_comparisons"][0]["checks"]["post_mask_tv_attn_allclose"])

    def test_compare_worker_outputs_detects_post_mask_divergence(self):
        keep_result = {
            "run_dir": "/tmp/keep",
            "trace_file": "/tmp/keep.jsonl",
            "eval_summary": {"metrics": {"accuracy": 1.0}},
            "compat_patch_enabled": True,
            "target_layer_compat_wrapped": True,
            "records": [
                {
                    "question_id": "q1",
                    "prepare_position_ids": None,
                    "effective_initial_position_ids": [0, 1, 2],
                    "target_layer_position_ids": [0, 1, 2],
                    "seq_len": 3,
                    "position_ids_max_plus_one": 3,
                    "position_ids_dense_contiguous": True,
                    "rotary_effective_span": 3,
                    "target_layer_keep_indices": [0, 2],
                    "target_layer_pre_mask_tv_attn": [[[0.2, 0.3, 0.5]]],
                    "target_layer_post_mask_tv_attn": [[[0.4, 0.0, 0.6]]],
                    "target_layer_pre_mask_tv_attn_summary": {
                        "shape": [1, 1, 3],
                        "mean": 1.0 / 3.0,
                        "min": 0.2,
                        "max": 0.5,
                        "kept_mean": 0.35,
                        "dropped_mean": 0.3,
                        "dropped_max": 0.3,
                    },
                    "target_layer_post_mask_tv_attn_summary": {
                        "shape": [1, 1, 3],
                        "mean": 1.0 / 3.0,
                        "min": 0.0,
                        "max": 0.6,
                        "kept_mean": 0.5,
                        "dropped_mean": 0.0,
                        "dropped_max": 0.0,
                    },
                    "answer": "yes",
                }
            ],
        }
        nonpos_result = {
            **keep_result,
            "records": [
                {
                    **keep_result["records"][0],
                    "target_layer_post_mask_tv_attn": [[[0.5, 0.0, 0.5]]],
                    "target_layer_post_mask_tv_attn_summary": {
                        "shape": [1, 1, 3],
                        "mean": 1.0 / 3.0,
                        "min": 0.0,
                        "max": 0.5,
                        "kept_mean": 0.5,
                        "dropped_mean": 0.0,
                        "dropped_max": 0.0,
                    },
                }
            ],
        }
        _summary, first_divergence = MODULE.compare_worker_outputs(keep_result, nonpos_result)
        self.assertEqual(first_divergence, "target-layer post-mask attention")


if __name__ == "__main__":
    unittest.main()
