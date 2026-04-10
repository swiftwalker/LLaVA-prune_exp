import os
import sys
import unittest

import torch

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

from strategies import get_strategy
from strategies.sparsevlm_adaptive_stratified import (
    SparseVLMAdaptiveStratifiedStrategy,
    allocate_adaptive_stratified_quotas,
    build_stratum_index,
    compute_target_keep_count,
    select_farthest_token_indices,
    validate_adaptive_stratified_config,
)


class AdaptiveStratifiedHelperTests(unittest.TestCase):
    def test_build_stratum_index_uses_original_patch_positions(self):
        current_patch_indices = torch.tensor([0, 1, 4, 5, 10, 15], dtype=torch.long)

        stratum_index = build_stratum_index(
            current_patch_indices=current_patch_indices,
            grid_size=2,
            patch_per_row=4,
        )

        self.assertEqual(stratum_index[0], [0, 1, 2, 3])
        self.assertEqual(stratum_index[1], [])
        self.assertEqual(stratum_index[2], [])
        self.assertEqual(stratum_index[3], [4, 5])

    def test_allocate_adaptive_quotas_refills_when_one_stratum_runs_out_of_capacity(self):
        quotas, deficits = allocate_adaptive_stratified_quotas(
            selected_counts={0: 0, 1: 0, 2: 2, 3: 2},
            candidate_counts={0: 1, 1: 5, 2: 0, 3: 0},
            target_keep=8,
            n_low=6,
            grid_size=2,
        )

        self.assertEqual(sum(quotas.values()), 6)
        self.assertEqual(quotas[0], 1)
        self.assertEqual(quotas[1], 5)
        self.assertEqual(quotas[2], 0)
        self.assertEqual(quotas[3], 0)
        self.assertEqual(deficits[0], 2.0)
        self.assertEqual(deficits[1], 2.0)

    def test_compute_target_keep_count_respects_minimum_remaining_tokens(self):
        self.assertEqual(
            compute_target_keep_count(num_visual=4, prune_ratio=0.9, min_visual_tokens_after_prune=4),
            4,
        )
        self.assertEqual(
            compute_target_keep_count(num_visual=10, prune_ratio=0.3, min_visual_tokens_after_prune=2),
            7,
        )

    def test_validate_config_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            validate_adaptive_stratified_config(
                initial_v_token_num=576,
                patch_per_row=24,
                grid_size=5,
                high_ratio=0.7,
                intra_stratum_mode="random",
            )

        with self.assertRaises(ValueError):
            validate_adaptive_stratified_config(
                initial_v_token_num=576,
                patch_per_row=24,
                grid_size=6,
                high_ratio=1.5,
                intra_stratum_mode="random",
            )

        with self.assertRaises(ValueError):
            validate_adaptive_stratified_config(
                initial_v_token_num=576,
                patch_per_row=24,
                grid_size=6,
                high_ratio=0.7,
                intra_stratum_mode="unknown",
            )

    def test_select_farthest_token_indices_returns_unique_legal_candidates(self):
        chosen = select_farthest_token_indices(
            candidate_token_indices=[0, 1, 2, 3],
            current_patch_indices=torch.tensor([0, 3, 12, 15], dtype=torch.long),
            selected_patch_indices=[5],
            n_select=2,
            patch_per_row=4,
        )

        self.assertEqual(len(chosen), 2)
        self.assertEqual(len(set(chosen)), 2)
        self.assertTrue(all(idx in {0, 1, 2, 3} for idx in chosen))

    def test_registry_instantiates_new_strategy(self):
        strategy = get_strategy(
            "sparsevlm_adaptive_stratified",
            {
                "prune_ratio": 0.5,
                "high_ratio": 0.7,
                "grid_size": 6,
                "patch_per_row": 24,
                "intra_stratum_mode": "random",
            },
        )
        self.assertIsInstance(strategy, SparseVLMAdaptiveStratifiedStrategy)


if __name__ == "__main__":
    unittest.main()
