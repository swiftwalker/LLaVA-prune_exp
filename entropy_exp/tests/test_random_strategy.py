import os
import sys
import unittest

import torch

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

from strategies.random import RandomStrategy


class RandomStrategyTests(unittest.TestCase):
    def test_same_seed_produces_same_keep_indices(self):
        strategy = RandomStrategy({"prune_ratio_map": {2: 0.25}})

        torch.manual_seed(123)
        keep_a, info_a = strategy.compute_keep_mask(
            None, 0, 8, 8, 2, device=torch.device("cpu")
        )

        torch.manual_seed(123)
        keep_b, info_b = strategy.compute_keep_mask(
            None, 0, 8, 8, 2, device=torch.device("cpu")
        )

        self.assertTrue(torch.equal(keep_a, keep_b))
        self.assertEqual(info_a["keep_indices"].tolist(), info_b["keep_indices"].tolist())
        self.assertEqual(info_a["importance_scores"].tolist(), info_b["importance_scores"].tolist())

    def test_prune_ratio_controls_keep_count_and_sorted_indices(self):
        strategy = RandomStrategy({"prune_ratio_map": {5: 0.5}})

        torch.manual_seed(7)
        keep_indices, info = strategy.compute_keep_mask(
            None, 0, 10, 10, 5, device=torch.device("cpu")
        )

        self.assertEqual(len(keep_indices), 5)
        self.assertEqual(info["num_visual_before"], 10)
        self.assertEqual(info["num_visual_after"], 5)
        self.assertEqual(info["num_pruned"], 5)
        self.assertTrue(torch.equal(keep_indices, keep_indices.sort().values))
        self.assertTrue(torch.all((keep_indices >= 0) & (keep_indices < 10)))
        self.assertEqual(len(info["pruned_indices"]), 5)
        self.assertTrue(all(idx not in info["keep_indices"] for idx in info["pruned_indices"]))

    def test_zero_prune_ratio_keeps_all_tokens(self):
        strategy = RandomStrategy({"prune_ratio_map": {1: 0.0}})

        torch.manual_seed(9)
        keep_indices, info = strategy.compute_keep_mask(
            None, 0, 6, 6, 1, device=torch.device("cpu")
        )

        self.assertEqual(len(keep_indices), 6)
        self.assertEqual(info["num_pruned"], 0)
        self.assertEqual(info["keep_indices"].tolist(), list(range(6)))
        self.assertEqual(info["pruned_indices"].tolist(), [])


if __name__ == "__main__":
    unittest.main()
