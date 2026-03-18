import os
import sys
import unittest

import torch

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

from strategies.sparsevlm import (
    compute_visual_scores_from_attention,
    prune_visual_tokens,
    select_text_raters,
)


class SparseVLMHelperTests(unittest.TestCase):
    def test_select_text_raters_threshold_selection(self):
        H_v = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        H_q = torch.tensor([[1.0, 0.0], [0.8, 0.0], [0.0, 1.0]])

        rater_indices, relevance_scores = select_text_raters(H_v, H_q)

        self.assertEqual(rater_indices.tolist(), [0, 1])
        self.assertEqual(tuple(relevance_scores.shape), (3,))

    def test_select_text_raters_respects_special_token_mask(self):
        H_v = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        H_q = torch.tensor([[1.0, 0.0], [0.8, 0.0], [0.0, 1.0]])
        special_mask = torch.tensor([True, False, True])

        rater_indices, _ = select_text_raters(H_v, H_q, special_token_mask=special_mask)

        self.assertEqual(rater_indices.tolist(), [1])

    def test_select_text_raters_falls_back_when_all_candidates_masked(self):
        H_v = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        H_q = torch.tensor([[1.0, 0.0], [0.8, 0.0], [0.0, 1.0]])
        special_mask = torch.tensor([True, True, True])

        rater_indices, _ = select_text_raters(
            H_v, H_q, special_token_mask=special_mask, fallback_topk=4
        )

        self.assertEqual(rater_indices.tolist(), [0, 1, 2])

    def test_compute_visual_scores_from_attention_handles_batched_attention(self):
        attn = torch.zeros((1, 2, 5, 5), dtype=torch.float32)
        attn[0, 0, 3, 1:3] = torch.tensor([0.2, 0.8])
        attn[0, 0, 4, 1:3] = torch.tensor([0.6, 0.4])
        attn[0, 1, 3, 1:3] = torch.tensor([0.4, 0.6])
        attn[0, 1, 4, 1:3] = torch.tensor([0.2, 0.8])

        scores = compute_visual_scores_from_attention(
            attn_weights=attn,
            rater_indices=torch.tensor([0, 1]),
            text_positions=torch.tensor([3, 4]),
            visual_positions=torch.tensor([1, 2]),
        )

        expected = torch.tensor([0.35, 0.65])
        self.assertTrue(torch.allclose(scores, expected, atol=1e-6))

    def test_compute_visual_scores_from_attention_handles_head_only_attention(self):
        attn = torch.zeros((2, 5, 5), dtype=torch.float32)
        attn[0, 4, 1:3] = torch.tensor([0.9, 0.1])
        attn[1, 4, 1:3] = torch.tensor([0.5, 0.5])

        scores = compute_visual_scores_from_attention(
            attn_weights=attn,
            rater_indices=torch.tensor([1]),
            text_positions=torch.tensor([3, 4]),
            visual_positions=torch.tensor([1, 2]),
        )

        expected = torch.tensor([0.7, 0.3])
        self.assertTrue(torch.allclose(scores, expected, atol=1e-6))

    def test_prune_visual_tokens_respects_minimum_remaining_tokens(self):
        visual_embeds = torch.arange(12, dtype=torch.float32).view(4, 3)
        visual_scores = torch.tensor([0.1, 0.4, 0.2, 0.8])

        pruned_embeds, keep_indices, pruned_indices = prune_visual_tokens(
            current_visual_embeds=visual_embeds,
            visual_scores=visual_scores,
            prune_ratio=0.5,
            min_visual_tokens_after_prune=2,
        )

        self.assertEqual(keep_indices.tolist(), [1, 3])
        self.assertEqual(pruned_indices.tolist(), [0, 2])
        self.assertEqual(tuple(pruned_embeds.shape), (2, 3))

    def test_prune_visual_tokens_noop_when_minimum_floor_blocks_pruning(self):
        visual_embeds = torch.arange(12, dtype=torch.float32).view(4, 3)
        visual_scores = torch.tensor([0.1, 0.4, 0.2, 0.8])

        pruned_embeds, keep_indices, pruned_indices = prune_visual_tokens(
            current_visual_embeds=visual_embeds,
            visual_scores=visual_scores,
            prune_ratio=0.9,
            min_visual_tokens_after_prune=4,
        )

        self.assertTrue(torch.equal(pruned_embeds, visual_embeds))
        self.assertEqual(keep_indices.tolist(), [0, 1, 2, 3])
        self.assertEqual(pruned_indices.tolist(), [])

    def test_prune_visual_tokens_rejects_invalid_ratio(self):
        visual_embeds = torch.arange(6, dtype=torch.float32).view(2, 3)
        visual_scores = torch.tensor([0.1, 0.2])

        with self.assertRaises(ValueError):
            prune_visual_tokens(
                current_visual_embeds=visual_embeds,
                visual_scores=visual_scores,
                prune_ratio=1.5,
                min_visual_tokens_after_prune=1,
            )


if __name__ == "__main__":
    unittest.main()
