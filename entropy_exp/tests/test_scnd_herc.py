import os
import sys
import unittest

import torch


SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

from strategies.scnd_herc import (  # noqa: E402
    _facility_pair_coverage,
    _facility_state,
    normalize_rater_distributions,
    reconcile_evidence,
    restrict_reference_distributions,
)
from strategies.scnd_visual_roles import (  # noqa: E402
    ROLE_LOCAL_PATCH,
    ROLE_NEWLINE,
    VIEW_BASE,
    VIEW_LOCAL,
    visual_token_structure_ids,
)
from strategies.sparsevlm import (  # noqa: E402
    compute_rater_visual_scores_from_attention,
    compute_visual_scores_from_attention,
)


def _square_structure(num_tokens):
    return visual_token_structure_ids(
        torch.arange(num_tokens),
        {
            "layout_kind": "square",
            "base_token_count": num_tokens,
            "local_patch_token_count": 0,
            "newline_token_count": 0,
            "local_grid_width": 0,
            "total_token_count": num_tokens,
        },
    )


class SCNDHERCUnitTests(unittest.TestCase):
    def test_per_rater_scores_average_to_legacy_score(self):
        attention = torch.zeros((1, 2, 7, 7), dtype=torch.float32)
        attention[0, 0, 5, 1:5] = torch.tensor([0.8, 0.1, 0.05, 0.05])
        attention[0, 1, 5, 1:5] = torch.tensor([0.6, 0.2, 0.1, 0.1])
        attention[0, 0, 6, 1:5] = torch.tensor([0.1, 0.1, 0.3, 0.5])
        attention[0, 1, 6, 1:5] = torch.tensor([0.2, 0.1, 0.3, 0.4])
        kwargs = {
            "attn_weights": attention,
            "rater_indices": torch.tensor([0, 1]),
            "text_positions": torch.tensor([5, 6]),
            "visual_positions": torch.tensor([1, 2, 3, 4]),
        }

        per_rater = compute_rater_visual_scores_from_attention(**kwargs)
        legacy = compute_visual_scores_from_attention(**kwargs)

        self.assertEqual(tuple(per_rater.shape), (2, 4))
        torch.testing.assert_close(per_rater.mean(dim=0), legacy, rtol=0, atol=0)

    def test_query_reconciliation_recovers_evidence_hidden_by_mean_rater_score(self):
        distributions = normalize_rater_distributions(
            torch.tensor(
                [
                    [0.49, 0.49, 0.01, 0.01],
                    [0.01, 0.01, 0.49, 0.49],
                ]
            )
        )
        result = reconcile_evidence(
            mode="herc_v1",
            legacy_keep_indices=torch.tensor([0, 1]),
            scalar_score=torch.ones(4),
            current_distributions=distributions,
            reference_distributions=None,
            current_visual_embeds=torch.eye(4),
            current_original_indices=torch.arange(4),
            structure_ids=_square_structure(4),
            tau=0.65,
            profile_pressure=1.0,
            candidate_pool_multiplier=2.0,
            max_swap_ratio=0.5,
        )

        keep = result["keep_indices"].tolist()
        stats = result["stats"]
        self.assertEqual(keep, [1, 2])
        self.assertEqual(stats["evidence_reconcile_query_swap_count"], 1)
        self.assertGreater(
            stats["evidence_query_coverage_after_query"],
            stats["evidence_query_coverage_before"],
        )

    def test_audit_only_preserves_proposal_indices(self):
        distributions = normalize_rater_distributions(torch.rand(3, 9, generator=torch.Generator().manual_seed(7)))
        legacy = torch.tensor([0, 3, 7])
        result = reconcile_evidence(
            mode="audit_only",
            legacy_keep_indices=legacy,
            scalar_score=torch.linspace(1.0, 0.1, 9),
            current_distributions=distributions,
            reference_distributions=None,
            current_visual_embeds=torch.eye(9),
            current_original_indices=torch.arange(9),
            structure_ids=_square_structure(9),
            tau=0.65,
            profile_pressure=1.0,
            candidate_pool_multiplier=2.0,
            max_swap_ratio=0.5,
        )

        self.assertEqual(result["keep_indices"].tolist(), legacy.tolist())
        self.assertFalse(result["stats"]["evidence_reconcile_applied"])
        self.assertEqual(result["stats"]["evidence_reconcile_query_swap_count"], 0)
        self.assertEqual(result["stats"]["evidence_reconcile_context_swap_count"], 0)

    def test_vectorized_facility_pair_coverage_matches_brute_force(self):
        torch.manual_seed(11)
        embeds = torch.randn(7, 5)
        saliency = torch.tensor([0.4, 0.2, 0.1, 0.8, 0.5, 0.3, 0.6])
        pool = torch.arange(7)
        selected = torch.tensor([0, 3, 5])
        incoming = torch.tensor([1, 2, 4, 6])
        outgoing = selected
        facility = _facility_state(embeds, pool, saliency, selected)

        pair_coverage = _facility_pair_coverage(
            facility,
            selected,
            incoming,
            outgoing,
        )
        expected = torch.empty_like(pair_coverage)
        for incoming_column, incoming_index in enumerate(incoming.tolist()):
            for outgoing_column, outgoing_index in enumerate(outgoing.tolist()):
                swapped = torch.tensor(
                    [index for index in selected.tolist() if index != outgoing_index] + [incoming_index]
                )
                expected[incoming_column, outgoing_column] = _facility_state(
                    embeds,
                    pool,
                    saliency,
                    swapped,
                )["coverage"]

        torch.testing.assert_close(pair_coverage, expected, rtol=1e-6, atol=1e-6)

    def test_context_reconciliation_repairs_view_collapse(self):
        structure = visual_token_structure_ids(
            torch.arange(8),
            {
                "layout_kind": "anyres_flat",
                "base_token_count": 4,
                "local_patch_token_count": 4,
                "newline_token_count": 0,
                "local_grid_width": 0,
                "total_token_count": 8,
            },
        )
        result = reconcile_evidence(
            mode="herc_v1",
            legacy_keep_indices=torch.tensor([0, 1, 2, 3]),
            scalar_score=torch.ones(8),
            current_distributions=torch.full((2, 8), 1.0 / 8.0),
            reference_distributions=None,
            current_visual_embeds=torch.eye(8),
            current_original_indices=torch.arange(8),
            structure_ids=structure,
            tau=0.65,
            profile_pressure=1.0,
            candidate_pool_multiplier=2.0,
            max_swap_ratio=0.5,
        )

        stats = result["stats"]
        self.assertGreater(stats["evidence_reconcile_context_swap_count"], 0)
        self.assertGreater(stats["evidence_view_coverage_after"], stats["evidence_view_coverage_before"])
        selected_views = structure["view_ids"].index_select(0, result["keep_indices"])
        self.assertIn(VIEW_BASE, selected_views.tolist())
        self.assertIn(VIEW_LOCAL, selected_views.tolist())

    def test_first_c_reference_is_restricted_and_used_for_continuity(self):
        first_c = normalize_rater_distributions(
            torch.tensor([[0.02, 0.02, 0.48, 0.48]])
        )
        survivor_indices = torch.tensor([0, 1, 2, 3])
        reference = restrict_reference_distributions(first_c, survivor_indices)
        current = normalize_rater_distributions(
            torch.tensor([[0.48, 0.48, 0.02, 0.02]])
        )
        result = reconcile_evidence(
            mode="herc_v1",
            legacy_keep_indices=torch.tensor([0, 1]),
            scalar_score=torch.ones(4),
            current_distributions=current,
            reference_distributions=reference,
            current_visual_embeds=torch.eye(4),
            current_original_indices=survivor_indices,
            structure_ids=_square_structure(4),
            tau=0.65,
            profile_pressure=1.0,
            candidate_pool_multiplier=2.0,
            max_swap_ratio=0.5,
        )

        stats = result["stats"]
        self.assertGreater(
            stats["evidence_query_coverage_after_context"],
            stats["evidence_query_coverage_before"],
        )
        self.assertTrue(any(index >= 2 for index in result["keep_indices"].tolist()))

    def test_hard_scalar_and_local_constraints_survive_swaps_deterministically(self):
        structure = visual_token_structure_ids(
            torch.arange(8),
            {
                "layout_kind": "anyres_flat",
                "base_token_count": 4,
                "local_patch_token_count": 4,
                "newline_token_count": 0,
                "local_grid_width": 0,
                "total_token_count": 8,
            },
        )
        kwargs = {
            "mode": "herc_v1",
            "legacy_keep_indices": torch.tensor([0, 1, 4, 5]),
            "scalar_score": torch.tensor([1.0, 0.9, 0.8, 0.7, 0.95, 0.85, 0.1, 0.05]),
            "current_distributions": normalize_rater_distributions(
                torch.tensor(
                    [
                        [0.4, 0.3, 0.1, 0.05, 0.05, 0.04, 0.03, 0.03],
                        [0.03, 0.03, 0.04, 0.05, 0.35, 0.30, 0.10, 0.10],
                    ]
                )
            ),
            "reference_distributions": None,
            "current_visual_embeds": torch.eye(8),
            "current_original_indices": torch.arange(8),
            "structure_ids": structure,
            "tau": 0.9,
            "profile_pressure": 1.0,
            "candidate_pool_multiplier": 2.0,
            "max_swap_ratio": 0.5,
            "local_floor_count": 2,
        }

        first = reconcile_evidence(**kwargs)
        second = reconcile_evidence(**kwargs)
        torch.testing.assert_close(first["keep_indices"], second["keep_indices"], rtol=0, atol=0)
        self.assertEqual(first["keep_indices"].numel(), 4)
        selected_roles = structure["role_ids"].index_select(0, first["keep_indices"])
        self.assertGreaterEqual(int((selected_roles == ROLE_LOCAL_PATCH).sum().item()), 2)
        self.assertGreaterEqual(
            first["stats"]["evidence_scalar_mass_after"] + 1e-7,
            first["stats"]["evidence_scalar_mass_floor"],
        )

    def test_anyres_spatial_structure_ids_handle_newline_tokens(self):
        structure = visual_token_structure_ids(
            torch.arange(14),
            {
                "layout_kind": "anyres_spatial_unpad",
                "base_token_count": 4,
                "local_patch_token_count": 8,
                "newline_token_count": 2,
                "local_grid_width": 4,
                "local_grid_height": 2,
                "crop_grid_width": 2,
                "crop_grid_height": 1,
                "total_token_count": 14,
            },
        )

        self.assertEqual(structure["view_ids"][:4].tolist(), [VIEW_BASE] * 4)
        self.assertEqual(structure["view_ids"][4:].tolist(), [VIEW_LOCAL] * 10)
        self.assertEqual(structure["role_ids"][[8, 13]].tolist(), [ROLE_NEWLINE, ROLE_NEWLINE])
        self.assertFalse(bool(structure["patch_mask"][[8, 13]].any().item()))
        self.assertEqual(structure["macrocell_ids"][[4, 5, 6, 7]].tolist(), [1, 1, 2, 2])
        self.assertEqual(structure["row_ids"][[4, 9]].tolist(), [2, 3])


if __name__ == "__main__":
    unittest.main()
