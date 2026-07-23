import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

from scnd_rep_bottleneck_fig import (  # noqa: E402
    bbox_area_ratio,
    coarse_grid_entropy,
    cosine_similarity_matrix,
    largest_component_ratio,
    mean_grid_distance,
    offdiag_values,
    overlap_metrics,
    pca_axis_limits,
    pca_project_pair,
    pca_spread,
    render_evidence_crop,
    render_pairwise_heatmap_panel,
    render_patch_evidence_overlay_crop,
    review_assessment,
    selected_hidden_by_patch_order,
    selected_hidden_by_saliency_order,
)


class ScndRepBottleneckFigTests(unittest.TestCase):
    def test_spatial_metrics_for_compact_patch_block(self):
        indices = [0, 1, 24, 25]

        self.assertAlmostEqual(coarse_grid_entropy(indices), 0.0)
        self.assertAlmostEqual(bbox_area_ratio(indices), 4 / 576)
        self.assertAlmostEqual(largest_component_ratio(indices), 1.0)
        self.assertGreater(mean_grid_distance(indices), 0.0)
        self.assertLess(mean_grid_distance(indices), 0.05)

    def test_spatial_entropy_separates_distant_cells(self):
        indices = [0, 575]
        expected = math.log(2) / math.log(36)

        self.assertAlmostEqual(coarse_grid_entropy(indices), expected)
        self.assertAlmostEqual(bbox_area_ratio(indices), 1.0)
        self.assertAlmostEqual(largest_component_ratio(indices), 0.5)

    def test_overlap_metrics(self):
        metrics = overlap_metrics([0, 1, 2], [2, 3])

        self.assertEqual(metrics["overlap_count"], 1)
        self.assertAlmostEqual(metrics["jaccard"], 0.25)

    def test_similarity_matrix_is_symmetric_with_unit_diagonal(self):
        hidden = np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        )

        sim = cosine_similarity_matrix(hidden)

        self.assertEqual(sim.shape, (3, 3))
        np.testing.assert_allclose(sim, sim.T, atol=1e-6)
        np.testing.assert_allclose(np.diag(sim), np.ones(3), atol=1e-6)

    def test_selected_hidden_uses_current_patch_mapping_and_raster_order(self):
        hidden = np.asarray(
            [
                [10.0, 0.0],
                [20.0, 0.0],
                [30.0, 0.0],
            ],
            dtype=np.float32,
        )
        stats_row = {
            "layer_2_current_patch_indices": [5, 2, 9],
            "layer_2_keep_patch_indices": [9, 2],
        }

        selected, patches = selected_hidden_by_patch_order(hidden, stats_row, layer=2)

        self.assertEqual(patches, [2, 9])
        np.testing.assert_allclose(selected, np.asarray([[20.0, 0.0], [30.0, 0.0]], dtype=np.float32))

    def test_selected_hidden_can_use_shared_saliency_order(self):
        hidden = np.asarray(
            [
                [10.0, 0.0],
                [20.0, 0.0],
                [30.0, 0.0],
            ],
            dtype=np.float32,
        )
        stats_row = {
            "layer_2_current_patch_indices": [5, 2, 9],
            "layer_2_current_rank_score": [0.2, 0.9, 0.5],
            "layer_2_keep_patch_indices": [9, 2],
        }

        selected, patches, scores = selected_hidden_by_saliency_order(hidden, stats_row, layer=2)

        self.assertEqual(patches, [2, 9])
        self.assertEqual(scores, [0.9, 0.5])
        np.testing.assert_allclose(selected, np.asarray([[20.0, 0.0], [30.0, 0.0]], dtype=np.float32))

    def test_offdiag_values_excludes_diagonal(self):
        sim = np.asarray(
            [
                [1.0, 0.2, 0.3],
                [0.2, 1.0, 0.4],
                [0.3, 0.4, 1.0],
            ],
            dtype=np.float32,
        )

        values = offdiag_values(sim)

        self.assertEqual(values.shape, (6,))
        self.assertNotIn(1.0, values.tolist())

    def test_pairwise_heatmap_masks_diagonal_with_fixed_scale(self):
        sim = np.asarray([[1.0, 0.0], [0.5, 1.0]], dtype=np.float32)

        panel = render_pairwise_heatmap_panel(sim, size=20, vmin=0.0, vmax=1.0)

        self.assertEqual(panel.size, (20, 20))
        self.assertEqual(panel.getpixel((4, 4)), (226, 229, 234))

    def test_pairwise_heatmap_can_render_mean_footer(self):
        sim = np.asarray([[1.0, 0.0], [0.5, 1.0]], dtype=np.float32)

        panel = render_pairwise_heatmap_panel(
            sim,
            size=96,
            vmin=0.0,
            vmax=1.0,
            footer_text="mean cos 0.25",
        )

        self.assertEqual(panel.size, (96, 96))
        footer = np.asarray(panel)[60:, :, :]
        self.assertGreater(int(np.sum(np.any(footer < 250, axis=-1))), 0)

    def test_evidence_crop_can_hide_roi_box(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "white.png"
            Image.new("RGB", (120, 120), "white").save(image_path)

            with_roi = render_evidence_crop(image_path, (6, 6, 18, 18), output_size=96, draw_roi=True)
            without_roi = render_evidence_crop(image_path, (6, 6, 18, 18), output_size=96, draw_roi=False)

        self.assertEqual(with_roi.size, (96, 96))
        self.assertEqual(without_roi.size, (96, 96))
        with_pixels = np.asarray(with_roi)
        without_pixels = np.asarray(without_roi)
        self.assertGreater(int(np.sum(np.all(with_pixels < 60, axis=-1))), 0)
        self.assertEqual(int(np.sum(np.all(without_pixels < 60, axis=-1))), 0)

    def test_retained_patch_overlay_can_hide_roi_box(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "white.png"
            Image.new("RGB", (120, 120), "white").save(image_path)

            with_roi = render_patch_evidence_overlay_crop(
                image_path,
                (6, 6, 18, 18),
                [0, 1, 24],
                color=(220, 40, 40),
                output_size=96,
                draw_roi=True,
            )
            without_roi = render_patch_evidence_overlay_crop(
                image_path,
                (6, 6, 18, 18),
                [0, 1, 24],
                color=(220, 40, 40),
                output_size=96,
                draw_roi=False,
            )

        self.assertEqual(with_roi.size, (96, 96))
        self.assertEqual(without_roi.size, (96, 96))
        self.assertGreater(int(np.sum(np.all(np.asarray(with_roi) < 60, axis=-1))), 0)
        self.assertEqual(int(np.sum(np.all(np.asarray(without_roi) < 60, axis=-1))), 0)

    def test_pca_pair_projection_is_deterministic_and_shared(self):
        topk_hidden = np.asarray(
            [
                [1.00, 0.00, 0.0],
                [1.02, 0.01, 0.0],
                [0.98, -0.01, 0.0],
            ],
            dtype=np.float32,
        )
        scnd_hidden = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ],
            dtype=np.float32,
        )

        first = pca_project_pair(topk_hidden, scnd_hidden)
        second = pca_project_pair(topk_hidden, scnd_hidden)

        np.testing.assert_allclose(first["topk_xy"], second["topk_xy"], atol=1e-6)
        np.testing.assert_allclose(first["scnd_xy"], second["scnd_xy"], atol=1e-6)
        self.assertEqual(first["topk_xy"].shape, (3, 2))
        self.assertEqual(first["scnd_xy"].shape, (3, 2))
        self.assertGreater(pca_spread(first["scnd_xy"]), pca_spread(first["topk_xy"]))
        limits = pca_axis_limits(first["topk_xy"], first["scnd_xy"])
        self.assertEqual(len(limits), 4)
        self.assertLess(limits[0], limits[1])
        self.assertLess(limits[2], limits[3])

    def test_review_assessment_prioritizes_strong_same_answer_cases(self):
        strong = {
            "dataset": "gqa",
            "selection_score": "1.35",
            "entropy_gain": "0.18",
            "grid_distance_gain": "0.08",
            "jaccard": "0.22",
            "topk_spatial_entropy_6x6": "0.65",
            "topk_largest_component_ratio": "0.35",
            "answer_agreement": "True",
        }
        weak = {
            **strong,
            "selection_score": "0.4",
            "entropy_gain": "0.02",
            "grid_distance_gain": "0.01",
            "jaccard": "0.55",
        }

        self.assertEqual(review_assessment(strong, image_exists=True)["priority_tier"], "A")
        self.assertNotEqual(review_assessment(weak, image_exists=True)["priority_tier"], "A")


if __name__ == "__main__":
    unittest.main()
