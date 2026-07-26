import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

from strategies.scnd_entropy_calibration import (  # noqa: E402
    ENTROPY_DEFINITION,
    apply_entropy_calibration,
    load_calibration_artifact,
    model_config_fingerprint,
)


class SCNDEntropyCalibrationTests(unittest.TestCase):
    def _artifact(self, path: Path, fingerprint: str) -> None:
        payload = {
            "schema_version": 1,
            "mode": "quantile_affine",
            "entropy_definition": ENTROPY_DEFINITION,
            "model_name": "llava-v1.6-vicuna-7b",
            "model_config_fingerprint": fingerprint,
            "capture_layer": 2,
            "quantiles": {
                "low_probability": 0.05,
                "high_probability": 0.95,
                "low_value": 0.60,
                "high_value": 0.80,
            },
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_identity_preserves_raw_entropy(self):
        result = apply_entropy_calibration(0.72, "identity", None)
        self.assertEqual(result["control_value"], 0.72)
        self.assertFalse(result["clipped_low"])
        self.assertFalse(result["clipped_high"])

    def test_quantile_affine_maps_and_clips(self):
        artifact = {
            "quantiles": {"low_value": 0.60, "high_value": 0.80},
            "_resolved_path": "/tmp/calibration.json",
        }
        self.assertAlmostEqual(
            apply_entropy_calibration(0.70, "quantile_affine", artifact)["control_value"],
            0.5,
        )
        self.assertEqual(apply_entropy_calibration(0.50, "quantile_affine", artifact)["control_value"], 0.0)
        self.assertEqual(apply_entropy_calibration(0.90, "quantile_affine", artifact)["control_value"], 1.0)

    def test_artifact_is_bound_to_model_and_layer(self):
        metadata = {
            "model_type": "llava",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "image_aspect_ratio": "anyres",
            "mm_patch_merge_type": "spatial_unpad",
        }
        fingerprint = model_config_fingerprint(metadata)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "calibration.json"
            self._artifact(path, fingerprint)
            loaded = load_calibration_artifact(
                path,
                expected_model_name="llava-v1.6-vicuna-7b",
                expected_model_fingerprint=fingerprint,
                expected_layer=2,
            )
            self.assertEqual(loaded["quantiles"]["low_value"], 0.60)

            with self.assertRaisesRegex(ValueError, "fingerprint"):
                load_calibration_artifact(path, expected_model_fingerprint="wrong")
            with self.assertRaisesRegex(ValueError, "layer mismatch"):
                load_calibration_artifact(path, expected_layer=3)


if __name__ == "__main__":
    unittest.main()
