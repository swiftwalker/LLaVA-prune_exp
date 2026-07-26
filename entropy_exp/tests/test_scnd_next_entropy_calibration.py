import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

from scnd_next_entropy_calibration import (  # noqa: E402
    build_calibration_artifact,
    iter_top_level_json_object,
    prepare_gqa_calibration_subset,
)


class SCNDNextEntropyCalibrationTests(unittest.TestCase):
    def test_streaming_json_and_stratified_unique_image_subset(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "questions.json"
            images = root / "images"
            images.mkdir()
            questions = {}
            for index in range(8):
                image_id = f"image_{index // 2}"
                (images / f"{image_id}.jpg").touch()
                questions[f"q{index}"] = {
                    "imageId": image_id,
                    "question": f"Question {index}?",
                    "answer": "yes",
                    "types": {
                        "structural": "verify" if index % 2 else "query",
                        "semantic": "obj" if index < 4 else "attr",
                    },
                }
            source.write_text(json.dumps(questions), encoding="utf-8")
            self.assertEqual(len(list(iter_top_level_json_object(source, chunk_size=17))), 8)

            output = root / "subset.jsonl"
            manifest_path = root / "manifest.json"
            manifest = prepare_gqa_calibration_subset(
                source=source,
                image_folder=images,
                output=output,
                manifest_path=manifest_path,
                target_images=4,
                seed=42,
                shard_count=3,
            )
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 4)
            self.assertEqual(len({row["image"] for row in rows}), 4)
            self.assertEqual(manifest["selected_unique_images"], 4)
            self.assertEqual([shard["sample_count"] for shard in manifest["shards"]], [2, 1, 1])
            shard_images = []
            for shard in manifest["shards"]:
                shard_rows = [
                    json.loads(line)
                    for line in Path(shard["path"]).read_text(encoding="utf-8").splitlines()
                ]
                shard_images.extend(row["image"] for row in shard_rows)
            self.assertCountEqual(shard_images, [row["image"] for row in rows])

    def test_build_artifact_uses_layer_entropy_quantiles(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            run_dir.mkdir()
            config = {
                "model": {"name": "toy-next"},
                "model_config_metadata": {
                    "model_type": "llava",
                    "hidden_size": 8,
                    "num_hidden_layers": 4,
                    "image_aspect_ratio": "anyres",
                    "mm_patch_merge_type": "spatial_unpad",
                },
                "_run_meta": {"dataset": "gqa"},
            }
            (run_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
            with (run_dir / "stats.jsonl").open("w", encoding="utf-8") as handle:
                for index, value in enumerate((0.60, 0.65, 0.75, 0.80)):
                    handle.write(json.dumps({"question_id": index, "layer_2_saliency_entropy_norm": value}) + "\n")

            output = Path(tmp_dir) / "artifact.json"
            artifact = build_calibration_artifact(run_dir, output, layer=2, q_low_probability=0.25, q_high_probability=0.75)
            self.assertEqual(artifact["model_name"], "toy-next")
            self.assertEqual(artifact["calibration_source"]["sample_count"], 4)
            self.assertLess(artifact["quantiles"]["low_value"], artifact["quantiles"]["high_value"])
            self.assertTrue(output.is_file())

    def test_build_artifact_combines_disjoint_shards(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            run_dirs = []
            for shard, values in enumerate(((0.60, 0.65), (0.75, 0.80))):
                run_dir = root / f"run_{shard}"
                run_dir.mkdir()
                config = {
                    "model": {"name": "toy-next"},
                    "model_config_metadata": {
                        "model_type": "llava",
                        "hidden_size": 8,
                        "num_hidden_layers": 4,
                    },
                    "_run_meta": {"dataset": "gqa"},
                }
                (run_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
                with (run_dir / "stats.jsonl").open("w", encoding="utf-8") as handle:
                    for index, value in enumerate(values):
                        handle.write(
                            json.dumps(
                                {
                                    "question_id": f"s{shard}-{index}",
                                    "layer_2_saliency_entropy_norm": value,
                                }
                            )
                            + "\n"
                        )
                run_dirs.append(run_dir)

            artifact = build_calibration_artifact(
                run_dirs,
                root / "artifact.json",
                layer=2,
                q_low_probability=0.25,
                q_high_probability=0.75,
            )
            self.assertEqual(artifact["calibration_source"]["sample_count"], 4)
            self.assertEqual(artifact["calibration_source"]["unique_question_count"], 4)
            self.assertEqual(len(artifact["calibration_source"]["runs"]), 2)


if __name__ == "__main__":
    unittest.main()
