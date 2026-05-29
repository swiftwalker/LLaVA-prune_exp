from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

from dataset_adapters import MMBENCH_DIRECT_ANSWER_PROMPT, count_dataset_samples, load_dataset_samples


class DatasetAdapterTests(unittest.TestCase):
    def test_textvqa_jsonl_is_loaded_as_normalized_samples(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "textvqa.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "question_id": "003a8a",
                                "image": "003a8a.jpg",
                                "text": "What is the brand?",
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "question_id": "b9dc40",
                                "image": "b9dc40.jpg",
                                "text": "What does the text say?",
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            samples = load_dataset_samples("textvqa", path)
            self.assertEqual(len(samples), 2)
            self.assertEqual(samples[0]["question_id"], "003a8a")
            self.assertEqual(samples[0]["image_path"], "003a8a.jpg")
            self.assertTrue(samples[0]["has_image"])
            self.assertEqual(count_dataset_samples("textvqa", path), 2)

    def test_scienceqa_json_list_supports_text_only_and_multimodal_items(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "scienceqa.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "id": "4",
                            "conversations": [{"from": "human", "value": "Which figure of speech is used?\nA. x\nB. y"}],
                        },
                        {
                            "id": "5",
                            "image": "5/image.png",
                            "conversations": [{"from": "human", "value": "<image>\nWhat is shown?\nA. x\nB. y"}],
                        },
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            samples = load_dataset_samples("scienceqa", path)
            self.assertEqual(len(samples), 2)
            self.assertFalse(samples[0]["has_image"])
            self.assertIsNone(samples[0]["image_path"])
            self.assertEqual(
                samples[0]["text"],
                "Which figure of speech is used?\nA. x\nB. y\n" + MMBENCH_DIRECT_ANSWER_PROMPT,
            )
            self.assertTrue(samples[1]["has_image"])
            self.assertEqual(samples[1]["image_path"], "5/image.png")
            self.assertEqual(
                samples[1]["text"],
                "What is shown?\nA. x\nB. y\n" + MMBENCH_DIRECT_ANSWER_PROMPT,
            )
            self.assertEqual(count_dataset_samples("scienceqa", path), 2)

    def test_mmbench_tsv_is_loaded_with_base64_images_and_direct_answer_prompt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "mmbench.tsv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["index", "question", "hint", "A", "B", "C", "D", "image"],
                    delimiter="\t",
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "index": "241",
                        "question": "Identify the best answer.",
                        "hint": "Read the experiment.",
                        "A": "choice A",
                        "B": "choice B",
                        "C": "",
                        "D": "",
                        "image": "ZmFrZV9pbWFnZQ==",
                    }
                )

            samples = load_dataset_samples("mmbench", path)
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0]["question_id"], "241")
            self.assertTrue(samples[0]["has_image"])
            self.assertIsNone(samples[0]["image_path"])
            self.assertEqual(samples[0]["image_base64"], "ZmFrZV9pbWFnZQ==")
            self.assertIn("Read the experiment.", samples[0]["text"])
            self.assertIn("A. choice A", samples[0]["text"])
            self.assertIn(MMBENCH_DIRECT_ANSWER_PROMPT, samples[0]["text"])
            self.assertEqual(count_dataset_samples("mmbench", path), 1)

    def test_mmvet_jsonl_is_loaded_as_normalized_samples_with_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "mmvet.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "question_id": "v1_0",
                        "image": "v1_0.png",
                        "text": "What is x?",
                        "answer": "-1",
                        "metadata": {"capability": "math"},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            samples = load_dataset_samples("mmvet", path)
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0]["question_id"], "v1_0")
            self.assertEqual(samples[0]["image_path"], "v1_0.png")
            self.assertTrue(samples[0]["has_image"])
            self.assertEqual(samples[0]["answer"], "-1")
            self.assertEqual(samples[0]["metadata"]["capability"], "math")
            self.assertEqual(count_dataset_samples("mmvet", path), 1)

    def test_ai2d_jsonl_is_loaded_as_normalized_samples(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ai2d.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "question_id": "ai2d_000000",
                        "image": "ai2d_000000.png",
                        "text": "Which option?\nA. cat\nB. dog\n" + MMBENCH_DIRECT_ANSWER_PROMPT,
                        "answer": "A",
                        "metadata": {"options": ["cat", "dog"]},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            samples = load_dataset_samples("ai2d", path)
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0]["question_id"], "ai2d_000000")
            self.assertEqual(samples[0]["image_path"], "ai2d_000000.png")
            self.assertTrue(samples[0]["has_image"])
            self.assertEqual(samples[0]["answer"], "A")
            self.assertIn(MMBENCH_DIRECT_ANSWER_PROMPT, samples[0]["text"])
            self.assertEqual(count_dataset_samples("ai2d", path), 1)


if __name__ == "__main__":
    unittest.main()
