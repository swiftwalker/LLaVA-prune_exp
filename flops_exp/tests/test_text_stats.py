import os
import sys
import unittest

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

from text_stats.compute_stats import summarize_text_records
from text_stats.schema import SampleTextRecord


class TextStatsTests(unittest.TestCase):
    def test_summarize_text_records_keeps_template_overhead_separate(self):
        records = [
            SampleTextRecord(
                sample_idx=0,
                dataset_name="mme",
                question_id="a",
                image_file="a.png",
                raw_text="hello",
                raw_text_token_len=10,
                template_overhead_tokens=20,
                templated_input_ids_len_pre_mm=30,
                image_token_len=576,
                prefill_seq_len=605,
            ),
            SampleTextRecord(
                sample_idx=1,
                dataset_name="mme",
                question_id="b",
                image_file="b.png",
                raw_text="world",
                raw_text_token_len=14,
                template_overhead_tokens=20,
                templated_input_ids_len_pre_mm=34,
                image_token_len=576,
                prefill_seq_len=609,
            ),
        ]

        summary = summarize_text_records(
            records=records,
            dataset_name="mme",
            tokenizer_name="tok",
            conv_mode="vicuna_v1",
            canonical_template_overhead_tokens=20,
            image_token_len=576,
        )

        self.assertEqual(summary.raw_text_token_len_mean, 12.0)
        self.assertEqual(summary.template_overhead_tokens, 20)
        self.assertEqual(summary.image_token_len, 576)


if __name__ == "__main__":
    unittest.main()
