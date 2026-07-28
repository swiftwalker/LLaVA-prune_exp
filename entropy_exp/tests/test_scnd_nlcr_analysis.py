import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

import scnd_nlcr_analysis as analysis  # noqa: E402
from scnd_nlcr_analysis import (  # noqa: E402
    UtilityScorer,
    build_parser,
    compare_answers_exact,
    full_metric_map,
    validate_aligned_rows,
)


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


class AlignmentTests(unittest.TestCase):
    def test_answers_and_stats_align_without_stats_prompt(self):
        answers = [
            {"question_id": "a", "prompt": "first"},
            {"question_id": "b", "prompt": "second"},
        ]
        stats = [
            {"question_id": "a", "sample_idx": 0},
            {"question_id": "b", "sample_idx": 1},
        ]
        validate_aligned_rows(answers, stats, label="toy")

    def test_alignment_rejects_reordered_rows(self):
        answers = [{"question_id": "a"}, {"question_id": "b"}]
        stats = [{"question_id": "b", "sample_idx": 0}, {"question_id": "a", "sample_idx": 1}]
        with self.assertRaisesRegex(ValueError, "alignment mismatch"):
            validate_aligned_rows(answers, stats, label="toy")

    def test_exact_answer_comparison_reports_payload_change(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference = root / "reference.jsonl"
            candidate = root / "candidate.jsonl"
            _write_jsonl(reference, [{"question_id": "1", "prompt": "q", "text": "yes"}])
            _write_jsonl(candidate, [{"question_id": "1", "prompt": "q", "text": "no"}])
            result = compare_answers_exact(reference, candidate)
            self.assertFalse(result["exact"])
            self.assertEqual(result["mismatch_count"], 1)


class UtilityScorerTests(unittest.TestCase):
    def test_gqa_correctness_uses_open_answer_normalization(self):
        scorer = UtilityScorer("gqa")
        qid, row = next(iter(scorer.data.items()))
        answer = {"question_id": qid, "text": f"{row['answer']}."}
        self.assertEqual(scorer.score(answer), 1.0)

    def test_pope_correctness_matches_official_yes_no_parser(self):
        scorer = UtilityScorer("pope")
        qid, label = next(iter(scorer.data.items()))
        prediction = "Yes, clearly." if label else "No, there is not."
        self.assertEqual(scorer.score({"question_id": qid, "text": prediction}), 1.0)

    def test_scienceqa_correctness_uses_option_letter(self):
        scorer = UtilityScorer("scienceqa")
        qid, problem = next(iter(scorer.data.items()))
        letter = "ABCDE"[int(problem["answer"])]
        self.assertEqual(scorer.score({"question_id": qid, "text": letter}), 1.0)


class RouteGateInterfaceTests(unittest.TestCase):
    def test_route_parser_accepts_full_reference_root(self):
        args = build_parser().parse_args(
            [
                "route-gate",
                "--legacy-root",
                "legacy",
                "--route-root",
                "route",
                "--output-dir",
                "report",
                "--full-reference-root",
                "full",
            ]
        )
        self.assertEqual(args.full_reference_root, Path("full"))
        self.assertIsNone(args.full_metric_json)

    def test_full_metric_map_reads_evaluated_runs(self):
        values = {"gqa": 61.0, "pope": 87.0}

        def fake_one_run(root, dataset, *, name_contains="", require_eval=False):
            self.assertTrue(require_eval)
            return Path(dataset)

        with (
            mock.patch.object(analysis, "DATASETS", tuple(values)),
            mock.patch.object(analysis, "one_run", side_effect=fake_one_run),
            mock.patch.object(
                analysis,
                "_metric_from_eval",
                side_effect=lambda run: ("metric", values[run.name]),
            ),
        ):
            self.assertEqual(full_metric_map(Path("full")), values)


if __name__ == "__main__":
    unittest.main()
