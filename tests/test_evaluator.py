import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import Evaluate_scenario as evaluator
import Fine_tuned_judge as student


class FakeJudge:
    backend = "distilkobert_student"
    model_name = "/tmp/best_student.pt"
    checkpoint_sha256 = "test-checkpoint-sha256"
    base_model = "monologg/distilkobert"
    base_revision = "test-revision"
    threshold = 0.5

    def predict(self, scenario: str) -> dict:
        return {
            "backend": self.backend,
            "model": self.model_name,
            "prediction": 1,
            "label": "voice_phishing",
            "vp_probability": 0.8,
            "confidence": 0.8,
            "probabilities": {"normal": 0.2, "voice_phishing": 0.8},
            "threshold": 0.5,
            "prompt_used": False,
        }


class FakeNormalJudge(FakeJudge):
    def predict(self, scenario: str) -> dict:
        result = super().predict(scenario)
        result.update({
            "prediction": 0,
            "label": "normal",
            "vp_probability": 0.3,
            "confidence": 0.7,
            "probabilities": {"normal": 0.7, "voice_phishing": 0.3},
        })
        return result


def kev_response(score: float = 2.0) -> dict:
    return {
        "answers": {
            key: {
                "score": score,
                "confidence": 0.75,
                "probabilities": [0.05, 0.15, 0.6, 0.15, 0.05],
                "legend": {str(i): text for i, text in enumerate(question["criteria"])},
            }
            for key, question in evaluator.KEV_QUESTIONS.items()
        }
    }


class CheckpointManifestTest(unittest.TestCase):
    def test_checkpoint_matches_published_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "models" / "realism" / "best_student.pt"
            model_path.parent.mkdir(parents=True)
            model_path.write_bytes(b"fine-tuned-checkpoint")
            sha256 = student.file_sha256(model_path)
            manifest_path = root / "model_manifest.json"
            manifest_path.write_text(json.dumps({
                "realism_student": {
                    "path": "models/realism/best_student.pt",
                    "sha256": sha256,
                }
            }))

            with patch.object(student, "SCRIPT_DIR", root.resolve()), patch.object(
                student, "MODEL_MANIFEST_PATH", manifest_path
            ):
                judge = student.DistilKoBERTJudge("models/realism/best_student.pt")

        self.assertEqual(judge.model_name, "models/realism/best_student.pt")
        self.assertEqual(judge.checkpoint_sha256, sha256)

    def test_checkpoint_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "models" / "realism" / "best_student.pt"
            model_path.parent.mkdir(parents=True)
            model_path.write_bytes(b"wrong-checkpoint")
            manifest_path = root / "model_manifest.json"
            manifest_path.write_text(json.dumps({
                "realism_student": {
                    "path": "models/realism/best_student.pt",
                    "sha256": "expected-sha256",
                }
            }))

            with patch.object(student, "SCRIPT_DIR", root.resolve()), patch.object(
                student, "MODEL_MANIFEST_PATH", manifest_path
            ), self.assertRaisesRegex(ValueError, "SHA-256"):
                student.DistilKoBERTJudge("models/realism/best_student.pt")


class ScenarioEvaluatorTest(unittest.TestCase):
    def test_missing_reference_pdf_is_optional(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "Scenario.pdf"
            self.assertEqual(evaluator.load_optional_reference(missing, 12000), "")

    def test_agreed_rubric_numbering_and_weights(self) -> None:
        self.assertEqual(
            list(evaluator.WEIGHTS),
            ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "C1", "C2", "D1", "D2"],
        )
        self.assertEqual(evaluator.AXIS_MAX, {"A": 30, "B": 25, "C": 20, "D": 15})
        self.assertIn("방어 공백", evaluator.KEV_QUESTIONS["A4"]["instructions"])
        self.assertIn("피해자", evaluator.KEV_QUESTIONS["B2"]["instructions"])
        self.assertIn("최초 접촉부터 최종 피해", evaluator.KEV_QUESTIONS["C1"]["instructions"])
        self.assertTrue(all("대화에 실제로 나타난 근거만" in question["instructions"] for question in evaluator.KEV_QUESTIONS.values()))
        self.assertNotIn("B4", evaluator.KEV_QUESTIONS)
        self.assertEqual(len(evaluator.KEV_QUESTIONS), 11)
        self.assertTrue(all(len(question["criteria"]) == 5 for question in evaluator.KEV_QUESTIONS.values()))

    def test_kev_score_is_normalized_and_student_adds_realism(self) -> None:
        with patch.object(evaluator, "call_kev", return_value=kev_response(2.0)):
            result = evaluator.evaluate_text(
                "테스트 시나리오",
                api_url="http://kev:8009",
                model="kev-latest",
                fine_tuned_judge=FakeJudge(),
            )

        self.assertEqual(result["ratings"]["A1"], 3.0)
        self.assertEqual(result["axis_scores"], {"A": 15.0, "B": 12.5, "C": 10.0, "D": 7.5})
        self.assertEqual(result["realism_prediction"], 1)
        self.assertEqual(result["realism_probability"], 0.8)
        expected = 100 * (0.8**0.30) * (0.5**0.25) * (0.5**0.15) * (0.5**0.15) * (0.5**0.15)
        self.assertAlmostEqual(result["ranking_score"], expected, places=5)
        self.assertFalse(result["fine_tuned_judgment"]["prompt_used"])
        self.assertIn(result["weakest_kev_axis"], evaluator.AXES)
        self.assertIn(result["weakest_kev_subcriterion"], evaluator.WEIGHTS)

    def test_ranking_explicitly_includes_generalizability(self) -> None:
        base = {"A": 0.8, "B": 0.8, "C": 0.8, "D": 0.8}
        score_high, _ = evaluator.calculate_ranking(0.8, base)
        score_low, _ = evaluator.calculate_ranking(0.8, {**base, "D": 0.2})
        self.assertLess(score_low, score_high)
        self.assertEqual(
            evaluator.RANKING_WEIGHTS,
            {"R": 0.30, "A": 0.25, "B": 0.15, "C": 0.15, "D": 0.15},
        )

    def test_binary_zero_is_ranked_instead_of_filtered(self) -> None:
        with patch.object(evaluator, "call_kev", return_value=kev_response(3.0)):
            result = evaluator.evaluate_text("일반으로 판정될 테스트", fine_tuned_judge=FakeNormalJudge())
        self.assertEqual(result["realism_prediction"], 0)
        self.assertEqual(result["realism_probability"], 0.3)
        self.assertGreater(result["ranking_score"], 0.0)
        self.assertNotEqual(result["weakest_kev_axis"], "R")

    def test_weakest_target_uses_only_kev_axes_and_lowest_item(self) -> None:
        response = kev_response(3.0)
        for key in evaluator.AXES["B"]:
            response["answers"][key]["score"] = 1.0
        response["answers"]["B3"]["score"] = 0.25
        with patch.object(evaluator, "call_kev", return_value=response):
            result = evaluator.evaluate_text("대화", fine_tuned_judge=FakeNormalJudge())

        self.assertEqual(result["weakest_kev_axis"], "B")
        self.assertEqual(result["weakest_kev_subcriterion"], "B3")
        self.assertNotIn("weakest_axis", result)

    def test_result_records_exact_student_metadata(self) -> None:
        with patch.object(evaluator, "call_kev", return_value=kev_response(2.0)):
            rows = evaluator.evaluate_records(
                [{"scenario": "메타데이터 테스트", "metadata": {"e2e_index": 1}}],
                fine_tuned_judge=FakeJudge(),
            )

        realism = rows[0]["models"]["realism"]
        self.assertEqual(realism["checkpoint_sha256"], "test-checkpoint-sha256")
        self.assertEqual(realism["base_model"], "monologg/distilkobert")
        self.assertEqual(realism["base_revision"], "test-revision")

    def test_call_kev_uses_systemone_schema(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = kev_response()
        with patch.object(evaluator.requests, "post", return_value=response) as post:
            evaluator.call_kev("시나리오", "참고", "http://localhost:8009", "kev-latest")

        url = post.call_args.args[0]
        payload = post.call_args.kwargs["json"]
        self.assertEqual(url, "http://localhost:8009/v1/systemone")
        self.assertEqual(payload["model"], "kev-latest")
        self.assertEqual(payload["state"]["scenario"], "시나리오")
        self.assertEqual(payload["questions"]["A4"]["type"], "score")

    def test_csv_contains_ranking_and_binary_realism(self) -> None:
        with patch.object(evaluator, "call_kev", return_value=kev_response(3.0)):
            rows = evaluator.evaluate_records(
                [{"scenario": "테스트 시나리오", "metadata": {"e2e_index": 1}}],
                api_url="http://kev:8009",
                model="kev-latest",
                fine_tuned_judge=FakeJudge(),
            )

        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "result.csv"
            evaluator.write_results(rows, Path(directory) / "result.jsonl", csv_path)
            with csv_path.open(encoding="utf-8-sig", newline="") as stream:
                record = next(csv.DictReader(stream))

        self.assertEqual(record["realism_prediction"], "1")
        self.assertEqual(record["realism_probability"], "0.8")
        self.assertIn("ranking_score", record)
        self.assertIn("D", record)
        self.assertIn("weakest_kev_subcriterion", record)
        self.assertNotIn("rubric_label", record)


if __name__ == "__main__":
    unittest.main()
