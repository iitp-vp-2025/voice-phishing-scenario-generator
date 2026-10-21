import unittest
from unittest.mock import patch

import Improve_scenario as improver


def dialogue() -> list[dict]:
    return [
        {"turn": index, "speaker": "상담원" if index % 2 else "피해자", "text": f"원본 발화 {index}"}
        for index in range(1, 41)
    ]


def evaluation(axis_b: float, item_b3: float, ranking: float) -> dict:
    return {
        "weakest_kev_axis": "B",
        "weakest_kev_subcriterion": "B3",
        "weakest_kev_subcriterion_score": item_b3,
        "axis_normalized": {"A": 0.7, "B": axis_b, "C": 0.7, "D": 0.7},
        "expected_scores": {
            "A1": 2.8, "A2": 2.8, "A3": 2.8, "A4": 2.8,
            "B1": 2.0, "B2": 1.5, "B3": item_b3,
            "C1": 2.8, "C2": 2.8, "D1": 2.8, "D2": 2.8,
        },
        "ranking_score": ranking,
    }


class RefinementTest(unittest.TestCase):
    def setUp(self) -> None:
        self.original = {
            "scenario_id": "original",
            "metadata": {"title": "제목", "dialogue": dialogue()},
            "dialogue": dialogue(),
            "evaluation": evaluation(0.4, 0.8, 45.0),
        }

    def test_validation_allows_only_three_changed_turns(self) -> None:
        result = improver.validate_refinement(
            {
                "changes": [
                    {"turn": 4, "text": "수정 발화 4"},
                    {"turn": 8, "text": "수정 발화 8"},
                ],
                "change_summary": "두 턴 수정",
            },
            self.original,
        )
        self.assertEqual(result["changed_turns"], [4, 8])

        with self.assertRaisesRegex(ValueError, "최대 3개"):
            improver.validate_refinement(
                {
                    "changes": [
                        {"turn": 1, "text": "수정 1"},
                        {"turn": 2, "text": "수정 2"},
                        {"turn": 3, "text": "수정 3"},
                        {"turn": 4, "text": "수정 4"},
                    ],
                    "change_summary": "너무 많이 수정",
                },
                self.original,
            )

    def test_verdict_requires_target_improvement_without_large_regression(self) -> None:
        refined = {
            "evaluation": evaluation(0.5, 1.4, 47.0),
        }
        accepted, deltas = improver.refinement_verdict(self.original, refined)
        self.assertTrue(accepted)
        self.assertGreater(deltas["axis_delta"], 0)
        self.assertGreater(deltas["subcriterion_delta"], 0)

        refined["evaluation"]["axis_normalized"]["A"] = 0.5
        accepted, _ = improver.refinement_verdict(self.original, refined)
        self.assertFalse(accepted)

    def test_validation_rejects_foreign_script_in_refinement(self) -> None:
        with self.assertRaisesRegex(improver.generator.LLMError, "중국어/일본어"):
            improver.validate_refinement(
                {
                    "changes": [{"turn": 4, "text": "확인这边입니다"}],
                    "change_summary": "외국어가 섞인 수정",
                },
                self.original,
            )

    def test_refined_metadata_keeps_scenario_and_dialogue_in_sync(self) -> None:
        candidate = {
            "title": "제목",
            "dialogue": [
                {**item, "text": "수정 발화 4" if item["turn"] == 4 else item["text"]}
                for item in dialogue()
            ],
            "changed_turns": [4],
            "change_summary": "4번 턴 수정",
        }
        fake_result = [{"scenario_id": "refined", "metadata": {}, "evaluation": {}}]
        with patch.object(improver.evaluator, "evaluate_records", return_value=fake_result) as evaluate:
            improver.evaluate_refinement(
                self.original,
                candidate,
                fine_tuned_judge=object(),
                reference="",
                kev_api_url="http://kev",
                kev_model="kev-latest",
            )

        submitted = evaluate.call_args.args[0][0]
        self.assertEqual(submitted["scenario"], submitted["metadata"]["scenario"])
        self.assertEqual(submitted["scenario"], submitted["metadata"]["dialogue_text"])


if __name__ == "__main__":
    unittest.main()
