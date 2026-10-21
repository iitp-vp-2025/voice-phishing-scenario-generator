import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

import Generate_scenario as generator


class GenerationPromptTest(unittest.TestCase):
    def test_selection_prompt_matches_ten_candidates(self) -> None:
        self.assertIn("10개의 기술(Tech) 키워드 후보", generator.SELECTION_PROMPT)
        self.assertIn("10개의 사회(Social) 키워드 후보", generator.SELECTION_PROMPT)
        self.assertIn("기술 1개, 사회 1개", generator.SELECTION_PROMPT)
        self.assertIn("양성 통화 분포 적합성", generator.SELECTION_PROMPT)
        self.assertIn("금융기관 또는 정부·공공기관 사칭", generator.SELECTION_PROMPT)
        self.assertIn("단순 기기 고객지원", generator.SELECTION_PROMPT)

    def test_uses_local_positive_reference_only_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fallback = Path(directory) / "positive_format_refs.jsonl"
            with patch.object(generator, "LOCAL_GENERATION_REFERENCE_PATH", fallback):
                self.assertEqual(generator.resolve_generation_reference_path(), "")
                fallback.write_text("{}\n", encoding="utf-8")
                self.assertEqual(
                    generator.resolve_generation_reference_path(),
                    str(fallback),
                )
                self.assertEqual(
                    generator.resolve_generation_reference_path("~/explicit.jsonl"),
                    str(Path("~/explicit.jsonl").expanduser()),
                )

    def test_includes_available_reference_pdfs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "tactic": root / "Tactic.pdf",
                "technique": root / "Technique.pdf",
                "scenario": root / "Scenario.pdf",
            }
            for path in paths.values():
                path.touch()

            texts = {
                paths["tactic"]: "전술 참고 내용",
                paths["technique"]: "기법 참고 내용",
                paths["scenario"]: "기존 시나리오 참고 내용",
            }
            with patch.dict(generator.PATHS, paths), patch.object(
                generator, "load_pdf_text", side_effect=lambda path: texts[path]
            ):
                prompt = generator.build_generation_prompt("기술", "사회", "선택 이유")

        self.assertIn("[전술 정보]\n전술 참고 내용", prompt)
        self.assertIn("[기법 정보]\n기법 참고 내용", prompt)
        self.assertIn("[기존 시나리오]\n기존 시나리오 참고 내용", prompt)
        self.assertIn("정확히 10턴", prompt)
        self.assertIn("speaker를 '상담원'과 '피해자'로 기계적으로 고정하지 않는다", prompt)
        self.assertIn("'피해자'라는 평가자용 명칭은 사용하지 않는다", prompt)
        self.assertIn("선택된 Tech와 Social 키워드", prompt)
        self.assertIn('"dialogue"', prompt)

    def test_omits_missing_references_and_scenario_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "tactic": root / "Tactic.pdf",
                "technique": root / "Technique.pdf",
                "scenario": root / "Scenario.pdf",
            }
            with patch.dict(generator.PATHS, paths), patch.object(generator, "load_pdf_text") as load_pdf:
                prompt = generator.build_generation_prompt("기술", "사회", "선택 이유")

        load_pdf.assert_not_called()
        self.assertNotIn("[전술 정보]", prompt)
        self.assertNotIn("[기법 정보]", prompt)
        self.assertNotIn("[기존 시나리오]", prompt)
        self.assertIn("정확히 10턴", prompt)

    def test_dialogue_parser_requires_exactly_ten_turns(self) -> None:
        dialogue = [
            {"turn": index, "speaker": "상담원" if index % 2 else "피해자", "text": f"발화 {index}"}
            for index in range(1, 11)
        ]
        parsed = generator.parse_dialogue_response(json.dumps({"title": "테스트", "dialogue": dialogue}))
        self.assertEqual(len(parsed["dialogue"]), 10)
        self.assertEqual(parsed["dialogue"][0]["turn"], 1)
        self.assertIn("상담원: 발화 1", generator.dialogue_text(parsed["dialogue"]))

        with self.assertRaisesRegex(generator.LLMError, "정확히 10턴"):
            generator.parse_dialogue_response(json.dumps({"title": "짧음", "dialogue": dialogue[:3]}))

    def test_dialogue_parser_rejects_chinese_and_japanese_scripts(self) -> None:
        dialogue = [
            {"turn": index, "speaker": "은행 담당자", "text": f"한국어 발화 {index}"}
            for index in range(1, 11)
        ]
        dialogue[3]["text"] = "확인这边입니다"
        with self.assertRaisesRegex(generator.LLMError, "중국어/일본어"):
            generator.parse_dialogue_response(
                json.dumps({"title": "한국어 제목", "dialogue": dialogue}, ensure_ascii=False)
            )

        dialogue[3]["text"] = "これで 완료입니다"
        with self.assertRaisesRegex(generator.LLMError, "중국어/일본어"):
            generator.parse_dialogue_response(
                json.dumps({"title": "한국어 제목", "dialogue": dialogue}, ensure_ascii=False)
            )

    def test_invalid_language_retries_with_the_same_prompt(self) -> None:
        invalid = {
            "title": "테스트",
            "dialogue": [
                {"turn": index, "speaker": "담당자", "text": "打扰了" if index == 1 else f"발화 {index}"}
                for index in range(1, 11)
            ],
        }
        valid = {
            "title": "테스트",
            "dialogue": [
                {"turn": index, "speaker": "담당자", "text": f"발화 {index}"}
                for index in range(1, 11)
            ],
        }
        with patch.object(
            generator,
            "query_llm",
            side_effect=[
                json.dumps(invalid, ensure_ascii=False),
                json.dumps(valid, ensure_ascii=False),
            ],
        ) as query:
            result = generator.generate_valid_dialogue("원본 프롬프트")

        self.assertEqual(result["title"], "테스트")
        self.assertEqual(query.call_count, 2)
        self.assertEqual(query.call_args_list[0].args[0], "원본 프롬프트")
        self.assertIn("원본 프롬프트", query.call_args_list[1].args[0])
        self.assertIn("중국어 한자", query.call_args_list[1].args[0])

    def test_format_reference_uses_only_positive_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "references.jsonl"
            path.write_text(
                "\n".join([
                    json.dumps({"text": "정상 통화", "label": 0}, ensure_ascii=False),
                    json.dumps({"text": "보이스피싱 통화 형식", "label": 1}, ensure_ascii=False),
                ]),
                encoding="utf-8",
            )
            examples = generator.load_format_references(str(path), count=3)
            prompt = generator.build_generation_prompt("기술", "사회", "이유", format_examples=examples)

        self.assertEqual(examples, ["보이스피싱 통화 형식"])
        self.assertIn("양성 통화 전사 참고자료", prompt)
        self.assertIn("최대한 가깝게 모방", prompt)
        self.assertIn("문장과 고유 사건 복사 금지", prompt)
        self.assertNotIn("정상 통화", prompt)


if __name__ == "__main__":
    unittest.main()
