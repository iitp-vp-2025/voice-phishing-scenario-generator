"""Minimal KEV-guided dialogue refinement used by the end-to-end pipeline."""

from __future__ import annotations

import csv
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import Evaluate_scenario as evaluator
import Generate_scenario as generator


MAX_REFINEMENT_ATTEMPTS = 2
MAX_CHANGED_TURNS = 3
MAX_NON_TARGET_DROP = 0.03


REFINE_PROMPT = """당신은 방어 연구용 한국어 보이스피싱 대화 데이터셋 편집자입니다.
아래 원본 대화의 전체 맥락을 유지하면서 지정된 KEV 세부항목 하나만 최소한으로 보완하세요.

[수정 대상]
- axis: {target_axis}
- subcriterion: {target_subcriterion} ({label})
- current expected score: {current_score}/4
- 평가 질문: {instructions}
- 한 단계 높은 목표 기준: {target_criterion}

[고정 조건]
- 제목, speaker, 모든 발화는 한국어로 유지하세요. 영문 기술명과 고유명사는 허용하지만 중국어 한자와 일본어 가나는 절대 사용하지 마세요.
- 제목, 등장 화자, 사건의 중심 맥락, 피해자, 기관·서비스, 공격 목표, 핵심 설득 구조와 결말을 바꾸지 마세요.
- 지정된 세부항목과 직접 관련된 발화만 최대 3개 선택해서 text만 교체하세요.
- 다른 축을 개선하려고 전체 대화를 다시 쓰지 마세요.
- rubric 명칭이나 평가 설명을 대화에 삽입하지 마세요.
- 부족한 요소를 반복하거나 노골적으로 강조하지 말고 기존 대화에 자연스럽게 녹이세요.
- 실제 개인정보, 계좌번호, URL, 전화번호, 인증번호를 추가하지 마세요.

[생성 이력 - 맥락 고정용이며 평가 설명으로 출력하지 않음]
- tech_keywords: {tech_keywords}
- social_keyword: {social_keyword}
- selected_reason: {selected_reason}

[원본]
{original_json}

{retry_feedback}

다음 JSON 객체 하나만 출력하세요. 코드블록과 설명문은 금지합니다.
{{
  "change_summary": "무엇을 최소 수정했는지 한 문장",
  "changes": [
    {{"turn": 18, "text": "해당 turn의 교체 문장"}}
  ]
}}
"""


def _original_dialogue(row: dict[str, Any]) -> list[dict[str, Any]]:
    dialogue = row.get("dialogue") or row.get("metadata", {}).get("dialogue")
    if not isinstance(dialogue, list) or not dialogue:
        raise ValueError("refinement할 구조화 dialogue가 없습니다.")
    return dialogue


def _original_title(row: dict[str, Any]) -> str:
    return str(row.get("metadata", {}).get("title", "")).strip()


def improvement_prompt(row: dict[str, Any], retry_feedback: str = "") -> str:
    result = row["evaluation"]
    axis = result["weakest_kev_axis"]
    item = result["weakest_kev_subcriterion"]
    score = float(result["weakest_kev_subcriterion_score"])
    target_level = min(4, math.floor(score) + 1)
    question = evaluator.KEV_QUESTIONS[item]
    metadata = row.get("metadata", {})
    original = {
        "title": _original_title(row),
        "dialogue": _original_dialogue(row),
    }
    return REFINE_PROMPT.format(
        target_axis=axis,
        target_subcriterion=item,
        label=evaluator.KR[item],
        current_score=round(score, 4),
        instructions=question["instructions"],
        target_criterion=question["criteria"][target_level],
        tech_keywords=", ".join(metadata.get("tech_keywords", [])),
        social_keyword=metadata.get("social_keyword", ""),
        selected_reason=metadata.get("selected_reason", ""),
        original_json=json.dumps(original, ensure_ascii=False),
        retry_feedback=retry_feedback,
    )


def validate_refinement(data: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    original_title = _original_title(row)
    changes = data.get("changes")
    summary = str(data.get("change_summary", "")).strip()
    if not isinstance(changes, list) or not changes:
        raise ValueError("refinement가 실제 대화를 수정하지 않았습니다.")
    if len(changes) > MAX_CHANGED_TURNS:
        raise ValueError(f"refinement는 최대 {MAX_CHANGED_TURNS}개 turn만 수정할 수 있습니다.")

    original = _original_dialogue(row)
    replacements = {}
    for item in changes:
        if not isinstance(item, dict):
            raise ValueError("refinement changes의 각 항목은 object여야 합니다.")
        turn = int(item.get("turn", 0))
        text = str(item.get("text", "")).strip()
        if not 1 <= turn <= len(original) or not text:
            raise ValueError(f"잘못된 refinement turn입니다: {turn}")
        if turn in replacements:
            raise ValueError(f"refinement turn이 중복되었습니다: {turn}")
        if text == str(original[turn - 1].get("text", "")).strip():
            raise ValueError(f"refinement turn {turn}의 text가 원본과 같습니다.")
        generator.validate_korean_output(text, f"refinement turn {turn}")
        replacements[turn] = text

    normalized = [
        {
            "turn": index,
            "speaker": str(item.get("speaker", "")).strip(),
            "text": replacements.get(index, str(item.get("text", "")).strip()),
        }
        for index, item in enumerate(original, 1)
    ]
    generator.validate_korean_output(original_title, "refinement title")
    for item in normalized:
        generator.validate_korean_output(item["speaker"], f"refinement turn {item['turn']} speaker")
        generator.validate_korean_output(item["text"], f"refinement turn {item['turn']} text")
    return {
        "title": original_title,
        "dialogue": normalized,
        "changed_turns": sorted(replacements),
        "change_summary": summary,
    }


def generate_refinement(row: dict[str, Any], api_url: str, model: str, retry_feedback: str = "") -> dict[str, Any]:
    raw = evaluator.call_llm(
        improvement_prompt(row, retry_feedback=retry_feedback),
        api_url=api_url,
        model=model,
        timeout=300,
        num_predict=1200,
    )
    return validate_refinement(evaluator.parse_json(raw), row)


def evaluate_refinement(
    original: dict[str, Any],
    candidate: dict[str, Any],
    fine_tuned_judge: Any,
    reference: str,
    kev_api_url: str,
    kev_model: str,
) -> dict[str, Any]:
    metadata = {
        **original.get("metadata", {}),
        "title": candidate["title"],
        "dialogue": candidate["dialogue"],
        "dialogue_text": generator.dialogue_text(candidate["dialogue"]),
        "original_scenario_id": original["scenario_id"],
        "changed_turns": candidate["changed_turns"],
        "change_summary": candidate["change_summary"],
    }
    metadata["scenario"] = metadata["dialogue_text"]
    return evaluator.evaluate_records(
        [{
            "scenario": metadata["dialogue_text"],
            "dialogue": candidate["dialogue"],
            "metadata": metadata,
        }],
        fine_tuned_judge=fine_tuned_judge,
        source=original.get("source_file", ""),
        reference=reference,
        api_url=kev_api_url,
        model=kev_model,
    )[0]


def refinement_verdict(original: dict[str, Any], refined: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    before = original["evaluation"]
    after = refined["evaluation"]
    axis = before["weakest_kev_axis"]
    item = before["weakest_kev_subcriterion"]
    axis_delta = after["axis_normalized"][axis] - before["axis_normalized"][axis]
    item_delta = after["expected_scores"][item] - before["expected_scores"][item]
    other_deltas = {
        key: after["axis_normalized"][key] - before["axis_normalized"][key]
        for key in evaluator.AXES
        if key != axis
    }
    ranking_delta = after["ranking_score"] - before["ranking_score"]
    accepted = (
        axis_delta > 0
        and item_delta > 0
        and all(delta >= -MAX_NON_TARGET_DROP for delta in other_deltas.values())
        and ranking_delta >= 0
    )
    return accepted, {
        "target_axis": axis,
        "target_subcriterion": item,
        "axis_delta": round(axis_delta, 6),
        "subcriterion_delta": round(item_delta, 6),
        "non_target_axis_deltas": {key: round(value, 6) for key, value in other_deltas.items()},
        "ranking_delta": round(ranking_delta, 6),
    }


def score_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    result = row["evaluation"]
    return {
        "ranking_score": result["ranking_score"],
        "normalized_scores": result["normalized_scores"],
        "axis_normalized": result["axis_normalized"],
        "expected_scores": result["expected_scores"],
        "weakest_kev_axis": result["weakest_kev_axis"],
        "weakest_kev_subcriterion": result["weakest_kev_subcriterion"],
    }


def refine_rows(
    rows: list[dict[str, Any]],
    fine_tuned_judge: Any,
    reference: str,
    kev_api_url: str,
    kev_model: str,
    generation_api_url: str,
    generation_model: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    final_rows = []
    traces = []
    for index, original in enumerate(rows, 1):
        print(
            f"\n--- [Refine] {index}/{len(rows)} scenario_id={original['scenario_id']} "
            f"target={original['evaluation']['weakest_kev_subcriterion']} ---"
        )
        accepted_row = None
        retry_feedback = ""
        for attempt in range(1, MAX_REFINEMENT_ATTEMPTS + 1):
            trace: dict[str, Any] = {
                "original_scenario_id": original["scenario_id"],
                "attempt": attempt,
                "target_axis": original["evaluation"]["weakest_kev_axis"],
                "target_subcriterion": original["evaluation"]["weakest_kev_subcriterion"],
                "created_at": datetime.now(UTC).isoformat(),
            }
            try:
                candidate = generate_refinement(
                    original,
                    api_url=generation_api_url,
                    model=generation_model,
                    retry_feedback=retry_feedback,
                )
                refined = evaluate_refinement(
                    original,
                    candidate,
                    fine_tuned_judge=fine_tuned_judge,
                    reference=reference,
                    kev_api_url=kev_api_url,
                    kev_model=kev_model,
                )
                accepted, deltas = refinement_verdict(original, refined)
                trace.update({
                    "status": "accepted" if accepted else "rejected",
                    "changed_turns": candidate["changed_turns"],
                    "change_summary": candidate["change_summary"],
                    "refined_scenario_id": refined["scenario_id"],
                    "deltas": deltas,
                    "original_dialogue": _original_dialogue(original),
                    "refined_dialogue": candidate["dialogue"],
                    "original_scores": score_snapshot(original),
                    "refined_scores": score_snapshot(refined),
                })
                traces.append(trace)
                if accepted:
                    refined["metadata"]["refinement_status"] = "accepted"
                    refined["metadata"]["refinement_target_axis"] = deltas["target_axis"]
                    refined["metadata"]["refinement_target_subcriterion"] = deltas["target_subcriterion"]
                    refined["metadata"]["refinement_deltas"] = deltas
                    accepted_row = refined
                    break
                retry_feedback = (
                    "[재시도 피드백]\n이전 수정은 채택 기준을 만족하지 못했습니다. "
                    f"목표 축 변화={deltas['axis_delta']}, 세부항목 변화={deltas['subcriterion_delta']}, "
                    f"랭킹 변화={deltas['ranking_delta']}였습니다. 같은 세부항목에 대해 더 작은 다른 수정을 하세요."
                )
            except Exception as exc:
                trace.update({"status": "error", "error": str(exc)})
                traces.append(trace)
                retry_feedback = f"[재시도 피드백]\n이전 출력 오류: {exc}. 출력 형식과 수정 범위를 지키세요."

        if accepted_row is None:
            initial_result = original["evaluation"]
            original = {
                **original,
                "metadata": {
                    **original.get("metadata", {}),
                    "refinement_status": "failed",
                    "refinement_target_axis": initial_result["weakest_kev_axis"],
                    "refinement_target_subcriterion": initial_result["weakest_kev_subcriterion"],
                },
            }
            final_rows.append(original)
            print("[WARN] refinement 채택 실패: 원본을 최종 데이터로 유지합니다.")
        else:
            final_rows.append(accepted_row)
            print(f"[*] refinement accepted: {accepted_row['scenario_id']}")
    return final_rows, traces


def write_trace(rows: list[dict[str, Any]], jsonl_path: Path, csv_path: Path) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    fields = [
        "original_scenario_id",
        "refined_scenario_id",
        "attempt",
        "target_axis",
        "target_subcriterion",
        "status",
        "changed_turns",
        "change_summary",
        "deltas",
        "error",
        "created_at",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **{key: row.get(key, "") for key in fields},
                "changed_turns": json.dumps(row.get("changed_turns", []), ensure_ascii=False),
                "deltas": json.dumps(row.get("deltas", {}), ensure_ascii=False),
            })


if __name__ == "__main__":
    raise SystemExit("Improve_scenario.py는 Run_end_to_end.py의 필수 refinement 단계에서 실행됩니다.")
