"""Evaluate every generated dialogue with DistilKoBERT + KEV.

R (voice-phishing realism) is the student's P(VP).  KEV scores the remaining
A-D rubric through its structured ``/v1/systemone`` API.  There is no pass
gate: every scenario receives a ranking score and remains in the output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from pdfminer.high_level import extract_text


load_dotenv()

SCRIPT_DIR = Path(__file__).parent.resolve()
GENERATE_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "generate_scenario"
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "evaluate_scenario"
DEFAULT_STUDENT_PATH = SCRIPT_DIR / "models" / "realism" / "best_student.pt"
PATHS = {
    "generated_jsonl": GENERATE_OUTPUT_DIR / "generated_scenarios.jsonl",
    "scenario_pdf": SCRIPT_DIR / "Scenario.pdf",
    "eval_jsonl": OUTPUT_DIR / "dialogue_evaluations.jsonl",
    "eval_csv": OUTPUT_DIR / "dialogue_evaluations.csv",
}

KEV_API_URL = os.environ.get("KEV_API_URL", "http://127.0.0.1:8009")
KEV_MODEL_NAME = os.environ.get("KEV_MODEL", "kev-latest")
KEV_API_KEY = os.environ.get("KEV_API_KEY", "")
SLLM_API_URL = os.environ.get("SLLM_API_URL") or os.environ.get("OLLAMA_API_URL", "")
SLLM_MODEL_NAME = os.environ.get("SLLM_MODEL", "")
DISTILKOBERT_MODEL_PATH = os.environ.get("DISTILKOBERT_MODEL_PATH", str(DEFAULT_STUDENT_PATH))
DISTILKOBERT_TOKENIZER_PATH = os.environ.get("DISTILKOBERT_TOKENIZER_PATH", "")
DISTILKOBERT_DEVICE = os.environ.get("DISTILKOBERT_DEVICE", "auto")
DISTILKOBERT_THRESHOLD = float(os.environ.get("DISTILKOBERT_THRESHOLD", "0.5"))

# Backward-compatible environment names used by the existing CLI and dashboard.
DISTILBERT_MODEL_PATH = os.environ.get("DISTILBERT_MODEL_PATH", DISTILKOBERT_MODEL_PATH)
DISTILBERT_TOKENIZER_PATH = os.environ.get("DISTILBERT_TOKENIZER_PATH", DISTILKOBERT_TOKENIZER_PATH)
DISTILBERT_DEVICE = os.environ.get("DISTILBERT_DEVICE", DISTILKOBERT_DEVICE)

# KEV rubric weights (total 90). R is produced separately by the student.
WEIGHTS = {
    "A1": 10,
    "A2": 10,
    "A3": 5,
    "A4": 5,
    "B1": 10,
    "B2": 10,
    "B3": 5,
    "C1": 10,
    "C2": 10,
    "D1": 10,
    "D2": 5,
}
AXES = {
    "A": ("A1", "A2", "A3", "A4"),
    "B": ("B1", "B2", "B3"),
    "C": ("C1", "C2"),
    "D": ("D1", "D2"),
}
AXIS_MAX = {axis: sum(WEIGHTS[key] for key in keys) for axis, keys in AXES.items()}
RANKING_WEIGHTS = {"R": 0.30, "A": 0.25, "B": 0.15, "C": 0.15, "D": 0.15}

KR = {
    "R": "Voice-Phishing Realism",
    "A": "Diversity",
    "A1": "Structural Distinctiveness",
    "A2": "Novel Use of Technology or Context",
    "A3": "Social-Engineering Mechanism Novelty",
    "A4": "Detector and Defensive Blind-Spot Specificity",
    "B": "Feasibility",
    "B1": "Resource, Access, and Technical Feasibility",
    "B2": "Victim Susceptibility Plausibility",
    "B3": "Institutional Factual Consistency",
    "C": "Completeness",
    "C1": "End-to-End Logical Consistency",
    "C2": "Scenario and Interaction-Flow Completeness",
    "D": "Generalizability and Adaptability",
    "D1": "Cross-Context Transferability",
    "D2": "Contextual Adaptability",
}


def _score_question(instructions: str, criteria: list[str]) -> dict[str, Any]:
    if len(criteria) != 5:
        raise ValueError("KEV score 질문은 정확히 5개 수준을 가져야 합니다.")
    evidence_rule = (
        "제공된 대화에 실제로 나타난 근거만 평가하고, 대화에 없는 배경·의도·절차를 "
        "임의로 가정하지 않는다. "
    )
    return {"type": "score", "instructions": evidence_rule + instructions, "criteria": criteria}


# The five entries map to KEV levels 0..4 and are displayed as ratings 1..5.
KEV_QUESTIONS: dict[str, dict[str, Any]] = {
    "A1": _score_question(
        "기존 대화 시나리오와 비교해 대화 전개, 설득 순서, 상호작용 구조가 실질적으로 다른지 평가한다. 기관명이나 키워드만 바꾼 경우는 낮게 평가한다.",
        [
            "기존 대화 구조를 거의 그대로 반복한다.",
            "표면 요소만 달라졌고 대화 단계와 설득 순서는 전형적이다.",
            "일부 대화 단계나 연결 방식이 달라 구별 가능하다.",
            "핵심 대화 전개와 설득 구조가 분명히 차별화된다.",
            "전체 대화 구조가 독창적이며 기존 시나리오와 뚜렷이 구별된다.",
        ],
    ),
    "A2": _score_question(
        "기술·제도·사회 맥락이 대화 속 신뢰 조건과 피해자 행동 요구를 얼마나 새롭게 바꾸는지 평가한다.",
        [
            "새 기술이나 맥락이 대화에 없거나 장식적으로만 언급된다.",
            "기술·맥락이 등장하지만 공격 방식에 거의 영향을 주지 않는다.",
            "기술·맥락이 일부 단계의 조건이나 행동을 바꾼다.",
            "기술·맥락이 신뢰 형성과 행동 요구에 구체적으로 기여한다.",
            "기술·맥락의 새로운 활용이 대화 구조의 핵심이다.",
        ],
    ),
    "A3": _score_question(
        "대화에서 피해자의 판단을 바꾸는 심리·사회적 메커니즘이 전형적인 권위, 긴급성, 손실회피의 단순 반복과 얼마나 구별되는지 평가한다.",
        [
            "설득 메커니즘이 없거나 성립하지 않는다.",
            "전형적인 권위·긴급성·손실회피만 단순 반복한다.",
            "전형적 방식에 구별 가능한 요소가 일부 결합된다.",
            "피해자 특성과 맥락에 맞춘 차별화된 설득 메커니즘이 있다.",
            "새로운 사회공학 메커니즘이 공격 성공 논리의 핵심이다.",
        ],
    ),
    "A4": _score_question(
        "확인 절차, 탐지 입력, 정책·책임·UI의 방어 공백을 이용하는 방식이 대화에서 구체적으로 드러나는지 평가한다. 다양성 축 안에서 평가한다.",
        [
            "대화에서 탐지 또는 방어 사각지대가 드러나지 않는다.",
            "사각지대를 막연히 언급하지만 실패 이유가 없다.",
            "특정 탐지 입력이나 절차의 한계를 설명한다.",
            "탐지 가정과 조직·절차상의 공백을 공격 흐름에 구체적으로 연결한다.",
            "여러 방어 계층의 구체적 사각지대가 대화의 차별성과 핵심 논리를 이룬다.",
        ],
    ),
    "B1": _score_question(
        "대화에서 주장하거나 사용하는 정보·자료·접근 권한·기술 단계가 현재 환경에서 현실적으로 가능한지 평가한다.",
        [
            "필수 자원이나 기술이 현실적으로 불가능하다.",
            "필수 정보·권한의 확보 경로가 없거나 기술 이름만 제시된다.",
            "대부분 가능하지만 일부 핵심 자원 또는 구현 조건이 불명확하다.",
            "필요 자원과 구현 조건이 구체적이고 현실적으로 확보 가능하다.",
            "모든 핵심 자원·접근·기술 조건이 현재 환경에서 명확하고 타당하다.",
        ],
    ),
    "B2": _score_question(
        "대화 속 피해자가 공식 확인 수단과 상황적 조건을 고려할 때 실제로 믿고 행동할 가능성이 얼마나 현실적인지 평가한다.",
        [
            "피해자가 속을 합리적 이유가 거의 없고 쉽게 확인 가능하다.",
            "권위나 긴급성만 있으며 확인하지 않을 이유가 부족하다.",
            "속을 만한 이유는 있으나 확인 수단이나 피해자 설정의 설명이 제한적이다.",
            "피해자 특성·상황·신뢰 근거가 구체적이어서 행동 가능성이 높다.",
            "확인 장벽과 신뢰 조건이 매우 설득력 있게 맞물려 행동 가능성이 매우 높다.",
        ],
    ),
    "B3": _score_question(
        "대화에서 사칭하거나 활용하는 기관·서비스의 역할, 권한, 정책, 공식 절차가 한국 현실과 일치하는지 평가한다.",
        [
            "한국의 실제 기관 권한이나 절차와 명백히 충돌한다.",
            "검증되지 않은 제도·앱·권한을 핵심 사실처럼 사용한다.",
            "대체로 가능하지만 일부 기관 역할이나 절차가 불확실하다.",
            "기관 역할과 공식 절차가 구체적이며 현실과 잘 부합한다.",
            "관련 기관·정책·인증 절차가 매우 정확하고 일관된다.",
        ],
    ),
    "C1": _score_question(
        "대화에 나타난 정보와 권한, 사용 기술, 법적·제도적 조건이 최초 접촉부터 최종 피해까지 모순 없이 이어지는지 평가한다.",
        [
            "핵심 단계가 서로 모순되거나 연결되지 않는다.",
            "확보하지 않은 정보·권한을 갑자기 사용하거나 큰 논리 공백이 있다.",
            "전체 흐름은 이해되지만 일부 단계의 원인과 결과가 불명확하다.",
            "각 단계의 전제와 결과가 구체적이고 일관되게 연결된다.",
            "정보·권한·기술·제도 조건이 전 과정에서 빈틈없이 일관된다.",
        ],
    ),
    "C2": _score_question(
        "최초 접촉, 신뢰 형성, 문제 제시, 피해자 반응, 행동 요구, 최종 결과까지 대화의 상호작용 흐름이 완결되어 있는지 평가한다.",
        [
            "핵심 단계가 다수 누락되어 대화가 완결되지 않는다.",
            "시작과 결과는 있으나 상호작용의 주요 연결 단계가 빠져 있다.",
            "필수 단계는 있으나 피해자 반응이나 채널 전환이 일부 불명확하다.",
            "상호작용의 시작부터 결과까지 자연스럽고 구체적으로 완결된다.",
            "모든 상호작용과 채널 전환이 명확하며 예외 상황까지 완결성 있게 구성된다.",
        ],
    ),
    "D1": _score_question(
        "대화에 나타난 핵심 설득 메커니즘을 유지하면서 기관, 서비스, 피해자 집단 또는 적용 상황을 바꿀 수 있는지 평가한다.",
        [
            "단일 사건이나 우연한 조건에 묶여 전이가 어렵다.",
            "명칭 교체 정도만 가능하고 다른 맥락에서는 논리가 약해진다.",
            "일부 유사 기관이나 피해자 집단으로 확장 가능하다.",
            "핵심 메커니즘을 여러 유사 맥락에 안정적으로 전이할 수 있다.",
            "다양한 기관·서비스·집단에 폭넓게 전이 가능한 일반적 메커니즘이다.",
        ],
    ),
    "D2": _score_question(
        "대화가 특정 명칭에만 의존하지 않고 다른 맥락의 실제 절차, 피해자 특성, 신뢰 조건에 맞게 조정될 수 있는지 평가한다.",
        [
            "맥락이 바뀌면 핵심 논리가 성립하지 않는다.",
            "기관명이나 피해자 이름을 바꾸는 수준에 그친다.",
            "일부 절차와 신뢰 조건을 조정할 수 있다.",
            "맥락별 절차·피해자 특성에 맞게 구체적으로 조정 가능하다.",
            "다양한 맥락에 깊이 적응하면서도 핵심 메커니즘과 일관성을 유지한다.",
        ],
    ),
}


def read_pdf(path: Path) -> str:
    return str(extract_text(str(path))).strip()


def short_text(text: str, limit: int) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "\n...[truncated]"


def load_optional_reference(path: Path, limit: int) -> str:
    if not path.is_file():
        print(f"[*] 참고 PDF가 없어 reference 없이 진행합니다: {path}")
        return ""
    print(f"[*] 평가 reference 사용: {path.name}")
    return short_text(read_pdf(path), limit)


def scenario_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def scenario_display(text: str, limit: int = 180) -> tuple[str, str]:
    title_match = re.search(r"^###\s*(.+)$", text, flags=re.MULTILINE)
    title = title_match.group(1).strip() if title_match else ""
    plain = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    plain = re.sub(r"\s+", " ", plain).strip()
    preview = plain if len(plain) <= limit else plain[:limit].rstrip() + "..."
    return title, preview


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            data = json.loads(line)
            scenario = data.get("scenario") or data.get("text") or data.get("raw_answer") or ""
            if scenario.strip():
                rows.append({"scenario": scenario.strip(), "metadata": {**data, "source_line": line_no}})
        return rows

    text = path.read_text(encoding="utf-8").strip()
    parts = [p.strip() for p in re.split(r"(?=^###\s*시나리오\s+\d+\s*:)", text, flags=re.MULTILINE)]
    records = []
    for part in parts:
        if re.search(r"^###\s*시나리오\s+\d+\s*:", part, flags=re.MULTILINE):
            title = re.search(r"^###\s*(.+)$", part, flags=re.MULTILINE)
            records.append({"scenario": part, "metadata": {"title": title.group(1).strip() if title else ""}})
    return records or ([{"scenario": text, "metadata": {}}] if text else [])


def load_evaluated_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        json.loads(line)["scenario_id"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def skip_evaluated(records: list[dict[str, Any]], output_jsonl: Path) -> list[dict[str, Any]]:
    done_ids = load_evaluated_ids(output_jsonl)
    pending = [record for record in records if scenario_id(record["scenario"]) not in done_ids]
    if done_ids:
        print(f"[*] 이미 평가된 시나리오: {len(records) - len(pending)}개 skip")
    return pending


def kev_endpoint(api_url: str) -> str:
    base = api_url.strip().rstrip("/")
    if not base:
        raise ValueError("KEV_API_URL이 설정되지 않았습니다.")
    return base if base.endswith("/v1/systemone") else f"{base}/v1/systemone"


def build_kev_payload(scenario: str, reference: str, model: str) -> dict[str, Any]:
    if not model.strip():
        raise ValueError("KEV_MODEL이 설정되지 않았습니다.")
    state: dict[str, str] = {"scenario": scenario.strip()}
    if reference.strip():
        state["reference_scenarios"] = reference.strip()
    return {"state": state, "model": model, "questions": KEV_QUESTIONS}


def call_kev(
    scenario: str,
    reference: str = "",
    api_url: str = KEV_API_URL,
    model: str = KEV_MODEL_NAME,
    timeout: int = 180,
) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {KEV_API_KEY}"} if KEV_API_KEY else None
    response = requests.post(
        kev_endpoint(api_url),
        json=build_kev_payload(scenario, reference, model),
        headers=headers,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict) or not isinstance(data.get("answers"), dict):
        raise ValueError("KEV 응답에 answers object가 없습니다.")
    return data


# Compatibility helpers used only by Improve_scenario.py's generative rewrite step.
def call_llm(
    prompt: str,
    api_url: str = SLLM_API_URL,
    model: str = SLLM_MODEL_NAME,
    timeout: int = 180,
    num_predict: int = 1024,
) -> str:
    if not api_url or not model:
        raise ValueError("생성형 호출에는 SLLM_API_URL과 SLLM_MODEL이 필요합니다.")
    response = requests.post(
        api_url,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {"num_ctx": 16384, "num_predict": num_predict},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return str(response.json().get("response", ""))


def parse_json(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("JSON 객체를 찾지 못했습니다.")
    value = json.loads(cleaned[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("JSON object가 필요합니다.")
    return value


def normalize_kev_answers(data: dict[str, Any]) -> dict[str, Any]:
    answers = data.get("answers")
    if not isinstance(answers, dict):
        raise ValueError("KEV 응답에 answers object가 없습니다.")

    expected_scores: dict[str, float] = {}
    ratings: dict[str, float] = {}
    subscores: dict[str, float] = {}
    confidences: dict[str, float | None] = {}
    probabilities: dict[str, Any] = {}
    legends: dict[str, Any] = {}
    for key, weight in WEIGHTS.items():
        answer = answers.get(key)
        if not isinstance(answer, dict) or "score" not in answer:
            raise ValueError(f"KEV 응답에 {key}.score가 없습니다.")
        score = float(answer["score"])
        if not math.isfinite(score) or not 0.0 <= score <= 4.0:
            raise ValueError(f"KEV {key}.score는 0~4 범위여야 합니다: {score}")
        expected_scores[key] = round(score, 6)
        ratings[key] = round(score + 1.0, 6)
        subscores[key] = round((score / 4.0) * weight, 6)
        confidence = answer.get("confidence")
        confidences[key] = round(float(confidence), 6) if confidence is not None else None
        probabilities[key] = answer.get("probabilities", [])
        legends[key] = answer.get("legend", {})

    axis_scores = {
        axis: round(sum(subscores[key] for key in keys), 6)
        for axis, keys in AXES.items()
    }
    axis_normalized = {
        axis: round(axis_scores[axis] / AXIS_MAX[axis], 6)
        for axis in AXES
    }
    weakest_axis = min(AXES, key=lambda axis: (axis_normalized[axis], -RANKING_WEIGHTS[axis], axis))
    weakest_subcriterion = min(
        AXES[weakest_axis],
        key=lambda key: (expected_scores[key], -WEIGHTS[key], key),
    )
    return {
        "expected_scores": expected_scores,
        "ratings": ratings,
        "subscores": subscores,
        "axis_scores": axis_scores,
        "axis_normalized": axis_normalized,
        "axis_rates": {axis: round(rate * 100.0, 2) for axis, rate in axis_normalized.items()},
        "kev_confidences": confidences,
        "kev_probabilities": probabilities,
        "kev_legends": legends,
        "weakest_kev_axis": weakest_axis,
        "weakest_kev_axis_rate": round(axis_normalized[weakest_axis] * 100.0, 2),
        "weakest_kev_subcriterion": weakest_subcriterion,
        "weakest_kev_subcriterion_score": expected_scores[weakest_subcriterion],
        "raw_kev_response": data,
    }


def calculate_ranking(realism_probability: float, axis_normalized: dict[str, float]) -> tuple[float, dict[str, float]]:
    normalized = {"R": float(realism_probability), **{axis: float(axis_normalized[axis]) for axis in AXES}}
    for axis, value in normalized.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"정규화 점수 {axis}는 0~1 범위여야 합니다: {value}")
    if any(value == 0.0 for value in normalized.values()):
        score = 0.0
    else:
        score = 100.0 * math.exp(
            sum(RANKING_WEIGHTS[axis] * math.log(value) for axis, value in normalized.items())
        )
    return round(score, 6), {axis: round(value, 6) for axis, value in normalized.items()}


def build_fine_tuned_judge(
    model_path: str = DISTILBERT_MODEL_PATH,
    tokenizer_path: str = DISTILBERT_TOKENIZER_PATH,
    device: str = DISTILBERT_DEVICE,
    max_length: int = 512,
    threshold: float = DISTILKOBERT_THRESHOLD,
) -> Any:
    from Fine_tuned_judge import DistilKoBERTJudge

    judge = DistilKoBERTJudge(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        device=device,
        max_length=max_length,
        threshold=threshold,
    )
    print(f"[*] DistilKoBERT checkpoint: {judge.model_path}")
    print(f"[*] DistilKoBERT SHA-256: {judge.checkpoint_sha256}")
    return judge


def fine_tuned_model_metadata(fine_tuned_judge: Any) -> dict[str, Any]:
    return {
        "backend": fine_tuned_judge.backend,
        "model": fine_tuned_judge.model_name,
        "checkpoint_sha256": fine_tuned_judge.checkpoint_sha256,
        "base_model": fine_tuned_judge.base_model,
        "base_revision": fine_tuned_judge.base_revision,
        "threshold": fine_tuned_judge.threshold,
        "max_length": getattr(fine_tuned_judge, "max_length", 512),
        "dialogue_aggregation": "top_2_mean",
    }


def attach_realism(
    result: dict[str, Any],
    scenario: str,
    fine_tuned_judge: Any,
    dialogue: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if dialogue and hasattr(fine_tuned_judge, "predict_dialogue"):
        judgment = fine_tuned_judge.predict_dialogue(dialogue)
    else:
        judgment = fine_tuned_judge.predict(scenario)
    probability = float(judgment["vp_probability"])
    ranking_score, normalized = calculate_ranking(probability, result["axis_normalized"])
    result.update({
        "realism_prediction": int(judgment["prediction"]),
        "realism_label": judgment["label"],
        "realism_probability": round(probability, 6),
        "realism_confidence": round(float(judgment["confidence"]), 6),
        "fine_tuned_judgment": judgment,
        "ranking_score": ranking_score,
        "normalized_scores": normalized,
    })
    result["summary"] = " · ".join(
        f"{axis} {normalized[axis] * 100:.1f}%" for axis in ("R", "A", "B", "C", "D")
    )
    return result


def evaluate_text(
    scenario: str,
    fine_tuned_judge: Any,
    dialogue: list[dict[str, Any]] | None = None,
    reference: str = "",
    api_url: str = KEV_API_URL,
    model: str = KEV_MODEL_NAME,
) -> dict[str, Any]:
    kev_response = call_kev(scenario, reference=reference, api_url=api_url, model=model)
    result = normalize_kev_answers(kev_response)
    return attach_realism(result, scenario, fine_tuned_judge, dialogue=dialogue)


def evaluate_records(
    records: list[dict[str, Any]],
    fine_tuned_judge: Any,
    source: str = "",
    reference: str = "",
    api_url: str = KEV_API_URL,
    model: str = KEV_MODEL_NAME,
) -> list[dict[str, Any]]:
    now = datetime.now(UTC).isoformat()
    rows = []
    for idx, record in enumerate(records, 1):
        scenario = record["scenario"]
        dialogue = record.get("dialogue") or record.get("metadata", {}).get("dialogue")
        print(f"\n--- [Evaluate] {idx}/{len(records)} scenario_id={scenario_id(scenario)} ---")
        result = evaluate_text(
            scenario,
            dialogue=dialogue,
            reference=reference,
            api_url=api_url,
            model=model,
            fine_tuned_judge=fine_tuned_judge,
        )
        rows.append({
            "scenario_id": scenario_id(scenario),
            "evaluated_at": now,
            "model": model,
            "models": {
                "rubric": {"backend": "kev_systemone", "model": model},
                "realism": fine_tuned_model_metadata(fine_tuned_judge),
            },
            "source_file": source,
            "metadata": record.get("metadata", {}),
            "dialogue": dialogue or [],
            "scenario": scenario,
            "criteria_labels": KR,
            "rubric_weights": WEIGHTS,
            "ranking_weights": RANKING_WEIGHTS,
            "evaluation": result,
        })
        print(f"Realism: {result['realism_prediction']} / P(VP)={result['realism_probability']}")
        print(f"Ranking score: {result['ranking_score']}")
    return rows


def write_results(rows: list[dict[str, Any]], jsonl_path: Path, csv_path: Path, append: bool = False) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "a" if append else "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    all_rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    fields = [
        "source_line",
        "e2e_index",
        "scenario_title",
        "scenario_preview",
        "scenario_id",
        "ranking_score",
        "realism_prediction",
        "realism_probability",
        "realism_confidence",
        "R",
        *AXES.keys(),
        *WEIGHTS.keys(),
        "weakest_kev_axis",
        "weakest_kev_subcriterion",
        "weakest_kev_axis_rate",
        "weakest_kev_subcriterion_score",
        "summary",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in all_rows:
            result = row["evaluation"]
            metadata = row.get("metadata", {})
            title, preview = scenario_display(row.get("scenario", ""))
            normalized = result.get("normalized_scores", {})
            writer.writerow({
                "source_line": metadata.get("source_line", ""),
                "e2e_index": metadata.get("e2e_index", ""),
                "scenario_title": title or metadata.get("title", ""),
                "scenario_preview": preview,
                "scenario_id": row["scenario_id"],
                "ranking_score": result.get("ranking_score", ""),
                "realism_prediction": result.get("realism_prediction", ""),
                "realism_probability": result.get("realism_probability", ""),
                "realism_confidence": result.get("realism_confidence", ""),
                "R": normalized.get("R", ""),
                **result["axis_scores"],
                **result["subscores"],
                "weakest_kev_axis": result.get("weakest_kev_axis", ""),
                "weakest_kev_subcriterion": result.get("weakest_kev_subcriterion", ""),
                "weakest_kev_axis_rate": result.get("weakest_kev_axis_rate", ""),
                "weakest_kev_subcriterion_score": result.get("weakest_kev_subcriterion_score", ""),
                "summary": result.get("summary", ""),
            })


def evaluate_file(
    input_path: Path = PATHS["generated_jsonl"],
    output_jsonl: Path = PATHS["eval_jsonl"],
    output_csv: Path = PATHS["eval_csv"],
    append: bool = False,
    reference_chars: int = 12000,
    api_url: str = KEV_API_URL,
    model: str = KEV_MODEL_NAME,
    reevaluate: bool = False,
    distilbert_model: str = DISTILBERT_MODEL_PATH,
    distilbert_tokenizer: str = DISTILBERT_TOKENIZER_PATH,
    distilbert_device: str = DISTILBERT_DEVICE,
    distilbert_max_length: int = 512,
) -> list[dict[str, Any]]:
    records = load_records(input_path)
    if not reevaluate:
        records = skip_evaluated(records, output_jsonl)
    if not records:
        print("[*] 새로 평가할 시나리오가 없습니다.")
        return []
    reference = load_optional_reference(PATHS["scenario_pdf"], reference_chars)
    fine_tuned_judge = build_fine_tuned_judge(
        model_path=distilbert_model,
        tokenizer_path=distilbert_tokenizer,
        device=distilbert_device,
        max_length=distilbert_max_length,
    )
    rows = evaluate_records(
        records,
        fine_tuned_judge=fine_tuned_judge,
        source=str(input_path),
        reference=reference,
        api_url=api_url,
        model=model,
    )
    write_results(rows, output_jsonl, output_csv, append=append or not reevaluate)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DistilKoBERT 현실성 + KEV rubric 평가")
    parser.add_argument("-i", "--input", type=Path, default=PATHS["generated_jsonl"])
    parser.add_argument("--kev-api-url", "--api-url", dest="kev_api_url", default=KEV_API_URL)
    parser.add_argument("--kev-model", "--model", dest="kev_model", default=KEV_MODEL_NAME)
    parser.add_argument("--distilbert-model", default=DISTILBERT_MODEL_PATH)
    parser.add_argument("--distilbert-tokenizer", default=DISTILBERT_TOKENIZER_PATH)
    parser.add_argument("--distilbert-device", default=DISTILBERT_DEVICE)
    parser.add_argument("--distilbert-max-length", type=int, default=512)
    parser.add_argument("--output-jsonl", type=Path, default=PATHS["eval_jsonl"])
    parser.add_argument("--output-csv", type=Path, default=PATHS["eval_csv"])
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--reevaluate", action="store_true")
    parser.add_argument("--reference-chars", type=int, default=12000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = evaluate_file(
        input_path=args.input,
        output_jsonl=args.output_jsonl,
        output_csv=args.output_csv,
        append=args.append,
        reference_chars=args.reference_chars,
        api_url=args.kev_api_url,
        model=args.kev_model,
        reevaluate=args.reevaluate,
        distilbert_model=args.distilbert_model,
        distilbert_tokenizer=args.distilbert_tokenizer,
        distilbert_device=args.distilbert_device,
        distilbert_max_length=args.distilbert_max_length,
    )
    print("\n[*] 평가 저장 완료")
    print(f"- JSONL: {args.output_jsonl}")
    print(f"- CSV:   {args.output_csv}")
    print(f"[*] 평가 건수: {len(rows)}")


if __name__ == "__main__":
    main()
