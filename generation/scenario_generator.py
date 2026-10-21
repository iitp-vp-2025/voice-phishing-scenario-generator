import re
import json
import os
import random
import csv
import requests
from pathlib import Path
from datetime import datetime, UTC

from dotenv import load_dotenv

load_dotenv()

from pdfminer.high_level import extract_text

# CONFIGURATION
PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLLECT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "collect_keyword"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "generate_scenario"

PATHS = {
    "tech_kw": COLLECT_OUTPUT_DIR / "tech_keywords_curated.txt",
    "social_kw": COLLECT_OUTPUT_DIR / "social_keywords_curated.txt",
    "tactic": PROJECT_ROOT / "Tactic.pdf",
    "technique": PROJECT_ROOT / "Technique.pdf",
    "scenario": PROJECT_ROOT / "Scenario.pdf",
    "generated_md": OUTPUT_DIR / "generated_scenarios.md",
    "generated_jsonl": OUTPUT_DIR / "generated_scenarios.jsonl",
}

API_URL = os.environ.get("SLLM_API_URL") or os.environ.get("OLLAMA_API_URL", "")
MODEL_NAME = os.environ.get("SLLM_MODEL", "")
DIALOGUE_TURNS = 10
MAX_DIALOGUE_GENERATION_ATTEMPTS = 2
REFERENCE_CHAR_LIMIT = 2500
FORMAT_REFERENCE_CHAR_LIMIT = 1200
FORBIDDEN_FOREIGN_SCRIPT = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]")
LOCAL_GENERATION_REFERENCE_PATH = PROJECT_ROOT / "tmp" / "positive_format_refs.jsonl"


def resolve_generation_reference_path(path_text: str = "") -> str:
    """명시 경로를 우선하고, 없으면 Git에 올리지 않는 로컬 양성 예시를 사용합니다."""
    explicit_path = path_text.strip()
    if explicit_path:
        return str(Path(explicit_path).expanduser())
    if LOCAL_GENERATION_REFERENCE_PATH.is_file():
        return str(LOCAL_GENERATION_REFERENCE_PATH)
    return ""


GENERATION_REFERENCE_PATH = resolve_generation_reference_path(
    os.environ.get("VP_GENERATION_REFERENCE_PATH", "")
)
GENERATION_REFERENCE_COUNT = int(os.environ.get("VP_GENERATION_REFERENCE_COUNT", "3"))


# PROMPTS
SELECTION_PROMPT = """당신은 세계 최고 수준의 보안 위협 분석가입니다.
아래에 10개의 기술(Tech) 키워드 후보와 10개의 사회(Social) 키워드 후보가 있습니다.

[기술(Tech) 키워드 후보]
{tech_candidates_str}

[사회(Social) 키워드 후보]
{social_candidates_str}

**[선택 기준 - 매우 엄격함]**
이 후보들 중에서, 일반인이 실생활에서 마주쳤을 때 의심 없이 속아 넘어갈 수 있는 **가장 지극히 현실적이고 자연스러운 수법(기술 1개, 사회 1개)**만을 선택하세요.

1. **거절(Reject) 기준**:
   - 억지스럽거나 인위적인 조합은 무조건 피하세요.
   - 직관적으로 말이 되지 않는 논리 도약이 있는 쌍은 버리세요.
   - 합이 맞지 않는 빈약한 조합을 그럴싸한 말장난으로 억지 포장(Speculative reasoning)하지 마세요.
   - 일반 사용자가 들었을 때 조금이라도 생소하거나 어색한 조합은 절대 금지합니다.
   - 특정 국가에만 존재하는 제도나 기관을 한국의 제도처럼 바꿔 붙여야 성립하는 조합은 버리세요.

2. **우대(Prefer) 기준**:
   - 참신함(Novelty)보다 **현실성(Realism)**이 압도적으로 더 중요합니다.
   - 현실에 실존하는 워크플로우(본인 인증, 환불, 예약, 결제 오류, 계정 접근 문제, 기관 사칭 등)에 완벽하게 녹아드는 자연스러운 미끼(Bait)를 고르세요.

3. **양성 통화 분포 적합성 기준**:
   - 새 키워드를 그대로 사용하면서도, 선택 조합이 한국의 금융기관 또는 정부·공공기관 사칭, 금융상품·대출·환급·지원 제도·행정 확인 중 하나와 자연스럽게 이어져야 합니다.
   - 단순 기기 고객지원, 정상 보안 점검, 소프트웨어 설정 변경만으로 끝나는 조합은 제외하세요.
   - 기관 사칭, 민감정보 확인, 금융 행동 요구, 별도 채널 전환, 공식 확인 방해 중 적어도 하나가 무리 없이 성립하는 조합을 고르세요.
   - 키워드 이름만 금융·행정과 관련된 것처럼 보이는지를 판단하지 말고, 실제 한국어 통화의 처음부터 행동 요구까지 자연스럽게 전개할 수 있는지를 판단하세요.

반드시 위 기준에 부합하는 가장 현실적인 최고의 조합 1쌍을 찾은 뒤, 아래 양식으로만 답변하세요.

[선택된 조합]
Reason: [왜 이 조합이 일반인에게 의심 없이 통할 수 있는 흔하고 현실적인 사기 흐름인지 1~2문장으로 객관적 설명]
Tech: [선택한 기술 키워드]
Social: [선택한 사회 키워드]
"""

GENERATION_PROMPT = """당신은 방어 연구용 한국어 보이스피싱 대화 데이터셋을 작성하는 분석가입니다.
아래 입력을 바탕으로 하나의 완결된 음성 통화 대화만 작성하세요.

{reference_sections}

{format_reference_section}

[선별된 기술(Tech) 키워드]
{tech_str}

[선별된 사회(Social) 키워드]
{sampled_social}

[각 키워드가 선택된 이유 (Why this combination was selected)]
{selected_reason}

위 정보와 키워드를 자연스럽게 결합하되, 평가용 설명문이나 별도 배경 요약을 만들지 말고 실제 대화에서 필요한 맥락이 드러나게 하세요.

[중요 조건]
1. 제목, speaker, 모든 발화는 한국어로 작성한다. 선택된 영문 기술명과 고유명사는 그대로 쓸 수 있지만, 중국어 한자나 일본어 가나는 절대 사용하지 말고 자연스러운 한국어로 바꾼다.
2. dialogue는 정확히 10턴으로 작성한다.
3. 한 화자의 한 번 발화를 한 turn으로 계산하고 turn은 1부터 빠짐없이 증가시킨다.
4. 각 발화는 실제 통화 전사처럼 짧은 맞장구, 되묻기, 말 끊김, 반복, 자기수정을 자연스럽게 포함할 수 있다. 다만 턴 수를 채우기 위한 의미 없는 반복은 금지한다.
5. 최초 연락, 신뢰 형성, 문제 제시, 피해자의 질문·의심·확인, 행동 요구, 최종 결과가 자연스럽게 이어져야 한다.
6. 기술 키워드와 사회 키워드는 설명문이 아니라 대화 흐름에 자연스럽게 나타나야 한다.
7. 현재 확인할 수 없는 정부 정책, 기관 권한, 공식 절차, 신규 제도를 사실처럼 만들지 않는다.
8. 국가·지역·기관·제도의 관계를 일관되게 유지하고, 외국의 고유 제도를 한국 제도처럼 표현하지 않는다.
9. 실제 개인정보, 계좌번호, URL, 전화번호, 인증번호는 만들지 않는다.
10. 기존 시나리오의 명칭만 바꾼 반복을 피하고 대화의 설득 구조와 상호작용을 차별화한다.
11. 하나의 시나리오만 출력하고 대안, 해설, 평가, 마크다운을 출력하지 않는다.
12. 양성 통화 전사 참고자료가 있으면 말투, 발화 밀도, 반복과 정정, 질문과 대응의 리듬, 신뢰 형성부터 행동 요구까지의 전개 순서를 최대한 가깝게 모방한다. 단, 기관명, 인물, 금액, 고유 사건과 문장을 그대로 복사하지 않는다.
13. speaker를 '상담원'과 '피해자'로 기계적으로 고정하지 않는다. 특히 '피해자'라는 평가자용 명칭은 사용하지 않는다. 발신자가 통화에서 주장하는 실제 역할과 상대 역할(예: 고객, 신청자)을 짧고 구체적으로 적고, 같은 화자의 연속 발화도 대화상 자연스러우면 허용한다.
14. 선택된 Tech와 Social 키워드가 제목에만 머물지 않고 통화의 사건, 신뢰 형성 또는 행동 요구에 각각 실질적으로 작용해야 한다.

[출력 양식]
다음 JSON 객체 하나만 출력하세요. JSON 밖의 텍스트와 코드블록은 금지합니다.
{{
  "title": "구체적이고 현실적인 대화 제목",
  "dialogue": [
    {{"turn": 1, "speaker": "발신자가 주장하는 역할", "text": "발화"}},
    {{"turn": 2, "speaker": "상대방의 실제 역할", "text": "발화"}}
  ]
}}
"""


# UTILITIES
class LLMError(RuntimeError):
    pass


def validate_korean_output(text: str, field: str) -> None:
    """한국어 데이터에 섞인 중국어 한자와 일본어 가나를 차단합니다."""
    invalid = list(dict.fromkeys(FORBIDDEN_FOREIGN_SCRIPT.findall(text)))
    if invalid:
        raise LLMError(f"{field}에 중국어/일본어 문자가 포함되었습니다: {''.join(invalid)}")


def query_llm(prompt: str, num_predict: int = 1024) -> str:
    """LLM으로 프롬프트를 전송하고 텍스트 응답을 반환합니다."""
    if not API_URL or not MODEL_NAME:
        raise LLMError("SLLM_API_URL과 SLLM_MODEL을 설정하세요.")
    try:
        res = requests.post(
            API_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "options": {"num_ctx": 16384, "num_predict": num_predict},
            },
            timeout=300,
        )
        res.raise_for_status()
        return res.json().get('response', '')
    except Exception as e:
        body = getattr(getattr(e, "response", None), "text", "")
        raise LLMError(f"LLM 통신 실패: {e} {body[:500]}") from e

def load_pdf_text(path: Path) -> str:
    return str(extract_text(str(path))).strip()

def load_reference_section(title: str, path: Path) -> str:
    if not path.is_file():
        return ""
    text = load_pdf_text(path)[:REFERENCE_CHAR_LIMIT].strip()
    return f"[{title}]\n{text}" if text else ""


def load_format_references(path_text: str, count: int = GENERATION_REFERENCE_COUNT) -> list[str]:
    if not path_text or count <= 0:
        return []
    path = Path(path_text).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"통화 형식 reference를 찾을 수 없습니다: {path}")

    rows = []
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    elif path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        raise ValueError("통화 형식 reference는 CSV 또는 JSONL이어야 합니다.")

    texts, seen = [], set()
    for row in rows:
        if str(row.get("label", "1")).strip() not in {"1", "1.0", "voice_phishing"}:
            continue
        text = str(row.get("text", "")).strip()
        if text and text not in seen:
            seen.add(text)
            texts.append(text[:FORMAT_REFERENCE_CHAR_LIMIT])
    return random.sample(texts, min(count, len(texts)))


def format_reference_section(examples: list[str]) -> str:
    if not examples:
        return ""
    rendered = "\n\n".join(f"[형식 예시 {index}]\n{text}" for index, text in enumerate(examples, 1))
    return f"""[양성 통화 전사 참고자료 - 문장과 고유 사건 복사 금지]
아래 자료는 실제 양성 데이터의 문체와 상호작용 구조를 보여주는 핵심 예시입니다.
출력 대화가 아래 예시와 같은 데이터 분포로 읽히도록 말투, 발화 길이와 밀도, 짧은 응답과 반복,
말 끊김과 자기수정, 질문과 대응의 리듬, 정상 상담처럼 시작해 요구가 구체화되는 전개 순서를
최대한 가깝게 모방하세요. 지나치게 매끈한 설명문이나 정돈된 안내문처럼 쓰지 마세요.

예시의 기관명, 인물, 금액, 고유 사건과 문장은 그대로 가져오지 말고, 현재 선택된 키워드와
새로운 사건에 맞게 바꾸세요. 표면 문장을 복사하지 않는 범위에서 전사 문체와 대화 구조는
적극적으로 닮게 작성하세요.

{rendered}"""


def build_generation_prompt(
    tech_str: str,
    sampled_social: str,
    selected_reason: str,
    format_examples: list[str] | None = None,
) -> str:
    tactic_section = load_reference_section("전술 정보", PATHS["tactic"])
    technique_section = load_reference_section("기법 정보", PATHS["technique"])
    scenario_section = load_reference_section("기존 시나리오", PATHS["scenario"])
    reference_sections = "\n\n".join(
        section for section in (tactic_section, technique_section, scenario_section) if section
    )

    return GENERATION_PROMPT.format(
        reference_sections=reference_sections,
        format_reference_section=format_reference_section(format_examples or []),
        tech_str=tech_str,
        sampled_social=sampled_social,
        selected_reason=selected_reason,
    )

def load_keywords(path: Path) -> list[str]:
    keywords, seen = [], set()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([^|]+)\s*\|\s*(.+)$", line.strip())
        if match:
            kw = match.group(2).strip()
            if kw.lower() not in seen:
                seen.add(kw.lower())
                keywords.append(kw)
    return keywords

def parse_dialogue_response(text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise LLMError("대화 생성 응답에서 JSON 객체를 찾지 못했습니다.")
    data = json.loads(cleaned[start:end + 1])
    title = str(data.get("title", "")).strip()
    dialogue = data.get("dialogue")
    if not title or not isinstance(dialogue, list):
        raise LLMError("대화 생성 응답에는 title과 dialogue가 필요합니다.")
    validate_korean_output(title, "title")
    if len(dialogue) != DIALOGUE_TURNS:
        raise LLMError(f"대화는 정확히 {DIALOGUE_TURNS}턴이어야 합니다: {len(dialogue)}턴")

    normalized = []
    for index, item in enumerate(dialogue, 1):
        if not isinstance(item, dict):
            raise LLMError(f"dialogue[{index}]는 object여야 합니다.")
        speaker = str(item.get("speaker", "")).strip()
        utterance = str(item.get("text", "")).strip()
        if not speaker or not utterance:
            raise LLMError(f"dialogue[{index}]의 speaker와 text는 비어 있을 수 없습니다.")
        validate_korean_output(speaker, f"dialogue[{index}].speaker")
        validate_korean_output(utterance, f"dialogue[{index}].text")
        normalized.append({"turn": index, "speaker": speaker, "text": utterance})
    return {"title": title, "dialogue": normalized}


def generate_valid_dialogue(prompt: str) -> dict:
    """잘못된 언어 또는 형식 출력은 같은 입력으로 한 번만 재생성합니다."""
    retry_prompt = prompt
    for attempt in range(1, MAX_DIALOGUE_GENERATION_ATTEMPTS + 1):
        raw_answer = query_llm(retry_prompt, num_predict=1800)
        try:
            return parse_dialogue_response(raw_answer)
        except (json.JSONDecodeError, LLMError) as exc:
            if attempt == MAX_DIALOGUE_GENERATION_ATTEMPTS:
                raise
            print(f"[WARN] 생성 출력 검증 실패, 같은 키워드로 재시도합니다: {exc}")
            retry_prompt = (
                f"{prompt}\n\n[재생성 지시]\n"
                f"이전 출력은 검증에 실패했습니다: {exc}\n"
                "처음부터 다시 작성하고, 제목·speaker·모든 발화에 중국어 한자와 "
                "일본어 가나를 절대 사용하지 마세요."
            )

    raise LLMError("대화 생성 재시도 횟수를 초과했습니다.")


def dialogue_text(dialogue: list[dict]) -> str:
    return "\n".join(f"{turn['speaker']}: {turn['text']}" for turn in dialogue)

def save_generated_scenario(
    generated: dict,
    tech: list[str],
    social: str,
    reason: str,
    reference_files: list[str],
    format_reference_path: str = "",
    format_reference_count: int = 0,
) -> dict:
    timestamp = datetime.now(UTC).isoformat()
    text = dialogue_text(generated["dialogue"])
    record = {
        "generated_at": timestamp,
        "title": generated["title"],
        "dialogue": generated["dialogue"],
        "dialogue_text": text,
        "tech_keywords": tech,
        "social_keyword": social,
        "selected_reason": reason,
        "reference_files": reference_files,
        "format_reference_file": Path(format_reference_path).name if format_reference_path else "",
        "format_reference_count": format_reference_count,
        "scenario": text,
    }

    PATHS["generated_jsonl"].parent.mkdir(parents=True, exist_ok=True)
    with open(PATHS["generated_jsonl"], "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    PATHS["generated_md"].parent.mkdir(parents=True, exist_ok=True)
    with open(PATHS["generated_md"], "a", encoding="utf-8") as f:
        f.write("\n\n---\n\n")
        f.write(f"## Generated at {timestamp}\n")
        f.write(f"- Tech: {', '.join(tech)}\n")
        f.write(f"- Social: {social}\n")
        if reason:
            f.write(f"- Reason: {reason}\n")
        f.write(f"\n### {generated['title']}\n\n")
        for turn in generated["dialogue"]:
            f.write(f"{turn['turn']:02d}. **{turn['speaker']}**: {turn['text']}\n\n")

    return record


# LOGIC
def select_best_keywords(tech_kws: list[str], social_kws: list[str]) -> tuple[list[str], str, str]:
    t_cands = random.sample(tech_kws, min(10, len(tech_kws)))
    s_cands = random.sample(social_kws, min(10, len(social_kws)))
    
    prompt = SELECTION_PROMPT.format(
        tech_candidates_str="\n".join([f"- {k}" for k in t_cands]),
        social_candidates_str="\n".join([f"- {k}" for k in s_cands])
    )
    
    print("--- [Step 1] LLM을 통한 최적 시너지 키워드 선별 ---")
    answer = query_llm(prompt)
    
    tech_match = re.search(r"Tech:\s*(.+)", answer, re.IGNORECASE)
    social_match = re.search(r"Social:\s*(.+)", answer, re.IGNORECASE)
    reason_match = re.search(r"Reason:\s*(.+)", answer, re.IGNORECASE)

    if not tech_match or not social_match:
        raise LLMError(f"키워드 선별 응답 형식이 올바르지 않습니다:\n{answer}")
    
    selected_tech = [tech_match.group(1).strip()]
    selected_social = social_match.group(1).strip()
    selected_reason = reason_match.group(1).strip() if reason_match else ""
    
    return selected_tech, selected_social, selected_reason

def generate_scenario_pipeline() -> dict | None:
    
    tech_pool = load_keywords(PATHS["tech_kw"])
    social_pool = load_keywords(PATHS["social_kw"])
    print(f"[*] Base Pool Sizes - Tech: {len(tech_pool)}, Social: {len(social_pool)}")
    
    sampled_tech, sampled_social, selected_reason = select_best_keywords(tech_pool, social_pool)
    tech_str = ", ".join(sampled_tech)
    
    print(f"[*] Chosen Tech:  [{tech_str}]")
    print(f"[*] Chosen Social: [{sampled_social}]\n")
    print(f"[*] Chosen Reason: [{selected_reason}]\n")
    
    print("--- [Step 2] 시나리오 1종 생성 중... ---")
    reference_files = [
        PATHS[key].name
        for key in ("tactic", "technique", "scenario")
        if PATHS[key].is_file()
    ]
    format_examples = load_format_references(GENERATION_REFERENCE_PATH, GENERATION_REFERENCE_COUNT)
    if format_examples:
        print(
            f"[*] 양성 통화 reference: "
            f"{Path(GENERATION_REFERENCE_PATH).name} ({len(format_examples)}개)"
        )
    prompt = build_generation_prompt(
        tech_str,
        sampled_social,
        selected_reason,
        format_examples=format_examples,
    )
    
    generated = generate_valid_dialogue(prompt)
    rendered = dialogue_text(generated["dialogue"])
    print("\n" + "="*60)
    print("[*] 완성된 보이스피싱 시나리오 결과")
    print("="*60)
    print(f"제목: {generated['title']}")
    print(f"대화 턴: {len(generated['dialogue'])}")
    print(rendered)
    record = save_generated_scenario(
        generated,
        sampled_tech,
        sampled_social,
        selected_reason,
        reference_files,
        format_reference_path=GENERATION_REFERENCE_PATH,
        format_reference_count=len(format_examples),
    )
    print(f"\n[*] 저장 완료: {PATHS['generated_md']}")
    print(f"[*] JSONL 저장 완료: {PATHS['generated_jsonl']}")
    return record

if __name__ == "__main__":
    generate_scenario_pipeline()
