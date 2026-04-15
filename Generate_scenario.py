import re
import random
import requests
from pathlib import Path

try:
    from pdfminer.high_level import extract_text
except ImportError:
    def extract_text(pdf_path): return ""

# CONFIGURATION
SCRIPT_DIR = Path(__file__).parent.resolve()

PATHS = {
    "tech_kw": SCRIPT_DIR / "tech_keywords_curated.txt",
    "social_kw": SCRIPT_DIR / "social_keywords_curated.txt",
    "tactic": SCRIPT_DIR / "Tactic.pdf",
    "technique": SCRIPT_DIR / "Technique.pdf",
    "scenario": SCRIPT_DIR / "Scenario.pdf",
}

API_URL = ""
MODEL_NAME = "gpt-oss:20b"


# PROMPTS
SELECTION_PROMPT = """당신은 세계 최고 수준의 보안 위협 분석가입니다.
아래에 5개의 기술(Tech) 키워드 후보와 5개의 사회(Social) 키워드 후보가 있습니다.

[기술(Tech) 키워드 후보]
{tech_candidates_str}

[사회(Social) 키워드 후보]
{social_candidates_str}

**[선택 기준 - 매우 엄격함]**
이 후보들 중에서, 일반인이 실생활에서 마주쳤을 때 의심 없이 속아 넘어갈 수 있는 **가장 지극히 현실적이고 자연스러운 수법(기술 1~2개, 사회 1개)**만을 선택하세요.

1. **거절(Reject) 기준**:
   - 억지스럽거나 인위적인 조합은 무조건 피하세요.
   - 직관적으로 말이 되지 않는 논리 도약이 있는 쌍은 버리세요.
   - 합이 맞지 않는 빈약한 조합을 그럴싸한 말장난으로 억지 포장(Speculative reasoning)하지 마세요.
   - 일반 사용자가 들었을 때 조금이라도 생소하거나 어색한 조합은 절대 금지합니다.

2. **우대(Prefer) 기준**:
   - 참신함(Novelty)보다 **현실성(Realism)**이 압도적으로 더 중요합니다.
   - 현실에 실존하는 워크플로우(본인 인증, 환불, 예약, 결제 오류, 계정 접근 문제, 기관 사칭 등)에 완벽하게 녹아드는 자연스러운 미끼(Bait)를 고르세요.

반드시 위 기준에 부합하는 가장 현실적인 최고의 조합 1쌍을 찾은 뒤, 아래 양식으로만 답변하세요.

[선택된 조합]
Reason: [왜 이 조합이 일반인에게 의심 없이 통할 수 있는 흔하고 현실적인 사기 흐름인지 1~2문장으로 객관적 설명]
Tech: [선택한 기술 키워드]
Social: [선택한 사회 키워드]
"""

GENERATION_PROMPT = """당신은 세계 최고 수준의 보안 위협 분석가(Threat Analyst)입니다.
아래는 다양한 보이스피싱 전술(Tactic), 기법(Technique), 기존 시나리오 및 관련 키워드입니다.

[전술 정보]
{tactic_text}

[기법 정보]
{technique_text}

[기존 시나리오]
{scenario_text}

[선별된 기술(Tech) 키워드]
{tech_str}

[선별된 사회(Social) 키워드]
{sampled_social}

[각 키워드가 선택된 이유 (Why this combination was selected)]
{selected_reason}

위 정보와 키워드를 바탕으로, 현재 기술 수준에서 실제로 발생 가능성이 매우 높으면서도(Plausibility) 기존 방어망의 사각지대를 찌르는 창의적인(Novelty) 신종 보이스피싱/사기 시나리오를 도출해 주세요.
결과물이 일관되게 출력되도록, 반드시 아래의 [중요 조건]과 [출력 양식]을 엄격하게 준수해야 합니다.

[중요 조건]
1. 모든 결과는 반드시 한국어로 작성할 것.
2. [현실성(Realism) 엄수]: 피해자가 일상에서 겪을 법한 지극히 현실적인 상황 설정. 너무 우연이 겹치거나 SF 영화 같은 과도한 기술 설정은 배제할 것. 당장 오늘이라도 일어날 수 있는 서민/기업 밀착형 범죄로 구성하라. 단, "보안 패치를 위해 송금하라"와 같이 상식적으로 비현실적이고 뻔한 억지스런 결말은 절대 피하라.
3. [융합의 자연스러움]: 기술 후킹 단어와 사회 트렌드 후킹 단어가 유기적으로 연결되어야 합니다. 기술은 범죄의 수단으로, 사회 이슈는 미끼(Bait)로 활용하는 것이 좋습니다.
4. [참신성(Novelty) 확보]: 단순 기관 사칭, 대출 빙자 등 뻔한 수법의 도메인만 바꾼 재작성은 금지. 일상적인 서비스와 사람의 심리적 취약점을 교묘하게 융합할 것.
5. [기존 비교]: 각 시나리오 마지막에는 기존 제공된 20개 시나리오 중 가장 공격 방식이나 대상이 유사하다고 판단되는 시나리오 번호를 명시할 것.

[추가 제약 및 자체 검증 (Self-Reflection)]
중요 제약사항 (Important constraints):
- 시나리오는 오늘날 실제 사용자가 직면할 법한 매우 현실적인 상황이어야 합니다.
- 기술(Tech) 키워드와 사회(Social) 키워드가 억지 없이 자연스럽게 연결되어야 합니다.
- 지나치게 복잡하거나 과도한 기술적 가정은 피하십시오.
- 다음과 같은 뻔하고 비현실적인 패턴으로 절대 마무리하지 마십시오:
  - '보안 패치' 명목으로 비용 지불 요구
  - '업데이트' 명목으로 송금 유도
  - "안전한 보호를 위해" 계정 정보 입력 요구

최종적으로 시나리오를 확정하기 직전에 스스로 다음을 점검하십시오:
1. 전체적인 흐름이 당장 일어날 것처럼 현실감 있게 다가오는가?
2. 미끼(Bait)와 사칭 수법이 너무 노골적이지 않고 자연스러운가?
3. 결말이 어색하거나 기존처럼 지나치게 상투적이지 않은가?
만약 하나라도 해당된다면, 더 현실적이고 교묘한 버전으로 내부 시나리오 로직을 다시 구상하여 출력하십시오.

[출력 양식]
각 시나리오는 반드시 아래의 마크다운 형식을 그대로 복사해서 빈칸을 채우는 방식으로 작성하세요. (불필요한 서론/결론은 생략)

### 시나리오 1: [구체적이고 현실적인 시나리오 제목]
- **핵심 줄거리**: [전체 흐름을 1~2문장으로 명확히 요약]
- **표적 자산**: [탈취하고자 하는 구체적인 정보/자산]
- **실현 가능성(Plausibility) 요소**: [현재 사회적/기술적 배경에서 이 사기가 통할 수밖에 없는 현실적 이유를 1~2문장 설명]
- **주요 공격 단계**:
  1. [초기 접근]: [내용]
  2. [신뢰 확보 및 기만]: [내용]
  3. [최종 탈취/목적 달성]: [내용]
- **가장 유사한 기존 시나리오**: [예: 기존 시나리오 5번] (핵심적인 유사점 1줄 요약)
"""


# UTILITIES
def query_llm(prompt: str) -> str:
    """LLM으로 프롬프트를 전송하고 텍스트 응답을 반환합니다."""
    try:
        res = requests.post(API_URL, json={"model": MODEL_NAME, "prompt": prompt, "stream": False}, timeout=120)
        res.raise_for_status()
        return res.json().get('response', '')
    except Exception as e:
        print(f"[WARN] LLM 통신 실패: {e}")
        return ""

def load_pdf_text(path: Path) -> str:
    if not path.exists() and path.parent.exists():
        for cand in path.parent.iterdir():
            if cand.is_file() and cand.suffix.lower() == ".pdf" and cand.name.lower() == path.name.lower():
                path = cand
                break
                
    if not path.exists(): return ""
    try:
        return str(extract_text(str(path))).strip()
    except Exception:
        return ""

def load_keywords(path: Path) -> list[str]:
    if not path.exists(): return []
    keywords, seen = [], set()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([^|]+)\s*\|\s*(.+)$", line.strip())
        if match:
            kw = match.group(2).strip()
            if kw.lower() not in seen:
                seen.add(kw.lower())
                keywords.append(kw)
    return keywords

def clean_markdown(text: str) -> str:
    return re.sub(r"\*\*(.*?)\*\*", r"\1", re.sub(r"<br\s*/?>", "\n", text))


# LOGIC
def select_best_keywords(tech_kws: list[str], social_kws: list[str]) -> tuple[list[str], str, str]:
    if not tech_kws: tech_kws = ["AI 음성 복제", "QR 결제", "원격 프로그램"]
    if not social_kws: social_kws = ["공연 티켓 양도", "건강 검진 예약", "선거 기부금 환급"]

    # 1. 뽑기 및 프롬프트 포맷팅
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
    
    selected_tech = [tech_match.group(1).strip()] if tech_match else [t_cands[0]]
    selected_social = social_match.group(1).strip() if social_match else s_cands[0]
    selected_reason = reason_match.group(1).strip() if reason_match else ""
    
    return selected_tech, selected_social, selected_reason

def generate_scenario_pipeline():
    
    tech_pool = load_keywords(PATHS["tech_kw"])
    social_pool = load_keywords(PATHS["social_kw"])
    print(f"[*] Base Pool Sizes - Tech: {len(tech_pool)}, Social: {len(social_pool)}")
    
    sampled_tech, sampled_social, selected_reason = select_best_keywords(tech_pool, social_pool)
    tech_str = ", ".join(sampled_tech)
    
    print(f"[*] Chosen Tech:  [{tech_str}]")
    print(f"[*] Chosen Social: [{sampled_social}]\n")
    print(f"[*] Chosen Reason: [{selected_reason}]\n")
    
    print("--- [Step 2] 시나리오 1종 생성 중... ---")
    prompt = GENERATION_PROMPT.format(
        tactic_text=load_pdf_text(PATHS["tactic"]),
        technique_text=load_pdf_text(PATHS["technique"]),
        scenario_text=load_pdf_text(PATHS["scenario"]),
        tech_str=tech_str,
        sampled_social=sampled_social,
        selected_reason=selected_reason
    )
    
    raw_answer = query_llm(prompt)
    if raw_answer:
        print("\n" + "="*60)
        print("[*] 완성된 보이스피싱 시나리오 결과")
        print("="*60)
        print(clean_markdown(raw_answer))

if __name__ == "__main__":
    generate_scenario_pipeline()