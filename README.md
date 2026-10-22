# sLLM Voice Phishing Generation Pipeline

뉴스 키워드 수집, 한국어 보이스피싱 대화 생성, KEV 평가, 최약 항목 refinement, 최종 랭킹을 한 번에 수행하고 전문가가 결과를 쉽게 검토할 수 있는 웹 화면을 제공합니다.

## How to Run

### 1. 저장소 설치

```bash
git clone https://github.com/iitp-vp-2025/voice-phishing-scenario-generator.git
cd voice-phishing-scenario-generator

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

### 2. 생성 모델 서버 실행

[Ollama](https://ollama.com/download)를 설치한 뒤 첫 번째 터미널에서 실행합니다. 이미 Ollama가 실행 중이면 이 명령은 생략합니다.

```bash
ollama serve
```

두 번째 터미널에서 사용할 모델을 받습니다.

```bash
ollama pull qwen3.8:27b-q4_K_M
curl -s http://127.0.0.1:11434/v1/models
```

응답에 `qwen3.8:27b-q4_K_M`이 보이면 준비된 상태입니다.

### 3. KEV 서버 실행

세 번째 터미널에서 [Kev](https://github.com/jaredpalmer/kev)를 파이프라인 저장소와 별도 폴더에 설치하고 실행합니다.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

cd ..
git clone https://github.com/jaredpalmer/kev.git
cd kev
uv sync --extra serve
uv run --extra serve python -m kev.serve \
  --run jaredpalmer/kev-0.8b \
  --host 127.0.0.1 \
  --port 8009
```

네 번째 터미널에서 서버 응답을 확인합니다.

```bash
curl -s http://127.0.0.1:8009/v1/models
```

응답에 `kev-latest`가 보이면 준비된 상태입니다.

서버를 다른 컴퓨터에서 실행한다면 로컬 터미널에서 다음 터널을 계속 열어 둡니다. 이 경우 아래 `.env`의 생성 모델 포트를 `11500`으로 바꿉니다.

```bash
ssh -p <SSH_PORT> -N \
  -L 11500:127.0.0.1:11500 \
  -L 8009:127.0.0.1:8009 \
  <USER>@<SERVER>
```

### 4. `.env` 설정

파이프라인 저장소의 `.env`를 열고 다음 값을 설정합니다. `NEWS_API_KEY`는 [NewsAPI](https://newsapi.org/register)에서 발급받습니다.

```dotenv
NEWS_API_KEY=발급받은_NewsAPI_키

SLLM_API_URL=http://127.0.0.1:11434/api/generate
SLLM_MODEL=qwen3.8:27b-q4_K_M

KEV_API_URL=http://127.0.0.1:8009
KEV_MODEL=kev-latest

DISTILKOBERT_MODEL_PATH=best_student.pt
DISTILKOBERT_DEVICE=auto
```

`best_student.pt`가 없으면 마지막 두 줄은 그대로 두어도 됩니다. 파이프라인이 R 평가를 건너뛰고 KEV A-D만 사용합니다.

### 5. 선택 파일 배치

파일이 있다면 아래 위치에 복사합니다. 전부 없어도 파이프라인은 실행됩니다.

| 위치 | 역할 |
|---|---|
| `best_student.pt` | 프로젝트 루트에 두며, DistilKoBERT의 R 평가에 사용 |
| `references/Tactic.pdf` | 공격 목적과 전체 전개 참고 |
| `references/Technique.pdf` | 구체적인 기술·사회공학 수법 참고 |
| `references/Scenario.pdf` | 기존 시나리오와 중복을 줄이기 위한 참고 |
| `references/positive_format_refs.jsonl` | 양성 대화의 문체와 상호작용 형식 참고 |

이 파일들은 Git에 포함되지 않습니다.

### 6. 파이프라인 실행

새 터미널에서 파이프라인 저장소로 이동하고 가상환경을 활성화합니다.

```bash
cd voice-phishing-scenario-generator
source .venv/bin/activate
```

먼저 1개로 동작을 확인합니다.

```bash
python main.py --count 1 --max-attempts 3
```

성공하면 10개를 생성합니다.

```bash
python main.py --count 10 --max-attempts 15
```

최종 결과는 다음 위치에 저장됩니다.

```text
outputs/end_to_end/run_YYYYMMDD_HHMMSS/dataset/dataset_sorted.csv
```

### 7. 웹 화면 실행

```bash
python -m streamlit run dashboard.py
```

브라우저에서 `http://127.0.0.1:8501`을 열고 `결과 검토`를 선택합니다.

## Project Structure

```text
voice-phishing-scenario-generator/
├── generation/                 # 키워드 수집 및 시나리오 생성
├── scenario_evaluation/        # R·KEV 평가 및 refinement
├── references/                 # 선택적인 로컬 참고 자료
├── best_student.pt             # 선택적인 R 평가 가중치(Git 제외)
├── main.py                     # 전체 파이프라인 실행
├── dashboard.py                # Streamlit 실행·검토 화면
├── .env.example                # 환경변수 예시
└── requirements.txt            # Python 패키지 목록
```

## Acknowledgement

This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. RS2025-02263169, Detection and Prediction of Emerging and Undiscovered Voice Phishing)
