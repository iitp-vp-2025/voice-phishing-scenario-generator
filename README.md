# Voice-phishing scenario pipeline

뉴스 키워드 수집 → 10턴 대화 생성 → DistilKoBERT/KEV 최초 평가 → KEV 최약 항목 최소 refinement → 재평가 → 웹 랭킹 검토를 한 번에 수행하는 방어 연구용 파이프라인입니다.

## 평가 구조

생성 결과의 본체는 설명문이 아닌 10턴의 구조화된 한국어 대화입니다. 같은 대화가 R과 A-D 평가의 원본이 되며, 생성 키워드와 선택 이유는 평가 입력이 아니라 실행 메타데이터로만 저장됩니다.

키워드 선별은 매 시나리오마다 기술 10개와 사회 10개 후보 중 한 조합만 고릅니다. 한국의 금융기관·공공기관 사칭 및 금융·행정 통화 흐름에 자연스럽게 연결할 수 있는 조합을 우선하고, 단순 기기 지원이나 정상 설정 변경으로 끝나는 조합은 제외합니다. 선택한 조합으로 구조화 대화도 하나만 생성하므로 `--count 10`은 생성 대화 10개이며 후보 대화 30개를 만들지 않습니다.

제목·화자·발화와 refinement 교체 문장에 중국어 한자 또는 일본어 가나가 섞이면 코드 검증에서 거부합니다. 최초 생성이 이 검증에 실패하면 같은 키워드와 reference로 한 번만 다시 생성하며, 두 번째 출력도 실패하면 저장하거나 평가하지 않습니다.

- **R 현실성 (30%)**: fine-tuned DistilKoBERT가 대화를 turn 경계 기반 512-token chunk로 평가합니다. 겹치는 chunk의 `P(voice_phishing)` 중 상위 2개 평균을 랭킹에 사용합니다.
- **A 다양성 (25%)**: A1 구조 차별성, A2 기술·맥락 활용, A3 사회공학 메커니즘, A4 탐지·방어 사각지대.
- **B 실행 가능성 (15%)**: B1 자원·접근·기술, B2 피해자 설득 가능성, B3 기관·제도 사실성.
- **C 완결성 (15%)**: C1 전체 논리 일관성, C2 시나리오·상호작용 흐름 완결성.
- **D 일반화 (15%)**: D1 맥락 간 전이, D2 맥락 적응.

A-D는 KEV `/v1/systemone`의 11개 독립 `score` 질문으로 전체 대화를 평가합니다. 모든 질문은 대화에 실제로 나타난 근거만 사용하며, 대화에 없는 배경이나 절차를 가정하지 않습니다. 각 질문은 명시적인 5단계 기준을 가지며 KEV의 0~4 기대 점수, 확률, confidence를 저장합니다.

모든 대화는 최초 평가 후 필수 refinement를 거칩니다. R은 refinement 대상 선정에서 제외하고, 정규화된 A-D 중 가장 낮은 축과 그 축의 최저 세부항목 하나만 선택합니다. 생성 모델은 관련 발화 최대 3개의 교체 문장만 반환하며, 재평가에서 목표 항목과 축이 개선되고 다른 축과 종합 랭킹이 유지된 경우에만 수정본을 채택합니다. 최대 두 번 모두 실패하면 원본을 유지하고 `refinement_status=failed`를 기록합니다.

랭킹은 정규화된 다섯 축의 가중 기하평균입니다.

```text
ranking = 100 × R^0.30 × A^0.25 × B^0.15 × C^0.15 × D^0.15
```

기존 100점 합산과 `A_채택/B_검토후채택/C_보완필요/D_제외` 라벨은 사용하지 않습니다.

## 주요 파일

```text
Collect_keyword.py      최근 뉴스에서 기술/사회 키워드 수집
Generate_scenario.py    sLLM으로 10턴 대화 JSON 생성
Fine_tuned_judge.py     DistilKoBERT checkpoint 및 turn chunk 추론
Evaluate_scenario.py    대화 기반 R + KEV A-D 평가와 랭킹
Run_pipeline.py         대화 1개 생성·평가·refinement
Run_end_to_end.py       수집·생성·최초평가·refinement·최종랭킹
Improve_scenario.py     KEV 최약 세부항목의 최소 turn patch 및 채택 검증
dashboard.py            Streamlit 실행/랭킹/전문가 검토 화면
```

`Tactic.pdf`, `Technique.pdf`, `Scenario.pdf`는 배포하지 않는 선택적 로컬 참고 자료입니다. 생성 단계는 존재하는 파일만 참고하고, 평가 단계는 `Scenario.pdf`가 있을 때만 기존 시나리오를 참고합니다. 파일이 없으면 해당 내용만 생략한 채 계속 실행합니다.

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

이 공개 저장소에는 데이터와 학습된 체크포인트가 포함되지 않습니다. R 평가가 필요하면 승인된 경로를 통해 체크포인트를 별도로 받은 뒤 `.env`의 `DISTILKOBERT_MODEL_PATH`에 로컬 경로를 지정합니다.

KEV는 이 앱과 별도 서버로 실행합니다. 예를 들어 KEV 저장소 환경에서:

```bash
uv run --extra serve python -m kev.serve --run jaredpalmer/kev-0.8b --port 8009
```

GPU 서버 설치, CPU/GPU 실행, SSH 터널 연결 방법은
[`docs/KEV_SERVER.md`](docs/KEV_SERVER.md)에 정리되어 있습니다.

앱의 `.env`에는 KEV 서버의 base URL을 설정합니다. `Evaluate_scenario.py`가 `/v1/systemone`을 자동으로 붙입니다.

```dotenv
KEV_API_URL=http://127.0.0.1:8009
KEV_MODEL=kev-latest
```

## Student checkpoint

이 프로젝트에서 사용하는 파일 위치:

```text
models/realism/best_student.pt
```

이 파일은 `monologg/distilkobert` encoder 전체와 dropout + 2-class linear head를 fine-tuning한 PyTorch `state_dict`입니다. class 순서는 `[normal, voice_phishing]`이고 기본 threshold는 `0.5`입니다. encoder/tokenizer 파일은 처음 실행할 때 Hugging Face에서 내려받아 캐시에 저장되며, remote-code revision은 `model_manifest.json`에 고정했습니다.

확인된 checkpoint SHA-256:

```text
a1f70185412879684b2b1739792f682fe6e0daeb18e215457a6891aa84e83f77
```

이 체크포인트는 학습 데이터에서 만들어진 비공개 파생 산출물이므로 Git 또는 Git LFS로 배포하지 않습니다. 승인된 사용자는 파일을 별도로 전달받아 위 경로 또는 `DISTILKOBERT_MODEL_PATH`가 가리키는 로컬 위치에 둡니다. 파이프라인은 실행 시 실제 파일의 SHA-256을 확인하며, 체크포인트가 없으면 R 평가를 실행할 수 없습니다.

## 실행

평가만 실행:

```bash
python Evaluate_scenario.py --reevaluate
```

생성 후 즉시 평가:

```bash
python Run_pipeline.py
```

전체 파이프라인:

```bash
python Run_end_to_end.py --count 10
```

`Improve_scenario.py`는 별도 선택 단계가 아니라 `Run_pipeline.py`와 `Run_end_to_end.py`에서 항상 실행되는 필수 feedback loop입니다.

생성용 양성 통화 예시는 `--generation-reference` 또는
`VP_GENERATION_REFERENCE_PATH`로 지정합니다. 둘 다 지정하지 않았는데
`tmp/positive_format_refs.jsonl`이 있으면 해당 파일을 자동으로 사용하고, 파일이 없으면
reference 없이 계속 실행합니다. 이 원문 파일은 로컬 전용이며 Git에는 포함하지 않습니다.

웹 화면:

```bash
streamlit run dashboard.py
```

각 end-to-end run에는 최초 평가, refinement trace, 최종 평가, 최종 정렬 데이터셋이 각각 JSONL/CSV로 저장됩니다. 단일 `Run_pipeline.py` 실행도 `outputs/run_pipeline/` 아래에 최초 평가, trace, 최종 평가를 분리해 남깁니다. `dataset/dataset_sorted.csv`는 최종 `ranking_score` 내림차순이며 전체 대화 JSON, R chunk 확률, 최약 KEV 항목, 수정 turn과 채택 상태를 포함합니다.

## 로컬 파일과 GitHub

이 저장소는 code-only 공개를 원칙으로 합니다. 원본 데이터뿐 아니라 생성 대화, refinement 결과, 평가 결과, 전문가 검토 기록 및 학습된 checkpoint도 커밋하지 않습니다.

`.gitignore`는 `.env`, `data/`, `dataset/`, `datasets/`, `outputs/`, `tmp/`, `models/`, `checkpoints/`, CSV/JSONL/Excel/Parquet, model weight 및 PDF 파일을 제외합니다. 로컬 실행 결과는 계속 생성되지만 Git에는 포함되지 않습니다. GitHub에 올리기 전 API key, 내부 서버 URL 및 비공개 파일이 추적 대상에 없는지 별도로 확인하세요.

`Report/`의 문서는 과거 실험과 설계를 보관한 로컬 archive이며, 현재 파이프라인의 권위 있는 설명은 이 `README.md`입니다.
