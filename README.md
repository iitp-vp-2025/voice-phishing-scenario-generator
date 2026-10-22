# sLLM Voice Phishing Generation Pipeline

뉴스 키워드 수집부터 한국어 보이스피싱 대화 생성, KEV 평가, 최약 항목 refinement, 최종 랭킹까지 한 번에 수행하고, 최종적으로 전문가가 결과를 쉽게 검토할 수 있는 웹 화면을 제공하는 파이프라인입니다.

## Project Structure

```graphql
sLLM-Voice-Phishing-Generation-Pipeline
├── generation
│   ├── __init__.py
│   ├── keyword_collector.py        # Collect and curate technology/social keywords
│   └── scenario_generator.py       # Generate structured 10-turn Korean dialogues
│
├── scenario_evaluation
│   ├── __init__.py
│   ├── evaluator.py                # Evaluate R and KEV A-D rubric scores
│   ├── realism_judge.py            # Run the fine-tuned DistilKoBERT classifier
│   ├── refiner.py                  # Refine the weakest KEV subcriterion
│   └── model_manifest.json         # Private checkpoint integrity metadata
│
├── references
│   └── README.md                   # Placement guide for local-only references
│
├── best_student.pt                 # Optional local R-model weight; not included in Git
├── main.py                         # (MAIN) Run the complete pipeline
├── dashboard.py                    # Streamlit execution and review page
├── .env.example                    # Environment variable template; commit this file
├── .gitignore                      # Exclude .env, data, outputs, references, and weights
├── requirements.txt                # Python dependencies
└── README.md
```

Data, generated results, reference contents, and model weights are local-only and are not included in this repository. `best_student.pt` is placed at the project root when R evaluation is needed.

## How to Run

Follow the steps below from top to bottom. The pipeline, the generation model server, and the KEV server are three separate processes.

### 1. Clone the repository

```bash
git clone https://github.com/iitp-vp-2025/voice-phishing-scenario-generator.git
cd voice-phishing-scenario-generator
```

### 2. Create the Python environment

Python 3.12 or 3.13 is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the following command whenever you open a new terminal for this project:

```bash
cd voice-phishing-scenario-generator
source .venv/bin/activate
```

### 3. Prepare the generation model server

This pipeline calls an Ollama-compatible `/api/generate` endpoint. If Ollama is not installed, install it from the [official Ollama download page](https://ollama.com/download), then start it and download a generation model.

```bash
ollama serve
```

In another terminal:

```bash
ollama pull <GENERATION_MODEL>
curl -s http://127.0.0.1:11434/v1/models
```

The value used for `<GENERATION_MODEL>` must appear in the model list returned by `curl`.

### 4. Prepare the KEV server

Install and run [Kev](https://github.com/jaredpalmer/kev) in a separate directory. The following example uses the smaller 0.8B model on port `8009`.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd ..
git clone https://github.com/jaredpalmer/kev.git
cd kev
uv sync --extra serve
uv run --extra serve python -m kev.serve \
  --run jaredpalmer/kev-0.8b \
  --host 127.0.0.1 \
  --port 8009
```

Keep that terminal open. In another terminal, verify the server:

```bash
curl -s http://127.0.0.1:8009/v1/models
```

The response must contain `kev-latest` before running the pipeline.

### 5. Connect to remote servers when necessary

Skip this step when Ollama and KEV run on the same computer as the pipeline. When both services run on a remote server, keep the following SSH tunnel open:

```bash
ssh -p <SSH_PORT> -N \
  -L 11500:127.0.0.1:11500 \
  -L 8009:127.0.0.1:8009 \
  <USER>@<SERVER>
```

Then verify both local tunnel endpoints:

```bash
curl -s http://127.0.0.1:11500/v1/models
curl -s http://127.0.0.1:8009/v1/models
```

### 6. Configure `.env`

Return to the pipeline repository, create `.env`, and edit it with the values verified above.

```bash
cd <PATH_TO>/voice-phishing-scenario-generator
cp .env.example .env
```

Example for locally running Ollama and KEV:

```dotenv
NEWS_API_KEY=your_newsapi_key

SLLM_API_URL=http://127.0.0.1:11434/api/generate
SLLM_MODEL=<GENERATION_MODEL>

KEV_API_URL=http://127.0.0.1:8009
KEV_MODEL=kev-latest

DISTILKOBERT_MODEL_PATH=best_student.pt
DISTILKOBERT_DEVICE=auto
```

When the Ollama SSH tunnel uses port `11500`, change only this line:

```dotenv
SLLM_API_URL=http://127.0.0.1:11500/api/generate
```

Create a [NewsAPI key](https://newsapi.org/register) if needed. `NEWS_API_KEY`, `SLLM_API_URL`, `SLLM_MODEL`, `KEV_API_URL`, and `KEV_MODEL` are required. Do not commit `.env`.

### 7. Add optional local files

- To use the R axis, copy `best_student.pt` next to `main.py`.
- To use local references, copy the four files listed in [Reference Files](#reference-files) into `references/`.
- If these files are unavailable, leave them out. Without `best_student.pt`, ranking automatically uses KEV A-D only.

### 8. Run a one-scenario smoke test

From the pipeline repository with `.venv` activated:

```bash
python main.py --count 1 --max-attempts 3
```

The pipeline runs in this order:

```text
News collection
→ keyword selection
→ 10-turn Korean dialogue generation
→ initial evaluation
→ weakest KEV subcriterion refinement
→ final evaluation and ranking
```

On success, the terminal prints `End-to-end pipeline complete` and the created run directory.

### 9. Run the full experiment

After the smoke test succeeds:

```bash
python main.py --count 10 --max-attempts 15
```

Results are created under:

```text
outputs/end_to_end/run_YYYYMMDD_HHMMSS/
├── generate_scenario/generated_scenarios.jsonl
├── evaluate_scenario/final_evaluations.csv
├── refine_scenario/refinement_trace.csv
├── dataset/dataset_sorted.csv
└── manifest.json
```

To evaluate an existing run again without collecting news or regenerating the initial dialogues:

```bash
python main.py --count 10 --reevaluate-run run_YYYYMMDD_HHMMSS
```

### 10. Open the expert review dashboard

```bash
python -m streamlit run dashboard.py
```

Open `http://127.0.0.1:8501` in a browser. Select `결과 검토` to inspect rankings, rubric scores, refined turns, and the generated dialogue.

### Common errors

| Message or symptom | What to check |
|---|---|
| `Connection refused` on `11500` or `8009` | The service or SSH tunnel is not running. Repeat the corresponding `curl` check. |
| `Address already in use` while opening a tunnel | Another tunnel already uses that local port. Test it with `curl` before opening another one. |
| `NEWS_API_KEY를 설정하세요` | Add a valid NewsAPI key to `.env`. |
| `DistilKoBERT checkpoint가 없어 KEV-only` | This is a warning, not a failure. Add `best_student.pt` only when R evaluation is required. |
| DistilKoBERT SHA-256 mismatch | Use the approved checkpoint recorded in `scenario_evaluation/model_manifest.json`. |

## Evaluation

| Axis | Description | Full mode | KEV-only mode |
|---|---|---:|---:|
| R | Voice-phishing realism from fine-tuned DistilKoBERT | 30% | Not used |
| A | Diversity | 25% | 35.71% |
| B | Feasibility | 15% | 21.43% |
| C | Completeness | 15% | 21.43% |
| D | Generalizability and adaptability | 15% | 21.43% |

- R uses the mean of the two highest `P(voice_phishing)` values from turn-boundary-preserving 512-token chunks.
- A-D use 11 KEV score questions and only evidence present in the generated dialogue.
- Refinement targets the weakest A-D subcriterion and replaces at most three turns.

```text
full ranking     = 100 × R^0.30 × A^0.25 × B^0.15 × C^0.15 × D^0.15
KEV-only ranking = 100 × A^0.3571 × B^0.2143 × C^0.2143 × D^0.2143
```

Each result records `evaluation_mode` as `full` or `kev_only`; KEV-only mode leaves all R fields empty instead of fabricating an R score.

## Reference Files

Place the four optional files under `references/`. Their contents are not included in Git.

| File | Role |
|---|---|
| `Tactic.pdf` | Defines the attack objective and overall progression for generation. |
| `Technique.pdf` | Provides concrete technical and social-engineering methods for generation. |
| `Scenario.pdf` | Helps generation avoid repetition and provides comparison context for KEV A-D. |
| `positive_format_refs.jsonl` | Provides dialogue style and interaction structure for generation. |

## Optional DistilKoBERT Model

Place `best_student.pt` directly in the project root, next to `main.py`:

```text
sLLM-Voice-Phishing-Generation-Pipeline/
├── best_student.pt
├── main.py
└── dashboard.py
```

This file enables the R-axis evaluation. If it is absent, the pipeline skips R and runs with KEV A-D only. The model weight and all reference contents remain excluded from Git.

## Acknowledgement

This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. RS2025-02263169, Detection and Prediction of Emerging and Undiscovered Voice Phishing)
