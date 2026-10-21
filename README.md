# sLLM Voice Phishing Generation Pipeline

뉴스 키워드 수집, 한국어 보이스피싱 대화 생성, KEV 평가, 최약 항목 refinement와 최종 랭킹을 한 번에 수행하는 파이프라인입니다. 승인된 DistilKoBERT checkpoint가 있으면 현실성 R 평가도 자동으로 추가합니다.

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
├── main.py                         # (MAIN) Run the complete pipeline
├── dashboard.py                    # Streamlit execution and review page
├── .env.example                    # Environment variable template; commit this file
├── .gitignore                      # Exclude .env, data, outputs, references, and weights
├── requirements.txt                # Python dependencies
└── README.md
```

Data, generated results, reference contents, and model weights are local-only and are not included in this repository. The tracked `references/README.md` keeps the expected local layout visible after cloning.

## How to Run?

1. Create and activate the Python environment.

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env
   ```

2. Configure `.env`.

   ```dotenv
   NEWS_API_KEY=your_newsapi_key

   SLLM_API_URL=http://127.0.0.1:11500/api/generate
   SLLM_MODEL=your_generation_model

   KEV_API_URL=http://127.0.0.1:8009
   KEV_MODEL=kev-latest

   DISTILKOBERT_MODEL_PATH=models/realism/best_student.pt
   DISTILKOBERT_DEVICE=auto
   ```

3. (Optional) Add generation/evaluation references under `references/`. See [Local Inputs](#local-inputs) for the exact filenames and purpose of each file. Missing references are skipped automatically.

4. (Optional) To include the R axis, place the approved private DistilKoBERT checkpoint at `DISTILKOBERT_MODEL_PATH`. Its SHA-256 must match `scenario_evaluation/model_manifest.json`. If the path is empty or the file does not exist, the pipeline automatically runs in KEV-only A-D mode.

5. Start or connect to the sLLM and KEV services. When they run on another machine, use an SSH tunnel.

   ```bash
   ssh -p <SSH_PORT> -N \
     -L 11500:127.0.0.1:11500 \
     -L 8009:127.0.0.1:8009 \
     <USER>@<SERVER>
   ```

6. Run the complete pipeline from the project root.

   ```bash
   python main.py --count 10
   ```

   ```text
   News collection
   → keyword selection
   → 10-turn dialogue generation
   → initial R/A/B/C/D evaluation (or A/B/C/D when no checkpoint exists)
   → weakest KEV subcriterion refinement
   → final evaluation and ranking
   ```

7. To re-evaluate an existing local run without regenerating dialogues:

   ```bash
   python main.py --count 10 --reevaluate-run run_YYYYMMDD_HHMMSS
   ```

8. Open the review dashboard.

   ```bash
   streamlit run dashboard.py
   ```

   Open `http://127.0.0.1:8501` in a browser.

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

## Local Inputs

Put optional reference inputs in `references/`. The pipeline checks these exact paths automatically; missing files are skipped.

| Local path | Used by | Purpose |
|---|---|---|
| `references/Tactic.pdf` | Generation | Supplies high-level attack-tactic context to the dialogue-generation prompt. |
| `references/Technique.pdf` | Generation | Supplies concrete technical and social-engineering technique context. |
| `references/Scenario.pdf` | Generation and KEV evaluation | Provides existing-scenario context during generation and optional comparison context during A-D evaluation. |
| `references/positive_format_refs.jsonl` | Generation | Supplies private positive-call examples for dialogue style and interaction structure. It is never sent to DistilKoBERT or KEV. 
| `models/realism/best_student.pt` | Optional R evaluation | Supplies the fine-tuned DistilKoBERT weights. Without it, the pipeline automatically uses KEV-only A-D ranking. |

`positive_format_refs.jsonl` accepts one JSON object per line. A minimal record is:

```json
{"text": "speaker와 발화가 포함된 통화 전사", "label": 1}
```

All files above remain excluded from Git. `outputs/` is also ignored, but it is created automatically by the pipeline and users should not place input files there.

## Acknowledgement

This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. RS2025-02263169, Detection and Prediction of Emerging and Undiscovered Voice Phishing)
