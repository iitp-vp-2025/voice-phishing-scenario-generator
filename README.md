# sLLM 기반 보이스피싱 시나리오 생성 및 평가 파이프라인

최근 뉴스에서 기술·사회 키워드를 수집하고, sLLM으로 한국어 보이스피싱 대화를 생성한 뒤 DistilKoBERT와 KEV rubric으로 평가·개선·정렬하는 방어 연구용 파이프라인입니다.

## Project Structure

```graphql
voice-phishing-scenario-generator
├── Collect_keyword.py              # Collect and curate technology/social keywords
├── Generate_scenario.py            # Generate structured 10-turn Korean dialogues
├── Evaluate_scenario.py            # Evaluate R and KEV A-D rubric scores
├── Fine_tuned_judge.py             # Run the fine-tuned DistilKoBERT classifier
├── Improve_scenario.py             # Refine the weakest KEV subcriterion
├── Run_pipeline.py                 # Generate and evaluate one scenario
├── Run_end_to_end.py               # (MAIN) Run collection, generation, evaluation, refinement, and ranking
├── dashboard.py                    # Streamlit dashboard for execution and review
│
├── docs
│   └── KEV_SERVER.md               # KEV installation, serving, and SSH tunnel guide
│
├── tests
│   ├── test_collect_keyword.py     # Keyword curation tests
│   ├── test_evaluator.py           # Rubric, ranking, and output tests
│   ├── test_fine_tuned_judge.py    # DistilKoBERT dialogue inference tests
│   ├── test_generate_scenario.py   # Prompt and dialogue validation tests
│   └── test_improve_scenario.py    # Refinement validation tests
│
├── .env.example                    # Environment variable template
├── .gitignore                      # Excludes data, outputs, references, and model weights
├── model_manifest.json             # Private checkpoint architecture and integrity metadata
├── requirements.txt                # Python dependencies
└── README.md
```

The following local artifacts are required only for the relevant features and are not included in this repository:

```text
models/realism/best_student.pt       # Private fine-tuned DistilKoBERT checkpoint
tmp/positive_format_refs.jsonl      # Optional positive-format generation references
Tactic.pdf                          # Optional local generation reference
Technique.pdf                       # Optional local generation reference
Scenario.pdf                        # Optional generation/evaluation reference
outputs/                            # Locally generated runs and review results
```

## How to Run?

1. Create and activate a Python virtual environment.

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Create the local environment configuration.

   ```bash
   cp .env.example .env
   ```

   Set the News API key, sLLM endpoint, KEV endpoint, and private checkpoint path in `.env`.

   ```dotenv
   NEWS_API_KEY=your_newsapi_key

   SLLM_API_URL=http://127.0.0.1:11500/api/generate
   SLLM_MODEL=your_generation_model

   KEV_API_URL=http://127.0.0.1:8009
   KEV_MODEL=kev-latest

   DISTILKOBERT_MODEL_PATH=models/realism/best_student.pt
   DISTILKOBERT_DEVICE=auto
   ```

3. Prepare the private local checkpoint.

   Place the approved fine-tuned DistilKoBERT `state_dict` at the path configured by `DISTILKOBERT_MODEL_PATH`. The checkpoint is not distributed through Git or Git LFS. Its SHA-256 must match `model_manifest.json`.

4. Start or connect to the model services.

   - Start an Ollama-compatible sLLM generation endpoint.
   - Start the KEV `/v1/systemone` service.
   - When the services run on another machine, connect them through SSH tunnels.

   KEV installation and serving instructions are available in [`docs/KEV_SERVER.md`](docs/KEV_SERVER.md).

5. Run the complete pipeline from the project root.

   ```bash
   python Run_end_to_end.py --count 10
   ```

   The default execution order is:

   ```text
   News collection
   → keyword curation and selection
   → 10-turn dialogue generation
   → initial R/A/B/C/D evaluation
   → weakest KEV subcriterion refinement
   → final evaluation and ranking
   ```

6. Other execution modes are available as follows.

   **Generate and evaluate one scenario:**

   ```bash
   python Run_pipeline.py
   ```

   **Re-evaluate an existing local run without regenerating dialogues:**

   ```bash
   python Run_end_to_end.py --count 10 --reevaluate-run run_YYYYMMDD_HHMMSS
   ```

   **Open the Streamlit dashboard:**

   ```bash
   streamlit run dashboard.py
   ```

   Then open `http://127.0.0.1:8501`.

7. Run the test suite.

   ```bash
   python -m unittest discover -s tests -v
   ```

## Evaluation

The generated dialogue is evaluated directly. Keyword-selection reasons and other generation metadata are not used as evaluation evidence.

| Axis | Description | Ranking weight |
|---|---|---:|
| R | Voice-phishing realism from the fine-tuned DistilKoBERT classifier | 30% |
| A | Diversity | 25% |
| B | Feasibility | 15% |
| C | Completeness | 15% |
| D | Generalizability and adaptability | 15% |

- **R** is the mean of the two highest `P(voice_phishing)` values from turn-boundary-preserving 512-token chunks.
- **A-D** are calculated from 11 independent KEV score questions. Each question has explicit levels from 0 to 4 and uses only evidence present in the dialogue.
- Refinement excludes R and targets only the weakest A-D axis and its weakest subcriterion. At most three dialogue turns are replaced, and the revision is accepted only when the target improves without a large regression in the remaining scores.

The final ranking uses a weighted geometric mean:

```text
ranking = 100 × R^0.30 × A^0.25 × B^0.15 × C^0.15 × D^0.15
```

## Data Privacy

This is a code-only repository. Raw data, positive reference transcripts, generated dialogues, refinement traces, evaluation results, expert reviews, PDF references, and trained checkpoints are not committed.

The `.gitignore` excludes local data directories, runtime outputs, CSV/JSONL/Excel/Parquet files, PDF references, and model weights. Local execution still creates results under `outputs/`, but those files remain outside Git tracking.

## Acknowledgement

This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. RS2025-02263169, Detection and Prediction of Emerging and Undiscovered Voice Phishing)
