# sLLM 기반 보이스피싱 시나리오 생성 및 평가 파이프라인

뉴스 키워드 수집, 한국어 보이스피싱 대화 생성, DistilKoBERT·KEV 평가, 최약 항목 refinement와 최종 랭킹을 한 번에 수행하는 방어 연구용 파이프라인입니다.

## Project Structure

```graphql
voice-phishing-scenario-generator
├── Collect_keyword.py          # Collect and curate technology/social keywords
├── Generate_scenario.py        # Generate structured 10-turn Korean dialogues
├── Evaluate_scenario.py        # Evaluate R and KEV A-D rubric scores
├── Fine_tuned_judge.py         # Run the fine-tuned DistilKoBERT classifier
├── Improve_scenario.py         # Refine the weakest KEV subcriterion
├── Run_end_to_end.py           # (MAIN) Run the complete pipeline
├── dashboard.py                # Streamlit execution and review page
├── model_manifest.json         # Private checkpoint integrity metadata
├── .env.example                # Environment variable template
├── .gitignore                  # Exclude data, outputs, references, and weights
├── requirements.txt            # Python dependencies
└── README.md
```

Data, generated results, PDF references, and model weights are local-only and are not included in this repository.

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

3. Place the approved private DistilKoBERT checkpoint at the path configured by `DISTILKOBERT_MODEL_PATH`. The file SHA-256 must match `model_manifest.json`.

4. Start or connect to the sLLM and KEV services. When they run on another machine, use an SSH tunnel.

   ```bash
   ssh -p <SSH_PORT> -N \
     -L 11500:127.0.0.1:11500 \
     -L 8009:127.0.0.1:8009 \
     <USER>@<SERVER>
   ```

5. Run the complete pipeline from the project root.

   ```bash
   python Run_end_to_end.py --count 10
   ```

   ```text
   News collection
   → keyword selection
   → 10-turn dialogue generation
   → initial R/A/B/C/D evaluation
   → weakest KEV subcriterion refinement
   → final evaluation and ranking
   ```

6. To re-evaluate an existing local run without regenerating dialogues:

   ```bash
   python Run_end_to_end.py --count 10 --reevaluate-run run_YYYYMMDD_HHMMSS
   ```

7. Open the review dashboard.

   ```bash
   streamlit run dashboard.py
   ```

   Open `http://127.0.0.1:8501` in a browser.

## Evaluation

| Axis | Description | Ranking weight |
|---|---|---:|
| R | Voice-phishing realism from fine-tuned DistilKoBERT | 30% |
| A | Diversity | 25% |
| B | Feasibility | 15% |
| C | Completeness | 15% |
| D | Generalizability and adaptability | 15% |

- R uses the mean of the two highest `P(voice_phishing)` values from turn-boundary-preserving 512-token chunks.
- A-D use 11 KEV score questions and only evidence present in the generated dialogue.
- Refinement targets the weakest A-D subcriterion and replaces at most three turns.

```text
ranking = 100 × R^0.30 × A^0.25 × B^0.15 × C^0.15 × D^0.15
```

## Local-only Files

The following paths are excluded from Git:

```text
models/realism/best_student.pt
tmp/positive_format_refs.jsonl
Tactic.pdf
Technique.pdf
Scenario.pdf
outputs/
```

## Acknowledgement

This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. RS2025-02263169, Detection and Prediction of Emerging and Undiscovered Voice Phishing)
