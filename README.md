# sLLM Voice Phishing Generation Pipeline

This pipeline collects news keywords, generates Korean voice-phishing dialogues, evaluates them with KEV, refines the weakest criterion, produces a final ranking, and provides a web interface for expert review.

## How to Run

### 1. Clone and install the repository

```bash
git clone https://github.com/iitp-vp-2025/voice-phishing-scenario-generator.git
cd voice-phishing-scenario-generator

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

### 2. Start the generation model server

Install [Ollama](https://ollama.com/download), then start it in the first terminal. Skip this command if Ollama is already running.

```bash
ollama serve
```

Download the generation model in a second terminal.

```bash
ollama pull qwen3.8:27b-q4_K_M
curl -s http://127.0.0.1:11434/v1/models
```

The server is ready when the response contains `qwen3.8:27b-q4_K_M`.

### 3. Start the KEV server

In a third terminal, install [Kev](https://github.com/jaredpalmer/kev) in a directory outside the pipeline repository and start the server.

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

Verify the server from a fourth terminal.

```bash
curl -s http://127.0.0.1:8009/v1/models
```

The server is ready when the response contains `kev-latest`.

If the generation model and KEV run on another computer, keep the following SSH tunnel open on the local computer. In this case, change the generation model port in `.env` to `11500`.

```bash
ssh -p <SSH_PORT> -N \
  -L 11500:127.0.0.1:11500 \
  -L 8009:127.0.0.1:8009 \
  <USER>@<SERVER>
```

### 4. Configure `.env`

Open `.env` in the pipeline repository and set the following values. Create a `NEWS_API_KEY` on [NewsAPI](https://newsapi.org/register).

```dotenv
NEWS_API_KEY=your_newsapi_key

SLLM_API_URL=http://127.0.0.1:11434/api/generate
SLLM_MODEL=qwen3.8:27b-q4_K_M

KEV_API_URL=http://127.0.0.1:8009
KEV_MODEL=kev-latest

DISTILKOBERT_MODEL_PATH=best_student.pt
DISTILKOBERT_DEVICE=auto
```

If `best_student.pt` is unavailable, leave the last two lines unchanged. The pipeline will skip R evaluation and rank the results using KEV criteria A-D only.

### 5. Add optional local files

Copy the following files to the specified locations when available. The pipeline can run without them.

| Location | Purpose |
|---|---|
| `best_student.pt` | Fine-tuned DistilKoBERT weights for R evaluation; place at the project root |
| `references/Tactic.pdf` | Reference for attack objectives and overall progression |
| `references/Technique.pdf` | Reference for technical and social-engineering methods |
| `references/Scenario.pdf` | Reference for reducing similarity to existing scenarios |
| `references/positive_format_refs.jsonl` | Reference for positive-dialogue style and interaction structure |

These files are excluded from Git.

### 6. Run the pipeline

Open a new terminal, return to the pipeline repository, and activate the virtual environment.

```bash
cd voice-phishing-scenario-generator
source .venv/bin/activate
```

Start with a one-scenario test.

```bash
python main.py --count 1 --max-attempts 3
```

After the test succeeds, generate ten scenarios.

```bash
python main.py --count 10 --max-attempts 15
```

The final ranked CSV is saved at:

```text
outputs/end_to_end/run_YYYYMMDD_HHMMSS/dataset/dataset_sorted.csv
```

### 7. Open the web interface

```bash
python -m streamlit run dashboard.py
```

Open `http://127.0.0.1:8501` in a browser and select the results review page.

## Project Structure

```text
voice-phishing-scenario-generator/
├── generation/                 # Keyword collection and scenario generation
├── scenario_evaluation/        # R and KEV evaluation and refinement
├── references/                 # Optional local reference files
├── best_student.pt             # Optional R-evaluation weights (excluded from Git)
├── main.py                     # Complete pipeline entry point
├── dashboard.py                # Streamlit execution and review interface
├── .env.example                # Environment variable template
└── requirements.txt            # Python dependencies
```

## Acknowledgement

This work was supported by Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korean government (MSIT) (No. RS2025-02263169, Detection and Prediction of Emerging and Undiscovered Voice Phishing).
