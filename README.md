# voice phishing scenario generator

**This repository presents the core pipeline for phishing scenario generation using an sLLM.**


This project explores how recent **technology trends** and **social-context trends** can be incorporated into phishing scenario generation.  
The pipeline first collects and curates trend keywords from recent news, then uses those keywords to generate realistic phishing scenarios with an sLLM.

---

## Project Structure

```bash
sLLM
├── Collect_keyword.py              # Collects recent news articles and curates Tech/Social keywords
├── Generate_scenario.py            # Generates phishing scenarios from curated keywords
│
├── news_articles_tech.jsonl        # Collected tech related news articles
├── news_articles_social.jsonl      # Collected social related news articles
├── news_keywords_raw_tech.txt      # Raw extracted tech keywords
├── news_keywords_raw_social.txt    # Raw extracted social keywords
├── tech_keywords_curated.txt       # Final curated tech keyword list
├── social_keywords_curated.txt     # Final curated social keyword list
│
├── Scenario.pdf                    # Reference phishing scenario examples
├── Tactic.pdf                      # Reference tactic document
└── Technique.pdf                   # Reference technique document
```

## Overview

The pipeline has two main stages:

### 1. Keyword Collection

`Collect_keyword.py` performs the following steps:

- Collect recent news articles from predefined **Tech** and **Social** queries
- Use an sLLM to extract short, concrete trend keywords
- Remove low quality or overly generic outputs
- Deduplicate keywords at semantic levels
- Save curated keyword sets

### 2. Scenario Generation

`Generate_scenario.py` performs the following steps:

- Load curated **Tech** and **Social** keyword pools
- Sample candidate keywords from both pools
- Ask the sLLM to select the most realistic **Tech & Social** combination
- Load reference materials such as tactic / technique / scenario documents
- Generate a phishing scenario in Korean based on the selected combination

The generation stage prioritizes **realism over novelty** by first selecting the most plausible keyword combination before producing the final scenario.

## How to Run

### 1.Prepare Configuration

Before running the scripts, you need to provide your own configuration values.

##### For `Collect_keyword.py`

Set the following:

- `NEWS_API_KEY`
- LLM inference server URL (`MODEL_URL`)

##### For `Generate_scenario.py`

Set the following:

- LLM inference server URL (`API_URL`)

You may also need the following files in the project root:

- `tech_keywords_curated.txt`
- `social_keywords_curated.txt`
- `Scenario.pdf`
- `Tactic.pdf`
- `Technique.pdf`

---

### 2. Run Keyword Collection

```bash
python Collect_keyword.py
```

### 3. Run Scenario Generation

```bash
python Generate_Scenario.py
```
