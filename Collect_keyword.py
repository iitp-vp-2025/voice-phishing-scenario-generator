import os
import re
import json
import random
import requests
from pathlib import Path
from datetime import datetime, timedelta, UTC

# CONFIGURATION
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
NEWS_API_URL = "https://newsapi.org/v2/everything"
MODEL_URL = ""
MODEL_NAME = "gpt-oss:20b"

# File paths
SCRIPT_DIR = Path(__file__).parent.resolve()

OUTPUT_TECH_ARTICLES = SCRIPT_DIR / "news_articles_tech.jsonl"
OUTPUT_SOCIAL_ARTICLES = SCRIPT_DIR / "news_articles_social.jsonl"

OUTPUT_TECH_RAW = SCRIPT_DIR / "news_keywords_raw_tech.txt"
OUTPUT_SOCIAL_RAW = SCRIPT_DIR / "news_keywords_raw_social.txt"

OUTPUT_TECH_CURATED = SCRIPT_DIR / "tech_keywords_curated.txt"
OUTPUT_SOCIAL_CURATED = SCRIPT_DIR / "social_keywords_curated.txt"

# Queries
TECH_QUERIES = [
    '"AI assistant" OR deepfake OR "voice clone"',
    '"mobile wallet" OR "QR payment" OR "digital identity"',
    '"smart home" OR IoT OR "biometric authentication"',
    '"remote work" OR VPN OR SSO OR "enterprise security"',
    'crypto OR "digital asset" OR wallet'
]

SOCIAL_QUERIES = [
    'sports OR tournament OR ticket OR resale',
    'election OR campaign donation OR voter notice',
    'tax refund OR public benefits OR subsidy notice',
    'hospital reservation OR insurance claim OR health screening', 
]

STOPWORDS = {"security", "update", "fraud", "scam", "policy", "issue", "cybersecurity", "phishing"}

# PROMPTS
PROMPT_EXTRACTION = """You are a research assistant tasked with extracting key trending concepts from recent news articles for defensive security research.

[News Articles]
{articles_text}

Task: Extract 1 to 3 specific trend keywords per article that could be plausibly used as SPECIFIC CONCRETE HOOKS in voice phishing scenarios.

STRICT RULES:
- Output ONLY short noun phrases (1~4 words).
- Do NOT extract generic attack methods (e.g., "security", "update", "fraud", "phishing"). Find concrete concepts (e.g., "deepfake voice clone", "ticket resale platform", "health screening appointment").
- Output exactly in this format: CATEGORY | keyword
- Do NOT output anything else. No scripts, no reasoning.
- The keyword must be able to act as a contextual hook when later combined with other themes.

Example outputs:
Finance | biometric mobile wallet
Tech | synthetic voice authentication
Social | concert ticket resale
Health | insurance claim process
"""

PROMPT_SEMANTIC_DEDUP = """You are a research assistant tasked with deduplicating a list of trending keywords.
Some keywords share the exact same underlying concept or semantic meaning, just phrased slightly differently.

[Candidate Keywords]
{keywords_text}

Task: Review the list and merge keywords ONLY if they mean the exact same thing or are exceedingly similar.
Keep the most descriptive, natural-sounding version as the representative keyword.
IMPORTANT: Do NOT drop a concept completely. Every distinct concept must be preserved.

STRICT RULES:
- Output ONLY the final logically reduced list in the same format: CATEGORY | keyword
- One keyword per line.
- Do NOT output any reasoning, markdown blocks outside the format, or original unmerged versions.

Example of merging:
If input is:
{axis} | {ex1}
{axis} | {ex2}
{axis} | {ex3}
You should output:
{axis} | {ex_merged}
"""

# ==========================================
# UTILS
def parse_line(line: str) -> tuple[str, str] | None:
    raw = line.strip()
    if not raw or raw.startswith("#"):
        return None
    m = re.match(r"^([^|]+)\s*\|\s*(.+)$", raw)
    if not m:
        return None
    category, keyword = m.group(1).strip(), m.group(2).strip()
    if not category or not keyword:
        return None
    return (category, keyword)

def execute_llm(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(MODEL_URL, json=payload, timeout=300)
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] LLM request failed: {e}")
        return ""

# STEP 1: COLLECT NEWS
def fetch_articles(query, from_date, to_date):
    if not NEWS_API_KEY:
        print("[WARN] NEWS_API_KEY is not set. Environment variable is missing.")
        return []
    params = {
        "q": query,
        "from": from_date, 
        "to": to_date,
        "sortBy": "publishedAt", 
        "language": "en", 
        "apiKey": NEWS_API_KEY, 
        "pageSize": 30
    }
    try:
        response = requests.get(NEWS_API_URL, params=params, timeout=30)
        response.raise_for_status()
        return response.json().get("articles", [])
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Fetching '{query}': {e}")
        return []

def get_articles_for_axis(queries, from_date, to_date, output_path):
    seen_urls, valid_articles = set(), []
    for query in queries:
        articles = fetch_articles(query, from_date, to_date)
        for article in articles:
            url = article.get("url")
            if not url or url in seen_urls: continue
            title, desc = article.get("title") or "", article.get("description") or ""
            if len(title) < 15 or len(desc) < 15 or title == "[Removed]": continue
            seen_urls.add(url)
            valid_articles.append({
                "title": title, "description": desc, "publishedAt": article.get("publishedAt"),
                "source": article.get("source", {}).get("name"), "url": url
            })
    
    with open(output_path, "w", encoding="utf-8") as f:
        for article in valid_articles:
            f.write(json.dumps(article, ensure_ascii=False) + "\n")
    return valid_articles

def collect_news():
    print("--- [Step 1] Collecting News ---")
    today = datetime.now(UTC)
    last_30_days = today - timedelta(days=30)
    to_date = today.strftime("%Y-%m-%d")
    from_date = last_30_days.strftime("%Y-%m-%d")

    tech_articles = get_articles_for_axis(TECH_QUERIES, from_date, to_date, OUTPUT_TECH_ARTICLES)
    print(f"Tech articles collected: {len(tech_articles)}")

    social_articles = get_articles_for_axis(SOCIAL_QUERIES, from_date, to_date, OUTPUT_SOCIAL_ARTICLES)
    print(f"Social articles collected: {len(social_articles)}")
    
    return tech_articles, social_articles

# STEP 2: EXTRACT KEYWORDS
def extract_axis_keywords(articles, output_path, axis_name):
    print(f"\n--- [Step 2] Extracting {axis_name} Keywords via sLLM ---")
    if not articles:
        print(f"No articles for {axis_name}.")
        return

    # Use 'w' mode to refresh the raw text on each run
    with open(output_path, "w", encoding="utf-8") as out_f:
        added = 0
        for i in range(0, len(articles), 5):
            batch = articles[i:i+5]
            articles_text = "".join(f"Article {idx}:\nTitle: {a['title']}\nDesc: {a.get('description','')}\n\n" for idx, a in enumerate(batch, 1))
            
            answer_text = execute_llm(PROMPT_EXTRACTION.format(articles_text=articles_text.strip()))

            for line in answer_text.splitlines():
                parsed = parse_line(line)
                if parsed:
                    out_f.write(f"{parsed[0]} | {parsed[1]}\n")
                    added += 1
        print(f"Extracted {added} raw {axis_name} keywords.")

def extract_keywords(tech_articles, social_articles):
    extract_axis_keywords(tech_articles, OUTPUT_TECH_RAW, "Tech")
    extract_axis_keywords(social_articles, OUTPUT_SOCIAL_RAW, "Social")

# STEP 3: SEMANTIC DEDUP & CURATION
def curate_axis(raw_path, curated_path, axis_name):
    print(f"\n--- [Step 3] Curating {axis_name} Keywords ---")
    if not raw_path.exists():
        print(f"Raw file {raw_path} not found.")
        return []

    # 1. Exact & Low-level Dedup + Stopword Filtering
    seen_keys = set()
    exact_dedupted = []
    
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_line(line)
            if not parsed: continue
            category, keyword = parsed
            kw_lower = keyword.lower()
            
            if kw_lower in seen_keys or len(kw_lower) < 2 or not (1 <= len(kw_lower.split()) <= 4):
                continue
            if any(stop in kw_lower for stop in STOPWORDS):
                continue
            
            seen_keys.add(kw_lower)
            exact_dedupted.append(f"{category} | {keyword}")

    if not exact_dedupted:
        print(f"No valid keywords found for {axis_name} after exact dedup.")
        return []

    # 2. Semantic Dedup via LLM
    print(f"Found {len(exact_dedupted)} unique {axis_name} keywords. Running semantic dedup...")
    keywords_text = "\n".join(exact_dedupted)
    
    if axis_name == "Tech":
        ex1, ex2, ex3, ex_merged = "voice clone", "deepfake voice", "synthetic voice call", "deepfake voice clone"
    else:
        ex1, ex2, ex3, ex_merged = "concert ticket resale", "ticket scalping fraud", "fake event tickets", "concert ticket resale fraud"
        
    semantic_answer = execute_llm(PROMPT_SEMANTIC_DEDUP.format(
        keywords_text=keywords_text,
        axis=axis_name,
        ex1=ex1, ex2=ex2, ex3=ex3, ex_merged=ex_merged
    ))

    final_curated = []
    seen_semantic = set()
    for line in semantic_answer.splitlines():
        parsed = parse_line(line)
        if parsed:
            category, keyword = parsed
            kw_lower = keyword.lower()
            if kw_lower not in seen_semantic:
                seen_semantic.add(kw_lower)
                final_curated.append(f"{category} | {keyword}")

    if not final_curated:
        print(f"[WARN] Semantic dedup returned 0 keywords! Falling back to exact dedup (keeping {len(exact_dedupted)} keywords).")
        final_curated = exact_dedupted

    # 3. Save Curated
    with open(curated_path, "w", encoding="utf-8") as f:
        for line in final_curated:
            f.write(line + "\n")
            
    print(f"Total curated {axis_name} keywords after semantic dedup: {len(final_curated)}")
    return final_curated

def curate_keywords():
    tech_curated = curate_axis(OUTPUT_TECH_RAW, OUTPUT_TECH_CURATED, "Tech")
    social_curated = curate_axis(OUTPUT_SOCIAL_RAW, OUTPUT_SOCIAL_CURATED, "Social")
    return tech_curated, social_curated

# MAIN EXECUTION
if __name__ == "__main__":
    tech_articles, social_articles = collect_news()
    
    if tech_articles or social_articles:
        extract_keywords(tech_articles, social_articles)
        
    curate_keywords()

