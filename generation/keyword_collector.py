import json
import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import requests

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "collect_keyword"
NEWS_API_URL = "https://newsapi.org/v2/everything"
NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")
MODEL_URL = os.environ.get("SLLM_API_URL") or os.environ.get("OLLAMA_API_URL", "")
MODEL_NAME = os.environ.get("SLLM_MODEL", "")

QUERIES = {
    "Tech": [
        '"AI assistant" OR deepfake OR "voice clone"',
        '"mobile wallet" OR "QR payment" OR "digital identity"',
        '"smart home" OR IoT OR "biometric authentication"',
        '"remote work" OR VPN OR SSO OR "enterprise security"',
        'crypto OR "digital asset" OR wallet',
    ],
    "Social": [
        "sports OR tournament OR ticket OR resale",
        "election OR campaign donation OR voter notice",
        "tax refund OR public benefits OR subsidy notice",
        "hospital reservation OR insurance claim OR health screening",
    ],
}

STOPWORDS = {"security", "update", "fraud", "scam", "policy", "issue", "cybersecurity", "phishing"}

PROMPT_EXTRACTION = """You are extracting trend keywords from recent news for defensive security research.

[News Articles]
{articles_text}

Extract 1 to 3 concrete contextual hook keywords per article.
Rules:
- Output ONLY short noun phrases, 1 to 4 words.
- Avoid generic attack words such as security, update, fraud, phishing.
- Format exactly: CATEGORY | keyword
- No reasoning, scripts, or extra text.

Examples:
Finance | biometric mobile wallet
Tech | synthetic voice authentication
Social | concert ticket resale
Health | insurance claim process
"""

def path_for(axis: str, kind: str) -> Path:
    lower = axis.lower()
    return OUTPUT_DIR / {
        "articles": f"news_articles_{lower}.jsonl",
        "raw": f"news_keywords_raw_{lower}.txt",
        "curated": f"{lower}_keywords_curated.txt",
    }[kind]


def parse_line(line: str) -> tuple[str, str] | None:
    match = re.match(r"^([^|]+)\s*\|\s*(.+)$", line.strip())
    if not match:
        return None
    category, keyword = match.group(1).strip(), match.group(2).strip()
    return (category, keyword) if category and keyword else None


def call_llm(prompt: str, timeout: int = 300) -> str:
    if not MODEL_URL or not MODEL_NAME:
        raise ValueError("SLLM_API_URL과 SLLM_MODEL을 설정하세요.")
    try:
        response = requests.post(
            MODEL_URL,
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False, "think": False},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.exceptions.RequestException as exc:
        body = getattr(getattr(exc, "response", None), "text", "")
        print(f"[ERROR] LLM request failed: {exc} {body[:500]}")
        return ""


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")


def fetch_articles(query: str, from_date: str, to_date: str) -> list[dict]:
    if not NEWS_API_KEY:
        raise ValueError("NEWS_API_KEY를 설정하세요.")
    params = {
        "q": query,
        "from": from_date,
        "to": to_date,
        "sortBy": "publishedAt",
        "language": "en",
        "apiKey": NEWS_API_KEY,
        "pageSize": 30,
    }
    try:
        response = requests.get(NEWS_API_URL, params=params, timeout=30)
        response.raise_for_status()
        return response.json().get("articles", [])
    except requests.exceptions.RequestException as exc:
        print(f"[ERROR] Fetching news for query '{query}': {exc}")
        return []


def clean_article(article: dict, seen_urls: set[str]) -> dict | None:
    title = article.get("title") or ""
    desc = article.get("description") or ""
    url = article.get("url")
    if not url or url in seen_urls or title == "[Removed]" or len(title) < 15 or len(desc) < 15:
        return None

    seen_urls.add(url)
    return {
        "title": title,
        "description": desc,
        "publishedAt": article.get("publishedAt"),
        "source": article.get("source", {}).get("name"),
        "url": url,
    }


def collect_axis(axis: str, from_date: str, to_date: str) -> list[dict]:
    seen_urls, articles = set(), []
    for query in QUERIES[axis]:
        for article in fetch_articles(query, from_date, to_date):
            cleaned = clean_article(article, seen_urls)
            if cleaned:
                articles.append(cleaned)

    write_jsonl(path_for(axis, "articles"), articles)
    print(f"{axis} articles collected: {len(articles)}")
    return articles


def collect_news(days: int = 30) -> dict[str, list[dict]]:
    print("--- [Step 1] Collecting News ---")
    today = datetime.now(UTC)
    from_date = (today - timedelta(days=days)).strftime("%Y-%m-%d")
    to_date = today.strftime("%Y-%m-%d")
    return {axis: collect_axis(axis, from_date, to_date) for axis in QUERIES}


def batch_text(batch: list[dict]) -> str:
    return "".join(
        f"Article {idx}:\nTitle: {article['title']}\nDesc: {article.get('description', '')}\n\n"
        for idx, article in enumerate(batch, 1)
    ).strip()


def extract_axis_keywords(axis: str, articles: list[dict], batch_size: int = 5) -> list[str]:
    print(f"\n--- [Step 2] Extracting {axis} Keywords via sLLM ---")
    lines = []

    for start in range(0, len(articles), batch_size):
        prompt = PROMPT_EXTRACTION.format(articles_text=batch_text(articles[start:start + batch_size]))
        lines.extend(
            f"{category} | {keyword}"
            for category, keyword in filter(None, map(parse_line, call_llm(prompt).splitlines()))
        )

    write_lines(path_for(axis, "raw"), lines)
    print(f"Extracted {len(lines)} raw {axis} keywords.")
    return lines


def extract_keywords(articles_by_axis: dict[str, list[dict]]) -> dict[str, list[str]]:
    return {axis: extract_axis_keywords(axis, articles_by_axis.get(axis, [])) for axis in QUERIES}


def valid_keyword(keyword: str, seen: set[str]) -> bool:
    normalized = keyword.lower()
    if normalized in seen or len(normalized) < 2 or not (1 <= len(normalized.split()) <= 4):
        return False
    return not any(stopword in normalized for stopword in STOPWORDS)


def exact_dedup(raw_path: Path) -> list[str]:
    seen, output = set(), []
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        parsed = parse_line(line)
        if not parsed:
            continue
        category, keyword = parsed
        if valid_keyword(keyword, seen):
            seen.add(keyword.lower())
            output.append(f"{category} | {keyword}")
    return output


def curate_axis(axis: str) -> list[str]:
    print(f"\n--- [Step 3] Curating {axis} Keywords ---")
    curated = exact_dedup(path_for(axis, "raw"))
    if not curated:
        print(f"No valid keywords found for {axis} after exact dedup.")
        write_lines(path_for(axis, "curated"), [])
        return []

    write_lines(path_for(axis, "curated"), curated)
    print(f"Total curated {axis} keywords after exact dedup: {len(curated)}")
    return curated


def curate_keywords() -> dict[str, list[str]]:
    return {axis: curate_axis(axis) for axis in QUERIES}


def main() -> None:
    articles_by_axis = collect_news()
    if any(articles_by_axis.values()):
        extract_keywords(articles_by_axis)
    curate_keywords()


if __name__ == "__main__":
    main()
