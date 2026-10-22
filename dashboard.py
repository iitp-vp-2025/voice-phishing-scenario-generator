import ast
import io
from html import escape
import json
import re
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import streamlit as st

import main as end_to_end
from generation import scenario_generator as generator
from scenario_evaluation import evaluator


PROJECT_ROOT = Path(__file__).parent.resolve()
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
HUMAN_REVIEW_DIR = OUTPUTS_DIR / "human_review"
HUMAN_REVIEW_CSV = HUMAN_REVIEW_DIR / "human_review_results.csv"
HUMAN_REVIEW_JSONL = HUMAN_REVIEW_DIR / "human_review_results.jsonl"

AXES = ["R", "A", "B", "C", "D"]
SUBSCORES = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "C1", "C2", "D1", "D2"]
AXIS_LABELS = {
    "R": "보이스피싱 현실성",
    "A": "다양성",
    "B": "실행 가능성",
    "C": "시나리오 완결성",
    "D": "확장 가능성",
}
AXIS_MAX = {
    "R": 1,
    "A": 30,
    "B": 25,
    "C": 20,
    "D": 15,
}
SECTION_LABELS = [
    ("핵심 줄거리", "핵심 줄거리"),
    ("표적 자산", "표적 자산"),
    ("실현 가능성(Plausibility) 요소", "실현 가능성"),
    ("실현 가능성", "실현 가능성"),
    ("주요 공격 단계", "주요 공격 단계"),
    ("가장 유사한 기존 시나리오", "유사 기존 시나리오"),
]
SUBSCORE_LABELS = {
    "A1": "공격 흐름 차별성",
    "A2": "기술·상황 활용도",
    "A3": "설득 방식 차별성",
    "A4": "탐지·대응 사각지대",
    "B1": "자원/접근/기술 가능성",
    "B2": "피해자 설득 가능성",
    "B3": "기관/제도 사실성",
    "C1": "전체 논리 일관성",
    "C2": "시나리오/상호작용 완결성",
    "D1": "다른 맥락으로 확장",
    "D2": "맥락 변화 적응 가능성",
}
SUBSCORE_WEIGHTS = {
    "A1": 10, "A2": 10, "A3": 5, "A4": 5,
    "B1": 10, "B2": 10, "B3": 5,
    "C1": 10, "C2": 10,
    "D1": 10, "D2": 5,
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
        header[data-testid="stHeader"] { display: none; }
        #MainMenu, footer { visibility: hidden; }
        .block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1420px; }
        [data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none; }
        h1 {
            font-size: 1.65rem !important;
            line-height: 1.35 !important;
            margin-top: 0 !important;
            margin-bottom: .4rem !important;
        }
        h2 { font-size: 1.35rem !important; margin-top: .7rem !important; }
        div[data-testid="stMetric"] {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 8px 10px;
        }
        div[data-testid="stMetric"] label { color: #475569 !important; }
        div[data-testid="stMetricValue"] { font-size: 1.55rem !important; }
        .intro-box {
            border: 1px solid #dbe5f0;
            border-radius: 8px;
            padding: 11px 14px;
            background: #f8fbff;
            margin-bottom: .45rem;
            line-height: 1.5;
        }
        .scenario-panel {
            border: 1px solid #d9e1ec;
            border-radius: 8px;
            padding: 16px 18px;
            background: #ffffff;
            line-height: 1.62;
        }
        .note-panel {
            border: 1px solid #d9e1ec;
            border-radius: 8px;
            padding: 14px 16px;
            background: #fbfcfe;
        }
        .tiny-label {
            color: #64748b;
            font-size: .82rem;
            font-weight: 600;
            letter-spacing: .02em;
            text-transform: uppercase;
            margin-bottom: .2rem;
        }
        .muted-small { color: #64748b; font-size: 0.9rem; line-height: 1.45; }
        div[data-testid="stElementToolbar"] { display: none; }
        div[data-testid="stDataFrame"] { border-radius: 8px; }
        .review-summary {
            border: 1px solid #d9e1ec;
            border-radius: 8px;
            background: #fbfcfe;
            margin: .7rem 0 1rem 0;
            overflow: hidden;
        }
        .review-summary-row {
            display: grid;
            grid-template-columns: 145px minmax(0, 1fr);
            gap: 18px;
            padding: 12px 16px;
            border-top: 1px solid #e5edf5;
            align-items: start;
        }
        .review-summary-row:first-child { border-top: 0; }
        .review-summary-key {
            color: #64748b;
            font-size: .9rem;
            font-weight: 750;
            white-space: nowrap;
        }
        .review-summary-value {
            color: #1f2937;
            line-height: 1.55;
            word-break: keep-all;
            overflow-wrap: anywhere;
        }
        .review-summary-value strong { font-weight: 750; }
        @media (max-width: 760px) {
            .review-summary-row {
                grid-template-columns: 1fr;
                gap: 4px;
                padding: 11px 13px;
            }
            .review-summary-key { white-space: normal; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def parse_jsonish(value: Any) -> Any:
    text = safe_text(value).strip()
    if not text:
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except Exception:
            continue
    return text


def clean_title(value: Any) -> str:
    title = safe_text(value).strip()
    title = re.sub(r"^#{1,6}\s*", "", title)
    title = re.sub(r"^시나리오\s*\d+\s*[:：]\s*", "", title)
    return title.strip()


def clean_scenario_body(value: Any) -> str:
    text = safe_text(value).strip()
    if not text:
        return ""
    lines = text.splitlines()
    if lines and re.match(r"^\s*#{0,6}\s*시나리오\s*\d+\s*[:：]", lines[0]):
        lines = lines[1:]
    return "\n".join(lines).strip()


def parse_scenario_sections(value: Any) -> dict[str, str]:
    body = clean_scenario_body(value)
    sections: dict[str, list[str]] = {}
    current = "원문"

    for raw_line in body.splitlines():
        line = raw_line.rstrip()
        normalized = line.strip()
        if normalized.startswith("-"):
            normalized = normalized[1:].strip()
        normalized = normalized.replace("**", "").strip()

        matched = False
        for label, display in SECTION_LABELS:
            if normalized.startswith(label):
                rest = normalized[len(label):].strip()
                if rest.startswith(":"):
                    rest = rest[1:].strip()
                current = display
                sections.setdefault(current, [])
                if rest:
                    sections[current].append(rest)
                matched = True
                break

        if matched:
            continue
        sections.setdefault(current, []).append(line)

    return {
        key: "\n".join(part for part in parts if part.strip()).strip()
        for key, parts in sections.items()
        if any(part.strip() for part in parts)
    }


def scenario_title(text: Any) -> str:
    for line in safe_text(text).splitlines():
        line = line.strip()
        if line.startswith("#"):
            return clean_title(line)
    return ""


def preview_text(text: Any, limit: int = 220) -> str:
    collapsed = " ".join(safe_text(text).split())
    return collapsed if len(collapsed) <= limit else collapsed[:limit].rstrip() + "..."


def fmt_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return safe_text(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.1f}"


def discover_source_files() -> list[Path]:
    patterns = [
        "run_pipeline/final_evaluations.csv",
        "evaluate_scenario/dialogue_evaluations.csv",
        # Legacy output names remain readable.
        "evaluate_scenario/scenario_evaluations_rubric.csv",
        "evaluate_scenario/scenario_evaluations.csv",
        "improve_scenario/improved_evaluations.csv",
    ]
    files: list[Path] = []
    seen: set[Path] = set()
    for run_dir in (OUTPUTS_DIR / "end_to_end").glob("*"):
        for path in [
            run_dir / "dataset" / "dataset_sorted.csv",
            run_dir / "evaluate_scenario" / "final_evaluations.csv",
        ]:
            if path.is_file():
                files.append(path)
                seen.add(path.resolve())
                break
    for pattern in patterns:
        for path in OUTPUTS_DIR.glob(pattern):
            resolved = path.resolve()
            if path.is_file() and resolved not in seen:
                files.append(path)
                seen.add(resolved)
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def source_label(path: Path) -> str:
    text = display_path(path)
    demo_match = re.search(r"run_ppt_demo_(\d{8})_(\d{6})", text)
    if demo_match:
        date = demo_match.group(1)
        time = demo_match.group(2)
        return f"{date[:4]}.{date[4:6]}.{date[6:]} {time[:2]}:{time[2:4]} 발표 데모"
    match = re.search(r"run_(\d{8})_(\d{6})", text)
    if match:
        date = match.group(1)
        time = match.group(2)
        return f"{date[:4]}.{date[4:6]}.{date[6:]} {time[:2]}:{time[2:4]} 실행 결과"
    if path.name == "scenario_evaluations.csv":
        return "단일 평가 결과"
    return text


@st.cache_data(show_spinner=False)
def load_jsonl_companion(path_text: str) -> dict[str, dict[str, Any]]:
    path = Path(path_text)
    candidates = [path.with_suffix(".jsonl")]
    if path.name == "dataset_sorted.csv":
        candidates.append(path.with_name("dataset_sorted.jsonl"))
    if path.name == "dialogue_evaluations.csv":
        candidates.append(path.with_name("dialogue_evaluations.jsonl"))
    if path.name == "scenario_evaluations_rubric.csv":
        candidates.append(path.with_name("scenario_evaluations_rubric.jsonl"))
    if path.name == "scenario_evaluations.csv":
        candidates.append(path.with_name("scenario_evaluations.jsonl"))

    rows_by_id: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        if not candidate.exists():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            scenario_id = row.get("scenario_id")
            if scenario_id:
                rows_by_id[str(scenario_id)] = row
    return rows_by_id


def enrich_from_jsonl(df: pd.DataFrame, source_path: Path) -> pd.DataFrame:
    if "scenario_id" not in df.columns:
        return df

    df = df.copy()
    companion = load_jsonl_companion(str(source_path))
    if not companion:
        return df

    for idx, row in df.iterrows():
        source = companion.get(safe_text(row.get("scenario_id")))
        if not source:
            continue
        evaluation = source.get("evaluation", {})
        metadata = source.get("metadata", {})
        values = {
            "scenario": source.get("scenario", ""),
            "dialogue_json": json.dumps(source.get("dialogue", []), ensure_ascii=False),
            "evaluation_mode": evaluation.get("evaluation_mode", "full"),
            "ranking_score": evaluation.get("ranking_score", ""),
            "realism_prediction": evaluation.get("realism_prediction", ""),
            "realism_probability": evaluation.get("realism_probability", ""),
            "realism_confidence": evaluation.get("realism_confidence", ""),
            "kev_confidences": json.dumps(evaluation.get("kev_confidences", {}), ensure_ascii=False),
            "tech_keywords": ", ".join(metadata.get("tech_keywords", [])) if isinstance(metadata.get("tech_keywords"), list) else metadata.get("tech_keywords", ""),
            "social_keyword": metadata.get("social_keyword", ""),
            "selected_reason": metadata.get("selected_reason", ""),
            "e2e_index": metadata.get("e2e_index", ""),
            "weakest_kev_axis": evaluation.get("weakest_kev_axis", ""),
            "weakest_kev_subcriterion": evaluation.get("weakest_kev_subcriterion", ""),
            "refinement_status": metadata.get("refinement_status", ""),
            "refinement_target_axis": metadata.get("refinement_target_axis", ""),
            "refinement_target_subcriterion": metadata.get("refinement_target_subcriterion", ""),
            "changed_turns": json.dumps(metadata.get("changed_turns", []), ensure_ascii=False),
            "change_summary": metadata.get("change_summary", ""),
        }
        for col, value in values.items():
            if col not in df.columns:
                df[col] = ""
            if not safe_text(df.at[idx, col]) and safe_text(value):
                df.at[idx, col] = value
    return df


@st.cache_data(show_spinner=False)
def load_source(path_text: str) -> pd.DataFrame:
    path = Path(path_text)
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = enrich_from_jsonl(df, path)
    df["source_file"] = display_path(path)

    for col in ["rank", "e2e_index", "scenario_title", "scenario_preview", "scenario_id", "evaluation_mode", "ranking_score", "realism_prediction", "realism_probability", "realism_confidence", "kev_confidences", *AXES, *SUBSCORES, "weakest_kev_axis", "weakest_kev_subcriterion", "refinement_status", "refinement_target_axis", "refinement_target_subcriterion", "changed_turns", "change_summary", "turn_count", "dialogue_json", "summary", "scenario", "tech_keywords", "social_keyword", "selected_reason"]:
        if col not in df.columns:
            df[col] = ""

    for col in ["ranking_score", "realism_prediction", "realism_probability", "realism_confidence", *AXES, *SUBSCORES]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if not df["scenario_title"].astype(str).str.strip().any():
        df["scenario_title"] = df["scenario"].apply(scenario_title)
    df["scenario_title"] = df["scenario_title"].apply(clean_title)
    df["scenario_title"] = df["scenario_title"].where(df["scenario_title"].astype(str).str.strip() != "", df["scenario_id"])

    if not df["scenario_preview"].astype(str).str.strip().any():
        df["scenario_preview"] = df["scenario"].apply(preview_text)
    df["scenario_preview"] = df["scenario_preview"].where(df["scenario_preview"].astype(str).str.strip() != "", df["summary"].apply(preview_text))
    return df


def review_option_label(row: pd.Series) -> str:
    e2e = safe_text(row.get("e2e_index")) or safe_text(row.get("rank")) or "-"
    score = fmt_number(row.get("ranking_score")) or "-"
    title = clean_title(row.get("scenario_title")) or safe_text(row.get("scenario_id"))
    return f"{e2e}번 · {score}점 · {title[:90]}"


def show_review_queue(df: pd.DataFrame) -> Any:
    display = pd.DataFrame({
        "번호": df["e2e_index"].where(df["e2e_index"].astype(str).str.strip() != "", df["rank"]).astype(str),
        "시나리오 제목": df["scenario_title"].apply(clean_title),
        "랭킹 점수": pd.to_numeric(df["ranking_score"], errors="coerce"),
        "P(VP)": pd.to_numeric(df["realism_probability"], errors="coerce"),
        "부족한 KEV 항목": (
            df["weakest_kev_axis"].astype(str) + " / " + df["weakest_kev_subcriterion"].astype(str)
        ),
        "평가 요약": df["summary"].astype(str),
    }, index=df.index)
    event = st.dataframe(
        display,
        use_container_width=True,
        height=220,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "번호": st.column_config.TextColumn("번호", width="small"),
            "시나리오 제목": st.column_config.TextColumn("시나리오 제목", width="large"),
            "랭킹 점수": st.column_config.NumberColumn("랭킹 점수", format="%.1f", width="small"),
            "P(VP)": st.column_config.NumberColumn("P(VP)", format="%.3f", width="small"),
            "부족한 KEV 항목": st.column_config.TextColumn("부족한 KEV 항목", width="small"),
            "평가 요약": st.column_config.TextColumn("평가 요약", width="large"),
        },
    )
    selection = getattr(event, "selection", None)
    rows = getattr(selection, "rows", []) if selection else []
    return df.iloc[rows[0]].name if rows else None


def choose_scenario(df: pd.DataFrame) -> Any:
    st.markdown("**시나리오 선택**")
    return st.selectbox(
        "시나리오 선택",
        list(df.index),
        format_func=lambda idx: review_option_label(df.loc[idx]),
        label_visibility="collapsed",
    )


def split_axis(value: Any) -> list[str]:
    parts = []
    for piece in safe_text(value).replace("[", "").replace("]", "").replace("'", "").split(","):
        piece = piece.strip()
        if piece:
            parts.append(piece)
    return parts


def lowest_subcriteria(row: pd.Series) -> pd.DataFrame:
    rows = []
    confidence_data = parse_jsonish(row.get("kev_confidences", ""))
    confidences = confidence_data if isinstance(confidence_data, dict) else {}
    for key in SUBSCORES:
        value = row.get(key)
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        weight = SUBSCORE_WEIGHTS[key]
        rows.append({
            "기준": f"{key} {SUBSCORE_LABELS[key]}",
            "점수": f"{score:g}/{weight}",
            "충족도": f"{round(score / weight * 100)}%",
            "KEV confidence": (
                f"{float(confidences[key]) * 100:.1f}%"
                if confidences.get(key) is not None else "-"
            ),
        })
    return pd.DataFrame(sorted(rows, key=lambda item: int(item["충족도"].rstrip("%")))[:5])


def show_review_summary(row: pd.Series) -> None:
    weakest = split_axis(row.get("weakest_kev_axis"))

    axis_text = " · ".join(f"{axis}. {AXIS_LABELS.get(axis, axis)}" for axis in weakest) if weakest else "약점축 정보 없음"
    summary = safe_text(row.get("summary")) or "-"
    prediction = safe_text(row.get("realism_prediction"))
    evaluation_mode = safe_text(row.get("evaluation_mode")) or "full"
    if evaluation_mode == "kev_only" or not prediction:
        realism_text = "미사용 (checkpoint 없음)"
    else:
        realism_text = "보이스피싱(1)" if prediction in {"1.0", "1"} else "일반/비보이스피싱(0)"
    rows = [
        ("현실성 이진 판정", realism_text),
        ("랭킹 점수", f"{fmt_number(row.get('ranking_score')) or '-'} / 100"),
        ("가장 부족한 축", axis_text),
        ("가장 부족한 세부항목", safe_text(row.get("weakest_kev_subcriterion")) or "-"),
        ("Refinement", safe_text(row.get("refinement_status")) or "-"),
        ("축별 정규화", summary),
    ]
    probability = safe_text(row.get("realism_probability"))
    if probability:
        try:
            probability = f"{float(probability) * 100:.1f}%"
        except ValueError:
            pass
        rows.insert(1, ("R · P(VP)", probability))
    html = ["<div class='review-summary'>"]
    for label, value in rows:
        html.append(
            "<div class='review-summary-row'>"
            f"<div class='review-summary-key'>{escape(label)}</div>"
            f"<div class='review-summary-value'>{escape(value)}</div>"
            "</div>"
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def show_evaluation_detail(row: pd.Series) -> None:
    st.markdown("**검토 자료**")
    axis_tab, subcriteria_tab, refinement_tab, keyword_tab, raw_tab = st.tabs(
        ["축별 점수", "낮은 세부 기준", "Refinement", "생성 키워드", "원문"]
    )

    with axis_tab:
        axis_rows = []
        evaluation_mode = safe_text(row.get("evaluation_mode")) or "full"
        ranking_weights = (
            evaluator.KEV_ONLY_RANKING_WEIGHTS
            if evaluation_mode == "kev_only"
            else evaluator.RANKING_WEIGHTS
        )
        for axis in AXES:
            value = row.get(axis)
            maximum = AXIS_MAX[axis]
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if pd.isna(numeric_value):
                continue
            normalized = numeric_value if axis == "R" else numeric_value / maximum
            axis_rows.append({
                "축": f"{axis}. {AXIS_LABELS[axis]}",
                "원점수": f"{numeric_value:.3f}/{maximum}",
                "정규화": f"{normalized * 100:.1f}%",
                "랭킹 비중": f"{ranking_weights[axis] * 100:.2f}%",
            })
        st.dataframe(pd.DataFrame(axis_rows), use_container_width=True, hide_index=True)
        st.caption("KEV score API는 생성형 평가 근거 문장을 만들지 않고 점수·확률·confidence를 반환합니다.")

    low = lowest_subcriteria(row)
    with subcriteria_tab:
        if not low.empty:
            st.dataframe(low, use_container_width=True, hide_index=True)
        else:
            st.write("세부 기준 점수가 없습니다.")

    with refinement_tab:
        st.write(f"상태: {safe_text(row.get('refinement_status')) or '-'}")
        st.write(
            "수정 대상: "
            f"{safe_text(row.get('refinement_target_axis')) or '-'} / "
            f"{safe_text(row.get('refinement_target_subcriterion')) or '-'}"
        )
        st.write(f"수정 turn: {safe_text(row.get('changed_turns')) or '-'}")
        st.write(f"변경 요약: {safe_text(row.get('change_summary')) or '-'}")

    with keyword_tab:
        if safe_text(row.get("tech_keywords")):
            st.markdown(f"**기술 키워드:** {safe_text(row.get('tech_keywords'))}")
        if safe_text(row.get("social_keyword")):
            st.markdown(f"**사회 이슈 키워드:** {safe_text(row.get('social_keyword'))}")
        if safe_text(row.get("selected_reason")):
            st.markdown("**선택 이유**")
            st.write(safe_text(row.get("selected_reason")))
        if not any(safe_text(row.get(col)) for col in ["tech_keywords", "social_keyword", "selected_reason"]):
            st.write("생성 키워드 정보가 없습니다.")

    with raw_tab:
        st.code(safe_text(row.get("scenario")) or "원문이 없습니다.", language="markdown")


def show_scenario_sections(row: pd.Series) -> None:
    dialogue = parse_jsonish(row.get("dialogue_json", ""))
    if isinstance(dialogue, list) and dialogue:
        st.caption(f"총 {len(dialogue)}턴")
        with st.container(border=True):
            for item in dialogue:
                if not isinstance(item, dict):
                    continue
                turn = item.get("turn", "")
                speaker = safe_text(item.get("speaker"))
                text = safe_text(item.get("text"))
                st.markdown(f"**{turn}. {speaker}**  \n{text}")
        return

    sections = parse_scenario_sections(row.get("scenario"))
    order = ["핵심 줄거리", "표적 자산", "실현 가능성", "주요 공격 단계", "유사 기존 시나리오"]
    shown = False

    for label in order:
        content = sections.get(label, "")
        if not content:
            continue
        with st.container(border=True):
            st.markdown(f"**{label}**")
            st.markdown(content)
        shown = True

    leftovers = [key for key in sections if key not in order and key != "원문"]
    if leftovers:
        with st.expander("기타 내용", expanded=False):
            for key in leftovers:
                st.markdown(f"**{key}**")
                st.markdown(sections[key])
        shown = True

    if not shown:
        with st.container(border=True):
            st.markdown(clean_scenario_body(row.get("scenario")) or "_시나리오 본문이 없습니다._")


def save_human_review(row: pd.Series, decision: str, notes: str, reviewer: str, reviewed_at: str) -> None:
    HUMAN_REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "scenario_id": safe_text(row.get("scenario_id")),
        "e2e_index": safe_text(row.get("e2e_index")),
        "scenario_title": safe_text(row.get("scenario_title")),
        "ranking_score": safe_text(row.get("ranking_score")),
        "evaluation_mode": safe_text(row.get("evaluation_mode")) or "full",
        "realism_prediction": safe_text(row.get("realism_prediction")),
        "realism_probability": safe_text(row.get("realism_probability")),
        "weakest_kev_axis": safe_text(row.get("weakest_kev_axis")),
        "human_decision": decision,
        "human_notes": notes,
        "reviewer_name": reviewer,
        "reviewed_at": reviewed_at,
        "source_file": safe_text(row.get("source_file")),
        "scenario": safe_text(row.get("scenario")),
    }

    existing = pd.read_csv(HUMAN_REVIEW_CSV, encoding="utf-8-sig") if HUMAN_REVIEW_CSV.exists() else pd.DataFrame()
    pd.concat([existing, pd.DataFrame([record])], ignore_index=True).to_csv(HUMAN_REVIEW_CSV, index=False, encoding="utf-8-sig")
    with open(HUMAN_REVIEW_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def human_review_form(row: pd.Series) -> None:
    with st.container(border=True):
        st.markdown("**전문가 검토 기록**")
        with st.form("human_review_form"):
            decision = st.selectbox("판정", ["수정필요", "검토", "채택", "제외"])
            notes = st.text_area("부족한 점 / 수정 의견")
            reviewer = st.text_input("검토자")
            reviewed_at = st.text_input("검토 시각", value=datetime.now().isoformat(timespec="seconds"))
            submitted = st.form_submit_button("검토 결과 저장")
    if submitted:
        save_human_review(row, decision, notes, reviewer, reviewed_at)
        st.success(f"Saved review to {display_path(HUMAN_REVIEW_CSV)}")


def pipeline_page() -> None:
    st.title("보이스피싱 시나리오 파이프라인")
    st.markdown(
        """
        <div class="intro-box">
        생성은 기존 sLLM을 사용합니다. checkpoint가 있으면 현실성 R과 KEV A-D를 함께 평가하고,
        없으면 R을 제외한 KEV A-D 점수만으로 모든 시나리오를 랭킹합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("pipeline_form"):
        left, middle, right = st.columns(3)
        with left:
            st.markdown("**실행 범위**")
            count = st.number_input("시나리오 수", min_value=1, max_value=100, value=10)
            days = st.number_input("뉴스 검색 기간(일)", min_value=1, max_value=365, value=30)
            max_attempts = st.number_input("생성 최대 시도", min_value=int(count), max_value=500, value=max(15, int(count)))
            run_name = st.text_input("실행 이름(선택)", placeholder="비워두면 시간으로 자동 생성")
            reevaluate_run = st.text_input("기존 run 재평가(선택)", placeholder="run_YYYYMMDD_HHMMSS 또는 경로")
        with middle:
            st.markdown("**생성 모델**")
            generation_api_url = st.text_input("생성 API URL", value=generator.API_URL)
            generation_model = st.text_input("생성 모델", value=generator.MODEL_NAME)
            st.markdown("**KEV rubric 모델**")
            kev_api_url = st.text_input("KEV API URL", value=evaluator.KEV_API_URL)
            kev_model = st.text_input("KEV 모델", value=evaluator.KEV_MODEL_NAME)
        with right:
            st.markdown("**현실성 판정 모델**")
            distilbert_model = st.text_input(
                "DistilKoBERT checkpoint 경로(선택)",
                value=evaluator.DISTILBERT_MODEL_PATH,
                help="파일이 없거나 비워두면 KEV-only A/B/C/D 평가로 자동 실행합니다.",
            )
            distilbert_tokenizer = st.text_input(
                "Tokenizer 경로(선택)",
                value=evaluator.DISTILBERT_TOKENIZER_PATH,
                help="비워두면 monologg/distilkobert tokenizer를 사용합니다.",
            )
            distilbert_device = st.selectbox(
                "추론 장치",
                ["auto", "cpu", "mps", "cuda"],
                index=["auto", "cpu", "mps", "cuda"].index(evaluator.DISTILBERT_DEVICE)
                if evaluator.DISTILBERT_DEVICE in {"auto", "cpu", "mps", "cuda"}
                else 0,
            )

        with st.expander("고급 설정", expanded=False):
            st.caption(
                "references/에 둔 Tactic.pdf, Technique.pdf, Scenario.pdf와 "
                "positive_format_refs.jsonl은 존재할 때 자동으로 사용합니다."
            )
            seed = st.number_input("Seed", min_value=0, value=42)
        submitted = st.form_submit_button("파이프라인 실행", type="primary")

    if submitted:
        missing = []
        if not reevaluate_run and (not generation_api_url or not generation_model):
            missing.append("생성 API URL/모델")
        if not kev_api_url or not kev_model:
            missing.append("KEV API URL/모델")
        if missing:
            st.error("필수 설정이 없습니다: " + ", ".join(missing))
            return

        args = SimpleNamespace(
            count=int(count),
            days=int(days),
            max_attempts=int(max_attempts),
            output_root=end_to_end.PROJECT_ROOT / "outputs" / "end_to_end",
            run_name=run_name.strip(),
            reevaluate_run=reevaluate_run.strip(),
            generation_api_url=generation_api_url.strip(),
            generation_model=generation_model.strip(),
            kev_api_url=kev_api_url.strip(),
            kev_model=kev_model.strip(),
            distilbert_model=distilbert_model.strip(),
            distilbert_tokenizer=distilbert_tokenizer.strip(),
            distilbert_device=distilbert_device,
            distilbert_max_length=512,
            seed=int(seed),
            reference_chars=12000,
        )
        output = io.StringIO()
        try:
            with st.spinner("파이프라인을 실행하고 있습니다..."):
                with redirect_stdout(output):
                    paths = end_to_end.run_pipeline(args)
            load_source.clear()
            load_jsonl_companion.clear()
            st.success(f"완료: {display_path(paths['run'])}")
            st.code(output.getvalue(), language="text")
            st.caption(f"정렬 데이터: {display_path(paths['sorted_csv'])}")
        except (Exception, SystemExit) as exc:
            st.error(str(exc))
            if output.getvalue():
                st.code(output.getvalue(), language="text")


def review_page() -> None:
    st.title("보이스피싱 시나리오 검토")
    st.markdown(
        """
        <div class="intro-box">
        생성된 원본 시나리오를 읽고, 평가상 부족한 부분을 확인한 뒤 전문가 검토 의견을 남기는 화면입니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    files = discover_source_files()
    if not files:
        st.error("원본 평가 CSV를 찾지 못했습니다.")
        return

    labels = [display_path(path) for path in files]
    selected_label = st.selectbox(
        "평가 데이터",
        labels,
        format_func=lambda label: source_label(files[labels.index(label)]),
    )
    df = load_source(str(files[labels.index(selected_label)]))
    st.caption(f"총 {len(df)}개 시나리오")
    selected_index = choose_scenario(df)
    row = df.loc[selected_index]

    st.divider()
    title = clean_title(row.get("scenario_title")) or safe_text(row.get("scenario_id"))
    st.subheader(title)

    metric_cols = st.columns(6)
    metric_cols[0].metric("랭킹", f"{fmt_number(row.get('ranking_score')) or '-'}/100")
    realism = row.get("realism_probability")
    if safe_text(row.get("evaluation_mode")) == "kev_only":
        realism_text = "미사용"
    else:
        try:
            realism_text = f"{float(realism) * 100:.1f}%" if pd.notna(realism) else "-"
        except (TypeError, ValueError):
            realism_text = "-"
    metric_cols[1].metric("R · P(VP)", realism_text)
    metric_cols[2].metric("A · 다양성", f"{fmt_number(row.get('A')) or '-'}/30")
    metric_cols[3].metric("B · 실행 가능성", f"{fmt_number(row.get('B')) or '-'}/25")
    metric_cols[4].metric("C · 완결성", f"{fmt_number(row.get('C')) or '-'}/20")
    metric_cols[5].metric("D · 일반화", f"{fmt_number(row.get('D')) or '-'}/15")

    show_review_summary(row)

    st.markdown("**시나리오 내용**")
    show_scenario_sections(row)

    show_evaluation_detail(row)

    st.divider()
    human_review_form(row)


def main() -> None:
    st.set_page_config(page_title="Voice Phishing Scenario Pipeline", layout="wide")
    inject_css()
    page = st.radio(
        "화면",
        ["파이프라인 실행", "결과 검토"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if page == "파이프라인 실행":
        pipeline_page()
    else:
        review_page()


if __name__ == "__main__":
    main()
