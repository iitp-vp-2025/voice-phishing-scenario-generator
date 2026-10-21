import argparse
import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import Collect_keyword as collector
import Evaluate_scenario as evaluator
import Generate_scenario as generator
import Improve_scenario as improver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="뉴스 수집/생성/DistilKoBERT 현실성/KEV rubric/랭킹을 실행합니다.")
    parser.add_argument("--count", type=int, default=10, help="생성하고 평가할 시나리오 수")
    parser.add_argument("--days", type=int, default=30, help="News API 검색 기간")
    parser.add_argument("--max-attempts", type=int, default=15, help="시나리오 생성 최대 시도 횟수")
    parser.add_argument("--output-root", type=Path, default=SCRIPT_DIR / "outputs" / "end_to_end")
    parser.add_argument("--run-name", default="", help="지정하지 않으면 timestamp로 자동 생성")
    parser.add_argument(
        "--reevaluate-run",
        "--resume-run",
        dest="reevaluate_run",
        default="",
        help="기존 run 폴더의 generated_scenarios.jsonl을 다시 평가/정렬",
    )
    parser.add_argument("--generation-api-url", "--api-url", dest="generation_api_url", default=generator.API_URL)
    parser.add_argument("--generation-model", "--model", dest="generation_model", default=generator.MODEL_NAME)
    parser.add_argument("--generation-reference", default=generator.GENERATION_REFERENCE_PATH)
    parser.add_argument("--generation-reference-count", type=int, default=generator.GENERATION_REFERENCE_COUNT)
    parser.add_argument("--kev-api-url", default=evaluator.KEV_API_URL)
    parser.add_argument("--kev-model", default=evaluator.KEV_MODEL_NAME)
    parser.add_argument("--distilbert-model", default=evaluator.DISTILBERT_MODEL_PATH)
    parser.add_argument("--distilbert-tokenizer", default=evaluator.DISTILBERT_TOKENIZER_PATH)
    parser.add_argument("--distilbert-device", default=evaluator.DISTILBERT_DEVICE)
    parser.add_argument("--distilbert-max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference-chars", type=int, default=12000)
    return parser.parse_args()


def make_run_dir(output_root: Path, run_name: str) -> Path:
    run_id = run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    if run_dir.exists():
        raise SystemExit(f"[ERROR] 이미 존재하는 run 폴더입니다: {run_dir}")
    run_dir.mkdir(parents=True)
    return run_dir


def resolve_run_dir(output_root: Path, run_name: str) -> Path:
    raw = Path(run_name).expanduser()
    candidates = [raw] if raw.is_absolute() else [
        Path.cwd() / raw,
        SCRIPT_DIR / raw,
        output_root / raw,
    ]
    if not raw.is_absolute() and len(raw.parts) == 1 and not raw.name.startswith("run_"):
        candidates.append(output_root / f"run_{raw.name}")

    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    tried = "\n".join(f"- {path.resolve()}" for path in candidates)
    raise SystemExit(f"[ERROR] 재평가할 run 폴더를 찾지 못했습니다: {run_name}\n{tried}")


def configure_modules(
    run_dir: Path,
    generation_api_url: str,
    generation_model: str,
    generation_reference: str = "",
    generation_reference_count: int = 3,
) -> dict[str, Path]:
    collect_dir = run_dir / "collect_keyword"
    generate_dir = run_dir / "generate_scenario"
    evaluate_dir = run_dir / "evaluate_scenario"
    refine_dir = run_dir / "refine_scenario"
    dataset_dir = run_dir / "dataset"

    collector.OUTPUT_DIR = collect_dir
    collector.MODEL_URL = generation_api_url
    collector.MODEL_NAME = generation_model

    generator.COLLECT_OUTPUT_DIR = collect_dir
    generator.OUTPUT_DIR = generate_dir
    generator.API_URL = generation_api_url
    generator.MODEL_NAME = generation_model
    generator.GENERATION_REFERENCE_PATH = generation_reference
    generator.GENERATION_REFERENCE_COUNT = generation_reference_count
    generator.PATHS = {
        "tech_kw": collect_dir / "tech_keywords_curated.txt",
        "social_kw": collect_dir / "social_keywords_curated.txt",
        "tactic": SCRIPT_DIR / "Tactic.pdf",
        "technique": SCRIPT_DIR / "Technique.pdf",
        "scenario": SCRIPT_DIR / "Scenario.pdf",
        "generated_md": generate_dir / "generated_scenarios.md",
        "generated_jsonl": generate_dir / "generated_scenarios.jsonl",
    }

    return {
        "run": run_dir,
        "collect": collect_dir,
        "generate": generate_dir,
        "evaluate": evaluate_dir,
        "refine": refine_dir,
        "dataset": dataset_dir,
        "initial_eval_jsonl": evaluate_dir / "initial_evaluations.jsonl",
        "initial_eval_csv": evaluate_dir / "initial_evaluations.csv",
        "final_eval_jsonl": evaluate_dir / "final_evaluations.jsonl",
        "final_eval_csv": evaluate_dir / "final_evaluations.csv",
        "refinement_jsonl": refine_dir / "refinement_trace.jsonl",
        "refinement_csv": refine_dir / "refinement_trace.csv",
        "sorted_jsonl": dataset_dir / "dataset_sorted.jsonl",
        "sorted_csv": dataset_dir / "dataset_sorted.csv",
        "manifest": run_dir / "manifest.json",
    }


def run_collection(days: int) -> None:
    print("\n=== [1/5] News API 수집 및 키워드 정제 ===")
    articles_by_axis = collector.collect_news(days=days)
    if any(articles_by_axis.values()):
        collector.extract_keywords(articles_by_axis)
    collector.curate_keywords()


def generate_records(count: int, max_attempts: int) -> list[dict]:
    print(f"\n=== [2/5] {generator.DIALOGUE_TURNS}턴 대화 시나리오 {count}개 생성 ===")
    records = []
    for attempt in range(1, max_attempts + 1):
        if len(records) >= count:
            break
        print(f"\n--- [Generate] {len(records) + 1}/{count} (attempt {attempt}/{max_attempts}) ---")
        try:
            record = generator.generate_scenario_pipeline()
        except Exception as exc:
            print(f"[WARN] 시나리오 생성 실패: {exc}")
            continue
        if record:
            record["e2e_index"] = len(records) + 1
            records.append(record)

    if len(records) < count:
        raise SystemExit(f"[ERROR] 시나리오 {count}개 중 {len(records)}개만 생성했습니다.")
    return records


def load_generated_records(path: Path, count: int) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"[ERROR] 생성 결과 파일이 없습니다: {path}")

    records = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        dialogue = record.get("dialogue")
        if not isinstance(dialogue, list) or not dialogue:
            continue
        scenario = generator.dialogue_text(dialogue)
        record["dialogue_text"] = scenario
        record["scenario"] = scenario
        record.setdefault("e2e_index", line_no)
        record.setdefault("source_line", line_no)
        records.append(record)

    if count and len(records) > count:
        records = records[:count]
    if not records:
        raise SystemExit(f"[ERROR] 평가할 시나리오가 없습니다: {path}")
    return records


def evaluate_records(
    records: list[dict],
    paths: dict[str, Path],
    args: argparse.Namespace,
    fine_tuned_judge: object,
) -> list[dict]:
    print(f"\n=== [3/5] 생성 대화 최초 R/A/B/C/D 평가 ===")
    reference = evaluator.load_optional_reference(evaluator.PATHS["scenario_pdf"], args.reference_chars)
    rows = evaluator.evaluate_records(
        [
            {
                "scenario": record["scenario"],
                "dialogue": record["dialogue"],
                "metadata": record,
            }
            for record in records
        ],
        fine_tuned_judge=fine_tuned_judge,
        source=str(generator.PATHS["generated_jsonl"]),
        reference=reference,
        api_url=args.kev_api_url,
        model=args.kev_model,
    )
    evaluator.write_results(rows, paths["initial_eval_jsonl"], paths["initial_eval_csv"], append=False)
    return rows


def refine_records(
    rows: list[dict],
    paths: dict[str, Path],
    args: argparse.Namespace,
    fine_tuned_judge: object,
) -> list[dict]:
    print("\n=== [4/5] KEV 최약 세부항목 기반 필수 refinement 및 재평가 ===")
    reference = evaluator.load_optional_reference(evaluator.PATHS["scenario_pdf"], args.reference_chars)
    final_rows, traces = improver.refine_rows(
        rows,
        fine_tuned_judge=fine_tuned_judge,
        reference=reference,
        kev_api_url=args.kev_api_url,
        kev_model=args.kev_model,
        generation_api_url=args.generation_api_url,
        generation_model=args.generation_model,
    )
    improver.write_trace(traces, paths["refinement_jsonl"], paths["refinement_csv"])
    evaluator.write_results(final_rows, paths["final_eval_jsonl"], paths["final_eval_csv"], append=False)
    return final_rows


def sort_rows(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda row: -(row["evaluation"].get("ranking_score") or 0.0))


def write_sorted_dataset(rows: list[dict], paths: dict[str, Path]) -> None:
    print("\n=== [5/5] refinement 이후 데이터셋을 R/A/B/C/D 랭킹 점수로 정렬 ===")
    paths["dataset"].mkdir(parents=True, exist_ok=True)

    with open(paths["sorted_jsonl"], "w", encoding="utf-8") as f:
        for rank, row in enumerate(rows, 1):
            output = {"rank": rank, **row}
            f.write(json.dumps(output, ensure_ascii=False) + "\n")

    fields = [
        "rank",
        "e2e_index",
        "scenario_title",
        "scenario_preview",
        "scenario_id",
        "ranking_score",
        "realism_prediction",
        "realism_probability",
        "realism_confidence",
        "r_aggregation",
        "r_chunk_probabilities",
        "R",
        *evaluator.AXES.keys(),
        *evaluator.WEIGHTS.keys(),
        "weakest_kev_axis",
        "weakest_kev_subcriterion",
        "weakest_kev_axis_rate",
        "weakest_kev_subcriterion_score",
        "refinement_status",
        "refinement_target_axis",
        "refinement_target_subcriterion",
        "changed_turns",
        "change_summary",
        "turn_count",
        "summary",
        "tech_keywords",
        "social_keyword",
        "selected_reason",
        "scenario",
        "dialogue_json",
    ]
    with open(paths["sorted_csv"], "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows, 1):
            result = row["evaluation"]
            metadata = row.get("metadata", {})
            title, preview = evaluator.scenario_display(row["scenario"])
            title = metadata.get("title", title)
            normalized = result.get("normalized_scores", {})
            judgment = result.get("fine_tuned_judgment", {})
            writer.writerow({
                "rank": rank,
                "e2e_index": metadata.get("e2e_index", ""),
                "scenario_title": title,
                "scenario_preview": preview,
                "scenario_id": row["scenario_id"],
                "ranking_score": result.get("ranking_score", ""),
                "realism_prediction": result.get("realism_prediction", ""),
                "realism_probability": result.get("realism_probability", ""),
                "realism_confidence": result.get("realism_confidence", ""),
                "r_aggregation": judgment.get("aggregation", "single"),
                "r_chunk_probabilities": json.dumps(judgment.get("chunk_probabilities", [])),
                "R": normalized.get("R", ""),
                **result["axis_scores"],
                **result["subscores"],
                "weakest_kev_axis": result.get("weakest_kev_axis", ""),
                "weakest_kev_subcriterion": result.get("weakest_kev_subcriterion", ""),
                "weakest_kev_axis_rate": result.get("weakest_kev_axis_rate", ""),
                "weakest_kev_subcriterion_score": result.get("weakest_kev_subcriterion_score", ""),
                "refinement_status": metadata.get("refinement_status", ""),
                "refinement_target_axis": metadata.get(
                    "refinement_target_axis", result.get("weakest_kev_axis", "")
                ),
                "refinement_target_subcriterion": metadata.get(
                    "refinement_target_subcriterion", result.get("weakest_kev_subcriterion", "")
                ),
                "changed_turns": json.dumps(metadata.get("changed_turns", []), ensure_ascii=False),
                "change_summary": metadata.get("change_summary", ""),
                "turn_count": len(row.get("dialogue", [])),
                "summary": result["summary"],
                "tech_keywords": ", ".join(metadata.get("tech_keywords", [])),
                "social_keyword": metadata.get("social_keyword", ""),
                "selected_reason": metadata.get("selected_reason", ""),
                "scenario": row["scenario"],
                "dialogue_json": json.dumps(row.get("dialogue", []), ensure_ascii=False),
            })


def write_manifest(paths: dict[str, Path], args: argparse.Namespace, rows: list[dict]) -> None:
    counts_by_realism = {"normal": 0, "voice_phishing": 0}
    for row in rows:
        label = row["evaluation"].get("realism_label")
        if label in counts_by_realism:
            counts_by_realism[label] += 1

    generation_format_files = sorted({
        file_name
        for row in rows
        if (file_name := row.get("metadata", {}).get("format_reference_file", ""))
    })
    generation_format_count = max(
        (
            row.get("metadata", {}).get("format_reference_count", 0)
            for row in rows
        ),
        default=0,
    )

    manifest = {
        "created_at": datetime.now().isoformat(),
        "run_dir": str(paths["run"]),
        "count": len(rows),
        "days": args.days,
        "models": {
            "generation": args.generation_model,
            "rubric": args.kev_model,
            "realism": rows[0]["models"]["realism"],
        },
        "references": {
            "generation": sorted({
                reference
                for row in rows
                for reference in row.get("metadata", {}).get("reference_files", [])
            }),
            "evaluation": [evaluator.PATHS["scenario_pdf"].name]
            if evaluator.PATHS["scenario_pdf"].is_file()
            else [],
            "generation_format": {
                "file": generation_format_files[0] if generation_format_files else "",
                "count": generation_format_count,
            },
        },
        "seed": args.seed,
        "reevaluate_run": args.reevaluate_run,
        "ranking_weights": evaluator.RANKING_WEIGHTS,
        "dialogue": {
            "turns": generator.DIALOGUE_TURNS,
            "r_aggregation": "top_2_mean",
        },
        "refinement": {
            "target_axes": list(evaluator.AXES),
            "max_attempts": improver.MAX_REFINEMENT_ATTEMPTS,
            "max_changed_turns": improver.MAX_CHANGED_TURNS,
            "max_non_target_drop": improver.MAX_NON_TARGET_DROP,
        },
        "counts_by_realism": counts_by_realism,
        "outputs": {key: str(value) for key, value in paths.items() if key != "run"},
    }
    paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def run_pipeline(args: argparse.Namespace) -> dict[str, Path]:
    if args.count <= 0:
        raise SystemExit("[ERROR] --count는 1 이상이어야 합니다.")
    if args.max_attempts < args.count:
        raise SystemExit("[ERROR] --max-attempts는 --count보다 크거나 같아야 합니다.")
    if not args.kev_api_url or not args.kev_model:
        raise SystemExit("[ERROR] KEV rubric 평가를 위한 --kev-api-url과 --kev-model이 필요합니다.")
    if not args.distilbert_model:
        raise SystemExit("[ERROR] DistilKoBERT 평가에는 --distilbert-model이 필요합니다.")
    if not args.generation_api_url or not args.generation_model:
        raise SystemExit("[ERROR] 생성과 refinement를 위한 --generation-api-url과 --generation-model이 필요합니다.")

    random.seed(args.seed)
    reference_names = [
        name
        for name in ("Tactic.pdf", "Technique.pdf", "Scenario.pdf")
        if (SCRIPT_DIR / name).is_file()
    ]
    print(f"[*] 생성 reference: {', '.join(reference_names) if reference_names else '없음'}")
    fine_tuned_judge = evaluator.build_fine_tuned_judge(
        model_path=args.distilbert_model,
        tokenizer_path=args.distilbert_tokenizer,
        device=args.distilbert_device,
        max_length=args.distilbert_max_length,
    )
    if args.reevaluate_run:
        run_dir = resolve_run_dir(args.output_root, args.reevaluate_run)
    else:
        run_dir = make_run_dir(args.output_root, args.run_name)
    paths = configure_modules(
        run_dir,
        args.generation_api_url,
        args.generation_model,
        generation_reference=args.generation_reference,
        generation_reference_count=args.generation_reference_count,
    )

    if args.reevaluate_run:
        print(f"\n=== [Reevaluate] 기존 생성 결과 다시 평가: {paths['run']} ===")
        generated = load_generated_records(generator.PATHS["generated_jsonl"], args.count)
        print(f"[*] Loaded generated scenarios: {len(generated)}")
    else:
        run_collection(args.days)
        generated = generate_records(args.count, args.max_attempts)
    evaluated = evaluate_records(generated, paths, args, fine_tuned_judge)
    refined = refine_records(evaluated, paths, args, fine_tuned_judge)
    sorted_rows = sort_rows(refined)
    write_sorted_dataset(sorted_rows, paths)
    write_manifest(paths, args, sorted_rows)

    print("\n[*] End-to-end pipeline complete")
    print(f"- Run folder:      {paths['run']}")
    print(f"- Generated JSONL: {generator.PATHS['generated_jsonl']}")
    print(f"- Initial evaluation: {paths['initial_eval_csv']}")
    print(f"- Refinement trace:   {paths['refinement_csv']}")
    print(f"- Final evaluation:   {paths['final_eval_csv']}")
    print(f"- Sorted dataset: {paths['sorted_csv']}")
    return paths


def main() -> None:
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()
