import argparse
import sys
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import Generate_scenario as generator
import Evaluate_scenario as evaluator
import Improve_scenario as improver
from Evaluate_scenario import (
    PATHS as EVAL_PATHS,
    evaluate_records,
    load_evaluated_ids,
    load_optional_reference,
    scenario_id,
    write_results,
)

RUN_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "run_pipeline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="시나리오 생성 후 바로 평가까지 실행합니다.")
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
    parser.add_argument("--output-jsonl", type=Path, default=RUN_OUTPUT_DIR / "final_evaluations.jsonl")
    parser.add_argument("--output-csv", type=Path, default=RUN_OUTPUT_DIR / "final_evaluations.csv")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--reevaluate", action="store_true")
    parser.add_argument("--reference-chars", type=int, default=12000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    generator.API_URL = args.generation_api_url
    generator.MODEL_NAME = args.generation_model
    generator.GENERATION_REFERENCE_PATH = args.generation_reference
    generator.GENERATION_REFERENCE_COUNT = args.generation_reference_count

    reference_names = [
        path.name
        for key in ("tactic", "technique", "scenario")
        if (path := generator.PATHS[key]).is_file()
    ]
    print(f"[*] 생성 reference: {', '.join(reference_names) if reference_names else '없음'}")
    fine_tuned_judge = evaluator.build_fine_tuned_judge(
        model_path=args.distilbert_model,
        tokenizer_path=args.distilbert_tokenizer,
        device=args.distilbert_device,
        max_length=args.distilbert_max_length,
    )

    generated = generator.generate_scenario_pipeline()
    if not generated:
        raise SystemExit("[ERROR] 시나리오 생성에 실패했습니다. LLM 서버 설정을 확인하세요.")

    if not args.reevaluate and scenario_id(generated["scenario"]) in load_evaluated_ids(args.output_jsonl):
        print("[*] 방금 생성된 시나리오는 이미 평가되어 있어 skip합니다.")
        return

    reference = load_optional_reference(EVAL_PATHS["scenario_pdf"], args.reference_chars)

    try:
        rows = evaluate_records(
            [{
                "scenario": generated["scenario"],
                "dialogue": generated["dialogue"],
                "metadata": generated,
            }],
            fine_tuned_judge=fine_tuned_judge,
            source=str(generator.PATHS["generated_jsonl"]),
            reference=reference,
            api_url=args.kev_api_url,
            model=args.kev_model,
        )
    except requests.exceptions.RequestException as exc:
        raise SystemExit(f"[ERROR] 평가 LLM 요청 실패: {exc}") from exc

    initial_jsonl = args.output_jsonl.with_name("initial_evaluations.jsonl")
    initial_csv = args.output_csv.with_name("initial_evaluations.csv")
    write_results(rows, initial_jsonl, initial_csv, append=args.append or not args.reevaluate)

    rows, traces = improver.refine_rows(
        rows,
        fine_tuned_judge=fine_tuned_judge,
        reference=reference,
        kev_api_url=args.kev_api_url,
        kev_model=args.kev_model,
        generation_api_url=args.generation_api_url,
        generation_model=args.generation_model,
    )
    trace_dir = args.output_jsonl.parent / "refine_scenario"
    improver.write_trace(
        traces,
        trace_dir / "refinement_trace.jsonl",
        trace_dir / "refinement_trace.csv",
    )
    write_results(rows, args.output_jsonl, args.output_csv, append=args.append or not args.reevaluate)

    print("\n[*] 생성+평가 완료")
    print(f"- 최초 평가:       {initial_csv}")
    print(f"- Refinement trace: {trace_dir / 'refinement_trace.csv'}")
    print(f"- 최종 평가 JSONL: {args.output_jsonl}")
    print(f"- 최종 평가 CSV:   {args.output_csv}")


if __name__ == "__main__":
    main()
