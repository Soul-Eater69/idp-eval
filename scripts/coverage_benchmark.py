"""Development benchmark for the production two-stage CoverageEvaluator.

The script measures extraction stability, fixed-requirement classification
stability, and optional end-to-end score stability. It uses a direct OpenAI judge
only for manual development runs; production ``create_judge()`` is untouched.

No model call occurs during imports or unit tests.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from idp_eval.evaluators.coverage import CoverageEvaluator
from idp_eval.scoring import calculate_coverage
from scripts.coverage_benchmark_utils import (
    exact_set_summary,
    load_cases,
    mean,
    numeric_stats,
)

DEFAULT_CASES_PATH = Path("benchmarks/coverage_cases.json")
DEFAULT_OUTPUT_DIR = Path("benchmark_results")
DEFAULT_MODEL = "gpt-4o-mini"
CONFIRM_THRESHOLD = 200


def estimate_calls(num_cases: int, runs: int, end_to_end: bool) -> dict[str, int]:
    """Returns planned calls for extraction, classification, and end-to-end."""
    extraction = num_cases * runs
    classification = num_cases * runs
    end_to_end_calls = num_cases * runs * 2 if end_to_end else 0
    return {
        "extraction": extraction,
        "classification": classification,
        "end_to_end": end_to_end_calls,
        "total": extraction + classification + end_to_end_calls,
    }


def status_agreement(
    runs: list[dict[str, dict[str, Any]]], requirement_ids: list[str]
) -> dict[str, Any]:
    """Returns per-requirement and mean majority status agreement."""
    per_requirement: dict[str, dict[str, Any]] = {}
    agreements: list[float] = []
    for requirement_id in requirement_ids:
        statuses = [
            run[requirement_id]["status"]
            for run in runs
            if requirement_id in run
        ]
        agreement = (
            max(Counter(statuses).values()) / len(statuses) if statuses else None
        )
        if agreement is not None:
            agreements.append(agreement)
        per_requirement[requirement_id] = {
            "status_per_run": statuses,
            "agreement": agreement,
        }
    return {
        "per_requirement": per_requirement,
        "mean_agreement": mean(agreements),
        "min_agreement": min(agreements) if agreements else None,
    }


def _build_openai_judge() -> tuple[Any, str]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set.")
    model = os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
    from phoenix.evals import LLM

    return LLM(provider="openai", client="openai", model=model, api_key=api_key), model


def _flush_tracing(timeout_ms: int = 10000) -> None:
    """Best-effort bounded flush for manual traced benchmark runs."""
    try:
        from opentelemetry import trace

        flush = getattr(trace.get_tracer_provider(), "force_flush", None)
        if callable(flush):
            flush(timeout_millis=timeout_ms)
    except Exception:  # noqa: BLE001 - development-only best effort
        pass


def _fixed_requirements(case: dict[str, Any], extracted: list[dict]) -> list[dict]:
    """Uses optional benchmark-fixed requirements, else extraction run one."""
    fixed = case.get("fixed_requirements")
    if not fixed:
        return extracted
    return [
        {"id": f"r{index}", "requirement": requirement}
        for index, requirement in enumerate(fixed, start=1)
    ]


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Runs the manual benchmark and returns its JSON-serializable report."""
    from idp_eval import EvaluationCase, EvaluationFramework, register_tracing

    cases = load_cases(args.cases_file)
    if args.cases is not None:
        cases = cases[: args.cases]
    plan = estimate_calls(len(cases), args.runs, args.end_to_end)
    print(f"Planned model calls: {plan}")
    if plan["total"] > CONFIRM_THRESHOLD and not args.yes:
        reply = input(f"This will make ~{plan['total']} model calls. Continue? [y/N] ")
        if reply.strip().lower() not in {"y", "yes"}:
            raise SystemExit("Aborted before making model calls.")

    if args.trace:
        register_tracing(project_name=args.project_name)
    judge, model = _build_openai_judge()
    evaluator = CoverageEvaluator(judge)
    framework = EvaluationFramework(
        evaluators=[evaluator], output="phoenix" if args.trace else None
    )

    failures: list[dict[str, Any]] = []
    successful_calls = 0
    case_reports: list[dict[str, Any]] = []
    for case in cases:
        eval_case = EvaluationCase(
            case_id=case["case_id"],
            input=case["input"],
            context=case["context"],
            output=case["output"],
        )
        extraction_runs: list[list[dict]] = []
        for run_number in range(1, args.runs + 1):
            try:
                extraction_runs.append(evaluator._extract_requirements(eval_case))
                successful_calls += 1
            except Exception as exc:  # noqa: BLE001 - record and continue
                failures.append(
                    {
                        "case_id": case["case_id"],
                        "phase": "extraction",
                        "run": run_number,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        fixed = _fixed_requirements(
            case, extraction_runs[0] if extraction_runs else []
        )
        classification_runs: list[dict[str, dict[str, Any]]] = []
        classification_scores: list[float] = []
        for run_number in range(1, args.runs + 1):
            if not fixed:
                break
            try:
                items = evaluator._build_items(
                    fixed, evaluator._classify_requirements(eval_case, fixed)
                )
                classification_runs.append({item["id"]: item for item in items})
                classification_scores.append(calculate_coverage(items))
                successful_calls += 1
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {
                        "case_id": case["case_id"],
                        "phase": "classification",
                        "run": run_number,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        end_to_end_scores: list[float] = []
        end_to_end_counts: list[float] = []
        if args.end_to_end:
            for run_number in range(1, args.runs + 1):
                try:
                    result = framework.evaluate(
                        eval_case,
                        run_name=f"coverage-run-{run_number:02d}",
                        dataset_name=args.dataset_name,
                    )["coverage"]
                    successful_calls += 2 if result.details["total_requirements"] else 1
                    if result.score is not None:
                        end_to_end_scores.append(result.score)
                    end_to_end_counts.append(float(result.details["total_requirements"]))
                except Exception as exc:  # noqa: BLE001
                    failures.append(
                        {
                            "case_id": case["case_id"],
                            "phase": "end_to_end",
                            "run": run_number,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )

        extraction_text = [
            [item["requirement"] for item in run] for run in extraction_runs
        ]
        case_reports.append(
            {
                "case_id": case["case_id"],
                "category": case.get("category"),
                "extraction": {
                    "runs": extraction_text,
                    "summary": exact_set_summary(extraction_text),
                },
                "classification": {
                    "fixed_requirements": fixed,
                    "agreement": status_agreement(
                        classification_runs, [item["id"] for item in fixed]
                    ),
                    "score_per_run": classification_scores,
                    "score_stats": numeric_stats(classification_scores),
                },
                "end_to_end": {
                    "score_per_run": end_to_end_scores,
                    "score_stats": numeric_stats(end_to_end_scores),
                    "requirement_count_per_run": [
                        int(value) for value in end_to_end_counts
                    ],
                    "requirement_count_stats": numeric_stats(end_to_end_counts),
                }
                if args.end_to_end
                else None,
            }
        )

    if args.trace:
        _flush_tracing()
    return {
        "config": {
            "model": model,
            "cases": len(cases),
            "runs": args.runs,
            "end_to_end": args.end_to_end,
            "trace": args.trace,
            "project_name": args.project_name if args.trace else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "calls": {
            "planned": plan,
            "successful": successful_calls,
            "failed_operations": len(failures),
        },
        "failures": failures,
        "cases": case_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Production coverage benchmark.")
    parser.add_argument("--cases-file", default=str(DEFAULT_CASES_PATH))
    parser.add_argument("--cases", type=int)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--end-to-end", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--project-name", default="coverage-stability-openai")
    parser.add_argument("--dataset-name", default="coverage-benchmark")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")

    report = run_benchmark(args)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = output_dir / f"coverage_benchmark_{timestamp}.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(
        f"Completed: {report['calls']['successful']} model calls, "
        f"{report['calls']['failed_operations']} failed operations."
    )
    print(f"Result file: {path}")


if __name__ == "__main__":
    main()
