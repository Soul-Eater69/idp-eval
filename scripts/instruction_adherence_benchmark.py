"""Development-only GT benchmark for two-stage instruction adherence.

The script isolates instruction extraction and fixed-GT classification, then
optionally repeats the production end-to-end evaluator. A separate semantic
diagnostic call compares each extraction with human GT without affecting scores.
No model call occurs during import or unit tests.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from idp_eval.evaluators.instruction_adherence import (
    InstructionAdherenceEvaluator,
)
from idp_eval.scoring import calculate_instruction_adherence
from scripts.coverage_benchmark_utils import (
    exact_set_summary,
    mean,
    numeric_stats,
)

DEFAULT_CASES_PATH = Path("instructions_gt.json")
DEFAULT_OUTPUT_DIR = Path("benchmark_results")
DEFAULT_MODEL = "gpt-4o-mini"
CONFIRM_THRESHOLD = 100
STATUSES = ("followed", "violated")


def estimate_calls(
    num_cases: int,
    runs: int,
    *,
    end_to_end: bool,
    diagnostics: bool,
) -> dict[str, int]:
    """Returns evaluator and diagnostic call counts before execution."""
    extraction = num_cases * runs
    classification = num_cases * runs
    end_to_end_calls = num_cases * runs * 2 if end_to_end else 0
    diagnostic = num_cases * runs if diagnostics else 0
    evaluator = extraction + classification + end_to_end_calls
    return {
        "extraction": extraction,
        "classification": classification,
        "end_to_end": end_to_end_calls,
        "evaluator": evaluator,
        "diagnostic": diagnostic,
        "total": evaluator + diagnostic,
    }


def instruction_text(value: str | list[str]) -> str:
    """Converts the fixture's string/list form to the model's text field."""
    if isinstance(value, str):
        return value
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError("instructions must be a string or list of strings.")
    return "\n".join(value)


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    """Loads and validates the instruction-adherence GT JSON array."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Instruction benchmark file must contain a JSON array.")
    required = (
        "case_id",
        "category",
        "instructions",
        "output",
        "gold_instructions",
        "gold_score",
    )
    for case in data:
        missing = [field for field in required if field not in case]
        if missing:
            raise ValueError(
                f"Instruction case {case.get('case_id', '?')!r} missing {missing}."
            )
        instruction_text(case["instructions"])
        gold = case["gold_instructions"]
        if not isinstance(gold, list) or not gold:
            raise ValueError(f"Case {case['case_id']!r} has no gold instructions.")
        ids = [item.get("id") for item in gold]
        if len(ids) != len(set(ids)) or any(not item for item in ids):
            raise ValueError(f"Case {case['case_id']!r} has invalid gold ids.")
        if any(item.get("status") not in STATUSES for item in gold):
            raise ValueError(f"Case {case['case_id']!r} has invalid gold status.")
    return data


_MATCH_SYSTEM = """\
You diagnose instruction extraction against human-reviewed GOLD INSTRUCTIONS.
The EXTRACTED INSTRUCTIONS must be grounded only in the supplied ORIGINAL
INSTRUCTIONS. Harmless semantic paraphrases count as matches. Different valid
decompositions are acceptable, including one extracted item representing several
closely related gold items or several extracted items representing one gold item.

For every gold instruction, report whether its full material instruction appears
somewhere in the extracted set and whether all material qualifiers are preserved.
List the extracted ids that represent it.

For every extracted instruction, report whether it is grounded in the original
instructions, list represented gold ids, and identify a semantic duplicate by id
or use an empty string. An instruction is grounded even when it is a redundant
restatement; duplication is reported separately. Do not treat paraphrasing as an
invention. Return every supplied gold id and extracted id exactly once.\
"""

_MATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "gold_instructions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "gold_id": {"type": "string"},
                    "represented": {"type": "boolean"},
                    "extracted_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "qualifier_preserved": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": [
                    "gold_id",
                    "represented",
                    "extracted_ids",
                    "qualifier_preserved",
                    "reason",
                ],
            },
        },
        "extracted_instructions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "grounded": {"type": "boolean"},
                    "gold_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "duplicate_of": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": [
                    "id",
                    "grounded",
                    "gold_ids",
                    "duplicate_of",
                    "reason",
                ],
            },
        },
    },
    "required": ["gold_instructions", "extracted_instructions"],
}


def diagnose_extraction(
    judge: Any,
    case: dict[str, Any],
    extracted: list[dict[str, str]],
) -> dict[str, Any]:
    """Runs and validates one benchmark-only semantic GT diagnostic."""
    gold = [
        {"id": item["id"], "instruction": item["instruction"]}
        for item in case["gold_instructions"]
    ]
    prompt = [
        {"role": "system", "content": _MATCH_SYSTEM},
        {
            "role": "user",
            "content": "\n\n".join(
                [
                    "[ORIGINAL INSTRUCTIONS]\n"
                    + instruction_text(case["instructions"]),
                    "[GOLD INSTRUCTIONS]\n"
                    + json.dumps(gold, ensure_ascii=False),
                    "[EXTRACTED INSTRUCTIONS]\n"
                    + json.dumps(extracted, ensure_ascii=False),
                ]
            ),
        },
    ]
    result = judge.generate_object(prompt=prompt, schema=_MATCH_SCHEMA)
    if not isinstance(result, dict):
        raise ValueError("GT diagnostic response must be an object.")
    gold_rows = result.get("gold_instructions")
    extracted_rows = result.get("extracted_instructions")
    if not isinstance(gold_rows, list) or not isinstance(extracted_rows, list):
        raise ValueError("GT diagnostic response lists are missing.")

    gold_ids = {item["id"] for item in gold}
    extracted_ids = {item["id"] for item in extracted}
    returned_gold = [item.get("gold_id") for item in gold_rows]
    returned_extracted = [item.get("id") for item in extracted_rows]
    if set(returned_gold) != gold_ids or len(returned_gold) != len(set(returned_gold)):
        raise ValueError("GT diagnostic returned invalid gold instruction ids.")
    if set(returned_extracted) != extracted_ids or len(returned_extracted) != len(
        set(returned_extracted)
    ):
        raise ValueError("GT diagnostic returned invalid extracted instruction ids.")
    for row in gold_rows:
        if any(item not in extracted_ids for item in row["extracted_ids"]):
            raise ValueError("GT diagnostic referenced an unknown extracted id.")
    for row in extracted_rows:
        if any(item not in gold_ids for item in row["gold_ids"]):
            raise ValueError("GT diagnostic referenced an unknown gold id.")
        duplicate = row["duplicate_of"]
        if duplicate and (duplicate not in extracted_ids or duplicate == row["id"]):
            raise ValueError("GT diagnostic returned invalid duplicate_of id.")
    return result


def diagnostic_metrics(diagnostic: dict[str, Any]) -> dict[str, Any]:
    """Computes transparent per-run extraction metrics from a diagnostic."""
    gold = diagnostic["gold_instructions"]
    extracted = diagnostic["extracted_instructions"]
    represented = [row["represented"] for row in gold]
    grounded = [row["grounded"] for row in extracted]
    qualifiers = [row["qualifier_preserved"] for row in gold]
    return {
        "recall": mean([float(value) for value in represented]),
        "precision": mean([float(value) for value in grounded]),
        "qualifier_preservation": mean([float(value) for value in qualifiers]),
        "duplicate_count": sum(bool(row["duplicate_of"]) for row in extracted),
        "invented_count": sum(not value for value in grounded),
        "missed_gold_ids": [
            row["gold_id"] for row in gold if not row["represented"]
        ],
        "invented_extracted_ids": [
            row["id"] for row in extracted if not row["grounded"]
        ],
    }


def summarize_extraction_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregates repeated extraction and semantic-diagnostic results."""
    texts = [
        [item["instruction"] for item in run["instructions"]] for run in runs
    ]
    metrics = [
        run["diagnostic_metrics"]
        for run in runs
        if run.get("diagnostic_metrics")
    ]
    return {
        "exact": exact_set_summary(texts),
        "precision": numeric_stats([item["precision"] for item in metrics]),
        "recall": numeric_stats([item["recall"] for item in metrics]),
        "qualifier_preservation": mean(
            [item["qualifier_preservation"] for item in metrics]
        ),
        "duplicate_count": sum(item["duplicate_count"] for item in metrics),
        "invented_count": sum(item["invented_count"] for item in metrics),
        "missed_gold_ids": sorted(
            {gold_id for item in metrics for gold_id in item["missed_gold_ids"]}
        ),
        "invented_extracted_ids": sorted(
            {
                extracted_id
                for item in metrics
                for extracted_id in item["invented_extracted_ids"]
            }
        ),
    }


def majority_status(statuses: list[str]) -> tuple[str | None, float | None]:
    """Returns deterministic majority status and agreement rate."""
    if not statuses:
        return None, None
    counts = Counter(statuses)
    majority = max(STATUSES, key=lambda status: counts[status])
    return majority, counts[majority] / len(statuses)


def macro_f1(gold: list[str], predicted: list[str]) -> float | None:
    """Returns macro F1 over statuses present in the human GT."""
    labels = sorted(set(gold))
    if not labels or len(gold) != len(predicted):
        return None
    scores: list[float] = []
    for label in labels:
        true_positive = sum(g == label and p == label for g, p in zip(gold, predicted))
        false_positive = sum(g != label and p == label for g, p in zip(gold, predicted))
        false_negative = sum(g == label and p != label for g, p in zip(gold, predicted))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return mean(scores)


def summarize_classification_runs(
    runs: list[dict[str, Any]],
    gold_instructions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarizes fixed-GT classification stability and accuracy."""
    per_instruction: list[dict[str, Any]] = []
    gold_statuses: list[str] = []
    majority_statuses: list[str] = []
    agreements: list[float] = []
    for gold in gold_instructions:
        statuses = [run["statuses"][gold["id"]] for run in runs]
        majority, agreement = majority_status(statuses)
        per_instruction.append(
            {
                "id": gold["id"],
                "instruction": gold["instruction"],
                "gold_status": gold["status"],
                "status_per_run": statuses,
                "majority_status": majority,
                "agreement": agreement,
                "majority_correct": majority == gold["status"],
            }
        )
        if majority is not None:
            gold_statuses.append(gold["status"])
            majority_statuses.append(majority)
        if agreement is not None:
            agreements.append(agreement)
    return {
        "per_instruction": per_instruction,
        "mean_agreement": mean(agreements),
        "min_agreement": min(agreements) if agreements else None,
        "majority_accuracy": mean(
            [
                float(gold == predicted)
                for gold, predicted in zip(gold_statuses, majority_statuses)
            ]
        ),
        "macro_f1": macro_f1(gold_statuses, majority_statuses),
    }


class CountingJudge:
    """Counts successful and failed real model calls without changing behavior."""

    def __init__(self, judge: Any):
        self._judge = judge
        self.successful = 0
        self.failed = 0

    def generate_object(self, *args, **kwargs):
        try:
            result = self._judge.generate_object(*args, **kwargs)
        except Exception:
            self.failed += 1
            raise
        self.successful += 1
        return result


def _build_openai_judge() -> tuple[CountingJudge, str]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set.")
    model = os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
    from phoenix.evals import LLM

    return CountingJudge(
        LLM(provider="openai", client="openai", model=model, api_key=api_key)
    ), model


@contextmanager
def _benchmark_span(
    enabled: bool,
    phase: str,
    case_id: str,
    run_number: int,
) -> Iterator[None]:
    """Adds compact labels around isolated benchmark calls when tracing."""
    if not enabled:
        yield
        return
    from opentelemetry import trace

    tracer = trace.get_tracer("idp_eval.instruction_benchmark")
    with tracer.start_as_current_span(
        f"instruction_benchmark.{phase}",
        attributes={
            "benchmark_case_id": case_id,
            "benchmark_phase": phase,
            "benchmark_run_number": run_number,
        },
    ):
        yield


def _flush_tracing(timeout_ms: int = 10000) -> bool:
    """Best-effort bounded flush for manual traced benchmark runs."""
    try:
        from opentelemetry import trace

        flush = getattr(trace.get_tracer_provider(), "force_flush", None)
        return bool(flush(timeout_millis=timeout_ms)) if callable(flush) else False
    except Exception:  # noqa: BLE001 - development-only best effort
        return False


def _record_failure(
    failures: list[dict[str, Any]],
    case_id: str,
    phase: str,
    run_number: int,
    exc: Exception,
) -> None:
    failures.append(
        {
            "case_id": case_id,
            "phase": phase,
            "run": run_number,
            "error": f"{type(exc).__name__}: {exc}",
        }
    )


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Runs the manual benchmark and returns a JSON-serializable report."""
    from idp_eval import EvaluationCase, EvaluationFramework, register_tracing

    cases = load_cases(args.cases_file)
    if args.cases is not None:
        cases = cases[: args.cases]
    diagnostics_enabled = not args.no_diagnostics
    plan = estimate_calls(
        len(cases),
        args.runs,
        end_to_end=args.end_to_end,
        diagnostics=diagnostics_enabled,
    )
    print(f"Planned evaluator calls: {plan['evaluator']} ({plan})")
    print(f"Planned diagnostic calls: {plan['diagnostic']}")
    if plan["total"] > CONFIRM_THRESHOLD and not args.yes:
        reply = input(f"This will make ~{plan['total']} model calls. Continue? [y/N] ")
        if reply.strip().lower() not in {"y", "yes"}:
            raise SystemExit("Aborted before making model calls.")

    if args.trace:
        register_tracing(project_name=args.project_name)
    judge, model = _build_openai_judge()
    evaluator = InstructionAdherenceEvaluator(judge)
    framework = EvaluationFramework(
        evaluators=[evaluator], output="phoenix" if args.trace else None
    )
    started_at = datetime.now(timezone.utc)
    failures: list[dict[str, Any]] = []
    case_reports: list[dict[str, Any]] = []

    for case in cases:
        eval_case = EvaluationCase(
            case_id=case["case_id"],
            input="",
            context="",
            output=case["output"],
            instructions=instruction_text(case["instructions"]),
        )
        extraction_runs: list[dict[str, Any]] = []
        for run_number in range(1, args.runs + 1):
            try:
                with _benchmark_span(
                    args.trace, "extraction", case["case_id"], run_number
                ):
                    extracted = evaluator._extract_instructions(eval_case.instructions)
            except Exception as exc:  # noqa: BLE001 - record and continue
                _record_failure(
                    failures, case["case_id"], "extraction", run_number, exc
                )
                continue

            run: dict[str, Any] = {
                "run": run_number,
                "instructions": extracted,
                "diagnostic": None,
                "diagnostic_metrics": None,
            }
            if diagnostics_enabled:
                try:
                    with _benchmark_span(
                        args.trace, "gt_match", case["case_id"], run_number
                    ):
                        diagnostic = diagnose_extraction(judge, case, extracted)
                    run["diagnostic"] = diagnostic
                    run["diagnostic_metrics"] = diagnostic_metrics(diagnostic)
                except Exception as exc:  # noqa: BLE001 - record and continue
                    _record_failure(
                        failures, case["case_id"], "gt_match", run_number, exc
                    )
            extraction_runs.append(run)

        fixed = [
            {"id": item["id"], "instruction": item["instruction"]}
            for item in case["gold_instructions"]
        ]
        classification_runs: list[dict[str, Any]] = []
        for run_number in range(1, args.runs + 1):
            try:
                with _benchmark_span(
                    args.trace, "classification", case["case_id"], run_number
                ):
                    judgments = evaluator._classify_instructions(
                        fixed, eval_case.output
                    )
                items = evaluator._build_items(fixed, judgments)
                classification_runs.append(
                    {
                        "run": run_number,
                        "statuses": {item["id"]: item["status"] for item in items},
                        "items": items,
                        "score": calculate_instruction_adherence(items),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                _record_failure(
                    failures, case["case_id"], "classification", run_number, exc
                )

        end_to_end_runs: list[dict[str, Any]] = []
        if args.end_to_end:
            dataset_name = case.get("dataset_name", args.dataset_name)
            for run_number in range(1, args.runs + 1):
                try:
                    result = framework.evaluate(
                        eval_case,
                        run_name=f"instruction-run-{run_number:02d}",
                        dataset_name=dataset_name,
                    )["instruction_adherence"]
                    end_to_end_runs.append(
                        {
                            "run": run_number,
                            "score": result.score,
                            "instruction_count": result.details["instruction_count"],
                            "items": result.details["instructions"],
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    _record_failure(
                        failures, case["case_id"], "end_to_end", run_number, exc
                    )

        classification_scores = [run["score"] for run in classification_runs]
        classification_summary = summarize_classification_runs(
            classification_runs, case["gold_instructions"]
        )
        classification_stats = numeric_stats(classification_scores)
        classification_stats["mae"] = mean(
            [abs(score - case["gold_score"]) for score in classification_scores]
        )
        end_to_end_scores = [
            run["score"] for run in end_to_end_runs if run["score"] is not None
        ]
        end_to_end_stats = numeric_stats(end_to_end_scores)
        end_to_end_stats["bias"] = (
            end_to_end_stats["mean"] - case["gold_score"]
            if end_to_end_stats["mean"] is not None
            else None
        )
        end_to_end_stats["mae"] = mean(
            [abs(score - case["gold_score"]) for score in end_to_end_scores]
        )
        case_reports.append(
            {
                "case_id": case["case_id"],
                "category": case["category"],
                "description": case.get("description"),
                "gold_score": case["gold_score"],
                "gold_instructions": case["gold_instructions"],
                "extraction": {
                    "runs": extraction_runs,
                    "summary": summarize_extraction_runs(extraction_runs),
                },
                "classification": {
                    "runs": classification_runs,
                    "summary": classification_summary,
                    "score_stats": classification_stats,
                },
                "end_to_end": {
                    "runs": end_to_end_runs,
                    "score_stats": end_to_end_stats,
                }
                if args.end_to_end
                else None,
            }
        )

    flush_succeeded = _flush_tracing() if args.trace else None
    finished_at = datetime.now(timezone.utc)
    return {
        "config": {
            "model": model,
            "cases": len(cases),
            "runs": args.runs,
            "end_to_end": args.end_to_end,
            "diagnostics": diagnostics_enabled,
            "trace": args.trace,
            "project_name": args.project_name if args.trace else None,
            "dataset_name": args.dataset_name,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "trace_force_flush_succeeded": flush_succeeded,
        },
        "calls": {
            "planned": plan,
            "successful": judge.successful,
            "failed": judge.failed,
        },
        "failures": failures,
        "cases": case_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Instruction-adherence GT stability benchmark."
    )
    parser.add_argument("--cases-file", default=str(DEFAULT_CASES_PATH))
    parser.add_argument("--cases", type=int)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--end-to-end", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument(
        "--project-name", default="instruction-adherence-gt-smoke"
    )
    parser.add_argument(
        "--dataset-name", default="instruction-adherence-gt-v1"
    )
    parser.add_argument("--no-diagnostics", action="store_true")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")

    report = run_benchmark(args)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = output_dir / f"instruction_adherence_benchmark_{timestamp}.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(
        f"Completed: {report['calls']['successful']} model calls, "
        f"{report['calls']['failed']} failed."
    )
    print(f"Result file: {path}")


if __name__ == "__main__":
    main()
