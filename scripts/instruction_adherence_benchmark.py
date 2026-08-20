"""Development-only GT benchmark for one-call instruction adherence.

The script repeats the holistic production evaluator and optionally compares
its returned instruction set with human-reviewed instructions through a separate
diagnostic call. Importing this module and running unit tests makes no model call.
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

from idp_eval.evaluators.instruction_adherence import InstructionAdherenceEvaluator
from scripts.coverage_benchmark_utils import exact_set_summary, mean, numeric_stats

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
    """Returns holistic evaluator and optional diagnostic call counts."""
    evaluation = num_cases * runs
    end_to_end_calls = num_cases * runs if end_to_end else 0
    diagnostic = num_cases * runs if diagnostics else 0
    evaluator = evaluation + end_to_end_calls
    return {
        "evaluation": evaluation,
        "end_to_end": end_to_end_calls,
        "evaluator": evaluator,
        "diagnostic": diagnostic,
        "total": evaluator + diagnostic,
    }


def instruction_text(value: str | list[str]) -> str:
    """Converts the fixture's string/list form to one rendered instruction field."""
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
You diagnose a one-call instruction-adherence result against human-reviewed GOLD
INSTRUCTIONS. Harmless semantic paraphrases and different valid decompositions
count as matches. For each gold instruction, report whether its full material
instruction and status appear in the evaluated set and whether qualifiers are
preserved. For each evaluated instruction, report whether it is grounded in the
original instructions and identify semantic duplicates. Return every supplied
gold id and evaluated id exactly once.\
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
                    "evaluated_ids": {"type": "array", "items": {"type": "string"}},
                    "qualifier_preserved": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": [
                    "gold_id",
                    "represented",
                    "evaluated_ids",
                    "qualifier_preserved",
                    "reason",
                ],
                "additionalProperties": False,
            },
        },
        "evaluated_instructions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "grounded": {"type": "boolean"},
                    "gold_ids": {"type": "array", "items": {"type": "string"}},
                    "duplicate_of": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["id", "grounded", "gold_ids", "duplicate_of", "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["gold_instructions", "evaluated_instructions"],
    "additionalProperties": False,
}


def diagnose_evaluation(
    judge: Any, case: dict[str, Any], evaluated: list[dict[str, Any]]
) -> dict[str, Any]:
    """Runs and validates one benchmark-only semantic GT diagnostic."""
    gold = [
        {
            "id": item["id"],
            "instruction": item["instruction"],
            "status": item["status"],
        }
        for item in case["gold_instructions"]
    ]
    compact = [
        {
            "id": item["id"],
            "instruction": item["instruction"],
            "status": item["status"],
        }
        for item in evaluated
    ]
    prompt = [
        {"role": "system", "content": _MATCH_SYSTEM},
        {
            "role": "user",
            "content": "\n\n".join(
                [
                    "[ORIGINAL INSTRUCTIONS]\n" + instruction_text(case["instructions"]),
                    "[GOLD INSTRUCTIONS]\n" + json.dumps(gold, ensure_ascii=False),
                    "[EVALUATED INSTRUCTIONS]\n"
                    + json.dumps(compact, ensure_ascii=False),
                ]
            ),
        },
    ]
    result = judge.generate_object(prompt=prompt, schema=_MATCH_SCHEMA)
    if not isinstance(result, dict):
        raise ValueError("GT diagnostic response must be an object.")
    gold_rows = result.get("gold_instructions")
    evaluated_rows = result.get("evaluated_instructions")
    if not isinstance(gold_rows, list) or not isinstance(evaluated_rows, list):
        raise ValueError("GT diagnostic response lists are missing.")

    gold_ids = {item["id"] for item in gold}
    evaluated_ids = {item["id"] for item in compact}
    returned_gold = [item.get("gold_id") for item in gold_rows]
    returned_evaluated = [item.get("id") for item in evaluated_rows]
    if set(returned_gold) != gold_ids or len(returned_gold) != len(set(returned_gold)):
        raise ValueError("GT diagnostic returned invalid gold instruction ids.")
    if set(returned_evaluated) != evaluated_ids or len(returned_evaluated) != len(
        set(returned_evaluated)
    ):
        raise ValueError("GT diagnostic returned invalid evaluated instruction ids.")
    for row in gold_rows:
        if any(item not in evaluated_ids for item in row["evaluated_ids"]):
            raise ValueError("GT diagnostic referenced an unknown evaluated id.")
    for row in evaluated_rows:
        if any(item not in gold_ids for item in row["gold_ids"]):
            raise ValueError("GT diagnostic referenced an unknown gold id.")
        duplicate = row["duplicate_of"]
        if duplicate and (duplicate not in evaluated_ids or duplicate == row["id"]):
            raise ValueError("GT diagnostic returned invalid duplicate_of id.")
    return result


def diagnostic_metrics(diagnostic: dict[str, Any]) -> dict[str, Any]:
    """Computes transparent semantic-set diagnostics."""
    gold = diagnostic["gold_instructions"]
    evaluated = diagnostic["evaluated_instructions"]
    return {
        "recall": mean([float(row["represented"]) for row in gold]),
        "precision": mean([float(row["grounded"]) for row in evaluated]),
        "qualifier_preservation": mean(
            [float(row["qualifier_preserved"]) for row in gold]
        ),
        "duplicate_count": sum(bool(row["duplicate_of"]) for row in evaluated),
        "invented_count": sum(not row["grounded"] for row in evaluated),
        "missed_gold_ids": [row["gold_id"] for row in gold if not row["represented"]],
        "invented_evaluated_ids": [row["id"] for row in evaluated if not row["grounded"]],
    }


def summarize_evaluation_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregates repeated one-call evaluation and diagnostic results."""
    texts = [
        [item["instruction"] for item in run["instructions"]] for run in runs
    ]
    metrics = [run["diagnostic_metrics"] for run in runs if run.get("diagnostic_metrics")]
    scores = [run["score"] for run in runs if run["score"] is not None]
    return {
        "score": numeric_stats(scores),
        "exact": exact_set_summary(texts),
        "precision": numeric_stats([item["precision"] for item in metrics]),
        "recall": numeric_stats([item["recall"] for item in metrics]),
        "qualifier_preservation": mean(
            [item["qualifier_preservation"] for item in metrics]
        ),
        "duplicate_count": sum(item["duplicate_count"] for item in metrics),
        "invented_count": sum(item["invented_count"] for item in metrics),
    }


def majority_status(statuses: list[str]) -> tuple[str | None, float | None]:
    """Returns deterministic majority status and agreement rate."""
    if not statuses:
        return None, None
    counts = Counter(statuses)
    majority = max(STATUSES, key=lambda status: counts[status])
    return majority, counts[majority] / len(statuses)


def macro_f1(gold: list[str], predicted: list[str]) -> float | None:
    """Returns macro F1 over statuses present in human GT."""
    labels = sorted(set(gold))
    if not labels or len(gold) != len(predicted):
        return None
    scores: list[float] = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, predicted))
        fp = sum(g != label and p == label for g, p in zip(gold, predicted))
        fn = sum(g == label and p != label for g, p in zip(gold, predicted))
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0.0)
    return mean(scores)


class CountingJudge:
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
    enabled: bool, phase: str, case_id: str, run_number: int
) -> Iterator[None]:
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
    evaluator = InstructionAdherenceEvaluator(judge, verbose=True)
    framework = EvaluationFramework(
        evaluators=[evaluator], output="phoenix" if args.trace else None
    )
    started_at = datetime.now(timezone.utc)
    failures: list[dict[str, Any]] = []
    case_reports: list[dict[str, Any]] = []

    for case in cases:
        eval_case = EvaluationCase(
            case_id=case["case_id"],
            output=case["output"],
            instructions=case["instructions"],
        )
        evaluation_runs: list[dict[str, Any]] = []
        for run_number in range(1, args.runs + 1):
            try:
                with _benchmark_span(
                    args.trace, "evaluation", case["case_id"], run_number
                ):
                    result = evaluator.evaluate(eval_case)
            except Exception as exc:  # noqa: BLE001
                _record_failure(failures, case["case_id"], "evaluation", run_number, exc)
                continue

            items = result.details.get("instructions", [])
            run: dict[str, Any] = {
                "run": run_number,
                "score": result.score,
                "instructions": items,
                "diagnostic": None,
                "diagnostic_metrics": None,
            }
            if diagnostics_enabled:
                try:
                    with _benchmark_span(
                        args.trace, "gt_match", case["case_id"], run_number
                    ):
                        diagnostic = diagnose_evaluation(judge, case, items)
                    run["diagnostic"] = diagnostic
                    run["diagnostic_metrics"] = diagnostic_metrics(diagnostic)
                except Exception as exc:  # noqa: BLE001
                    _record_failure(
                        failures, case["case_id"], "gt_match", run_number, exc
                    )
            evaluation_runs.append(run)

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
                            "items": result.details.get("instructions", []),
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    _record_failure(
                        failures, case["case_id"], "end_to_end", run_number, exc
                    )

        e2e_scores = [row["score"] for row in end_to_end_runs if row["score"] is not None]
        e2e_stats = numeric_stats(e2e_scores)
        e2e_stats["bias"] = (
            e2e_stats["mean"] - case["gold_score"]
            if e2e_stats["mean"] is not None
            else None
        )
        e2e_stats["mae"] = mean(
            [abs(score - case["gold_score"]) for score in e2e_scores]
        )
        case_reports.append(
            {
                "case_id": case["case_id"],
                "category": case["category"],
                "description": case.get("description"),
                "gold_score": case["gold_score"],
                "gold_instructions": case["gold_instructions"],
                "evaluation": {
                    "runs": evaluation_runs,
                    "summary": summarize_evaluation_runs(evaluation_runs),
                },
                "end_to_end": {
                    "runs": end_to_end_runs,
                    "score_stats": e2e_stats,
                }
                if args.end_to_end
                else None,
            }
        )

    flush_succeeded = _flush_tracing() if args.trace else None
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
            "finished_at": datetime.now(timezone.utc).isoformat(),
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
        description="One-call instruction-adherence GT stability benchmark."
    )
    parser.add_argument("--cases-file", default=str(DEFAULT_CASES_PATH))
    parser.add_argument("--cases", type=int)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--end-to-end", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--project-name", default="instruction-adherence-gt-smoke")
    parser.add_argument("--dataset-name", default="instruction-adherence-gt-v1")
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
