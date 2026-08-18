"""Fast, one-call whole-source coverage evaluator.

``SourceCoverageEvaluator`` (metric ``source_coverage``) measures how much of the
materially important information in the source/context is represented in the
output:

    context + output -> one itemized judge call -> Python scoring

The judge identifies important source items and classifies each item in the same
structured response. Python then derives ``covered`` / ``partial`` / ``missing``
scores (1.0 / 0.5 / 0.0) and averages them. The judge never emits a numeric
coverage score.

This one-call design is intentionally faster than task coverage, but its
denominator is not output-isolated: the judge can see the generated output while
identifying source items. Use ``TaskCoverageEvaluator`` when task scoping and a
fixed, output-isolated denominator are more important than minimum judge calls.
"""

from __future__ import annotations

import asyncio
import time

from idp_eval import tracing
from idp_eval.models import EvaluationCase, EvaluationResult, Evaluator
from idp_eval.prompts.source_coverage import (
    SOURCE_COVERAGE_SCHEMA_COMPACT,
    SOURCE_COVERAGE_SCHEMA_VERBOSE,
    render_source_coverage_prompt,
)
from idp_eval.rendering import render_value
from idp_eval.scoring import (
    calculate_coverage,
    coverage_label,
    coverage_status_from_binary,
    coverage_status_score,
)


def _normalize(text: str) -> str:
    """Normalizes source-item text for deterministic exact deduplication."""
    return " ".join(text.lower().split())


def _elapsed_ms(start: float) -> float:
    """Returns rounded elapsed milliseconds from a monotonic start time."""
    return round((time.monotonic() - start) * 1000, 1)


class SourceCoverageEvaluator(Evaluator):
    """One-call coverage of important information in the whole source.

    Direction: ``context + output -> itemized coverage judgment``. ``input`` is
    ignored. Higher score is better.
    """

    name = "source_coverage"
    required_fields = ("context", "output")
    annotator_kind = "LLM"

    def __init__(self, llm, *, verbose: bool = False):
        """Initializes the evaluator with one shared judge.

        Args:
            llm: Judge exposing ``generate_object(prompt, schema) -> dict``.
            verbose: Include concise reasons for partial/missing items in the
                same judge response. It does not change scoring semantics.
        """
        self._llm = llm
        self._verbose = verbose

    def evaluate(self, case: EvaluationCase) -> EvaluationResult:
        """Evaluates one case with exactly one structured judge call."""
        self.validate_case(case)
        return self._evaluate_validated(case)

    async def a_evaluate(
        self, case: EvaluationCase, *, judge_limiter: asyncio.Semaphore
    ) -> EvaluationResult:
        """Runs the one judge call under the framework's global limiter."""
        self.validate_case(case)
        async with judge_limiter:
            return await asyncio.to_thread(self._evaluate_validated, case)

    def _evaluate_validated(self, case: EvaluationCase) -> EvaluationResult:
        started = time.monotonic()
        prompt = render_source_coverage_prompt(
            context=render_value(case.context),
            output=render_value(case.output),
            verbose=self._verbose,
        )
        schema = (
            SOURCE_COVERAGE_SCHEMA_VERBOSE
            if self._verbose
            else SOURCE_COVERAGE_SCHEMA_COMPACT
        )

        judge_started = time.monotonic()
        with tracing.judge_span(
            "source_coverage.evaluate",
            {"idp_eval.metric": self.name, "idp_eval.stage": "evaluate"},
        ):
            response = self._llm.generate_object(prompt=prompt, schema=schema)
        evaluate_ms = _elapsed_ms(judge_started)

        raw_items = self._validate_response(response)
        items = self._build_items(raw_items)
        total_ms = _elapsed_ms(started)
        result = self._build_result(
            items, evaluate_ms=evaluate_ms, total_ms=total_ms
        )
        self._set_trace_attributes(result)
        return result

    def _validate_response(self, response) -> list[dict]:
        """Validates the one-call response without treating malformed data as empty."""
        if not isinstance(response, dict) or "items" not in response:
            raise ValueError("Malformed source coverage response: missing `items`.")
        raw_items = response["items"]
        if not isinstance(raw_items, list):
            raise ValueError(
                "Malformed source coverage response: `items` must be a list."
            )

        validated: list[dict] = []
        for index, item in enumerate(raw_items, start=1):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Malformed source coverage item {index}: expected an object."
                )
            source_item = item.get("source_item")
            present = item.get("meaningfully_present")
            full = item.get("fully_present")
            if not isinstance(source_item, str) or not source_item.strip():
                raise ValueError(
                    f"Malformed source coverage item {index}: `source_item` must "
                    "be a non-empty string."
                )
            if not isinstance(present, bool) or not isinstance(full, bool):
                raise ValueError(
                    f"Malformed source coverage item {index}: "
                    "meaningfully_present/fully_present must be booleans."
                )
            coverage_status_from_binary(present, full)
            if self._verbose and "reason" not in item:
                raise ValueError(
                    f"Malformed source coverage item {index}: verbose output "
                    "requires `reason`."
                )
            reason = item.get("reason", "")
            if reason is None:
                reason = ""
            if not isinstance(reason, str):
                raise ValueError(
                    f"Malformed source coverage item {index}: `reason` must be "
                    "a string."
                )
            validated.append(
                {
                    "source_item": source_item,
                    "meaningfully_present": present,
                    "fully_present": full,
                    "reason": reason,
                }
            )
        return validated

    def _build_items(self, raw_items: list[dict]) -> list[dict]:
        """Deduplicates, assigns stable ids, and derives statuses/scores."""
        seen: set[str] = set()
        items: list[dict] = []
        for raw in raw_items:
            normalized = _normalize(raw["source_item"])
            if normalized in seen:
                continue
            seen.add(normalized)
            status = coverage_status_from_binary(
                raw["meaningfully_present"], raw["fully_present"]
            )
            items.append(
                {
                    "id": f"s{len(items) + 1}",
                    "source_item": raw["source_item"],
                    "meaningfully_present": raw["meaningfully_present"],
                    "fully_present": raw["fully_present"],
                    "status": status,
                    "score": coverage_status_score(status),
                    "reason": raw["reason"],
                }
            )
        return items

    def _build_result(
        self, items: list[dict], *, evaluate_ms: float, total_ms: float
    ) -> EvaluationResult:
        """Builds the deterministic aggregate result and diagnostics."""
        if not items:
            return EvaluationResult(
                metric=self.name,
                score=None,
                label="not_applicable",
                explanation=(
                    "No important source items were identified in the supplied context."
                ),
                details={
                    "total_items": 0,
                    "covered_count": 0,
                    "partial_count": 0,
                    "missing_count": 0,
                    "items": [],
                    "final_item_count": 0,
                    "judge_call_count": 1,
                    "verbose": self._verbose,
                    "evaluate_ms": evaluate_ms,
                    "total_ms": total_ms,
                },
            )

        covered = sum(item["status"] == "covered" for item in items)
        partial = sum(item["status"] == "partial" for item in items)
        missing = sum(item["status"] == "missing" for item in items)
        score = calculate_coverage(items)
        count = len(items)
        if covered == count:
            explanation = f"All {count} source items are fully represented."
        elif missing == count:
            explanation = f"None of the {count} source items are represented."
        else:
            explanation = (
                f"{covered} of {count} source items are fully covered; "
                f"{partial} partial and {missing} missing."
            )
        return EvaluationResult(
            metric=self.name,
            score=score,
            label=coverage_label(score),
            explanation=explanation,
            details={
                "total_items": count,
                "covered_count": covered,
                "partial_count": partial,
                "missing_count": missing,
                "items": items,
                "final_item_count": count,
                "judge_call_count": 1,
                "verbose": self._verbose,
                "evaluate_ms": evaluate_ms,
                "total_ms": total_ms,
            },
        )

    def _set_trace_attributes(self, result: EvaluationResult) -> None:
        """Adds compact one-call diagnostics to the active case span."""
        details = result.details or {}
        tracing.set_current_span_attributes(
            {
                "source_coverage.item_count": details.get("final_item_count", 0),
                "source_coverage.judge_call_count": 1,
                "source_coverage.verbose": self._verbose,
                "source_coverage.evaluate_ms": details.get("evaluate_ms", 0.0),
                "source_coverage.total_ms": details.get("total_ms", 0.0),
                "source_coverage.covered_count": details.get("covered_count", 0),
                "source_coverage.partial_count": details.get("partial_count", 0),
                "source_coverage.missing_count": details.get("missing_count", 0),
            }
        )
