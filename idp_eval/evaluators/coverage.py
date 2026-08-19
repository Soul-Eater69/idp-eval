"""One-call whole-source coverage evaluator.

``CoverageEvaluator`` asks how much materially important information from the
full source/context is represented in the generated output. One structured judge
call identifies source items and returns two binary judgments per item; Python
derives covered/partial/missing statuses and the final normalized score.

Coverage is recall-like: unsupported additions are intentionally out of scope and
belong to faithfulness.
"""

from __future__ import annotations

import asyncio
import time

from idp_eval import tracing
from idp_eval.models import EvaluationCase, EvaluationResult, Evaluator
from idp_eval.prompts.coverage import (
    COVERAGE_SCHEMA_COMPACT,
    COVERAGE_SCHEMA_VERBOSE,
    render_coverage_prompt,
)
from idp_eval.rendering import render_value
from idp_eval.scoring import (
    calculate_coverage,
    coverage_label,
    coverage_status_from_binary,
    coverage_status_score,
)


def _normalize(text: str) -> str:
    """Normalizes source-item text for exact-match deduplication."""
    return " ".join(text.lower().split())


def _elapsed_ms(started: float) -> float:
    return (time.monotonic() - started) * 1000.0


class CoverageEvaluator(Evaluator):
    """Measures whole-source coverage using one structured judge call.

    Args:
        llm: Judge exposing ``generate_object(prompt, schema) -> dict``.
        verbose: Include the full item-level audit trail in ``result.details``.
            Compact mode returns counts/timing only. Scoring is identical.
    """

    name = "coverage"
    required_fields = ("context", "output")

    def __init__(self, llm, verbose: bool = False):
        self._llm = llm
        self._verbose = verbose

    def evaluate(self, case: EvaluationCase) -> EvaluationResult:
        """Evaluates one case with exactly one judge call."""
        self.validate_case(case)
        started = time.monotonic()
        prompt, schema = self._prompt_and_schema(case)
        with tracing.judge_span(
            "coverage.evaluate",
            {"idp_eval.metric": self.name, "idp_eval.stage": "evaluate"},
        ):
            response = self._llm.generate_object(prompt=prompt, schema=schema)
        return self._result_from_response(response, _elapsed_ms(started))

    def _prompt_and_schema(self, case: EvaluationCase) -> tuple[list[dict], dict]:
        prompt = render_coverage_prompt(
            context=render_value(case.context),
            output=render_value(case.output),
            verbose=self._verbose,
        )
        schema = (
            COVERAGE_SCHEMA_VERBOSE if self._verbose else COVERAGE_SCHEMA_COMPACT
        )
        return prompt, schema

    def _result_from_response(
        self, response: object, total_ms: float
    ) -> EvaluationResult:
        items = self._build_items(self._validate_response(response))
        if not items:
            return self._not_applicable(total_ms)
        return self._result(items, total_ms)

    async def a_evaluate(
        self, case: EvaluationCase, *, judge_limiter: asyncio.Semaphore
    ) -> EvaluationResult:
        """Evaluates asynchronously using the judge's native async method."""
        self.validate_case(case)
        started = time.monotonic()
        prompt, schema = self._prompt_and_schema(case)
        async with judge_limiter:
            with tracing.judge_span(
                "coverage.evaluate",
                {"idp_eval.metric": self.name, "idp_eval.stage": "evaluate"},
            ):
                async_generate = getattr(
                    self._llm, "async_generate_object", None
                )
                if callable(async_generate):
                    response = await async_generate(prompt=prompt, schema=schema)
                else:
                    response = await asyncio.to_thread(
                        self._llm.generate_object,
                        prompt=prompt,
                        schema=schema,
                    )
        return self._result_from_response(response, _elapsed_ms(started))

    def _validate_response(self, response: object) -> list[dict]:
        if not isinstance(response, dict) or "items" not in response:
            raise ValueError("Malformed coverage response: missing `items`.")
        raw_items = response["items"]
        if not isinstance(raw_items, list):
            raise ValueError("Malformed coverage response: `items` must be a list.")

        validated: list[dict] = []
        for index, item in enumerate(raw_items, start=1):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Malformed coverage item {index}: expected an object."
                )
            source_item = item.get("source_item")
            meaningfully_present = item.get("meaningfully_present")
            fully_present = item.get("fully_present")
            if not isinstance(source_item, str) or not source_item.strip():
                raise ValueError(
                    f"Malformed coverage item {index}: `source_item` must be a "
                    "non-empty string."
                )
            if not isinstance(meaningfully_present, bool) or not isinstance(
                fully_present, bool
            ):
                raise ValueError(
                    f"Malformed coverage item {index}: meaningfully_present/"
                    "fully_present must be booleans."
                )
            status = coverage_status_from_binary(
                meaningfully_present, fully_present
            )

            reason = item.get("reason", "")
            if not isinstance(reason, str):
                raise ValueError(
                    f"Malformed coverage item {index}: `reason` must be a string."
                )
            if self._verbose:
                if status == "covered" and reason != "":
                    raise ValueError(
                        f"Malformed coverage item {index}: covered items must use "
                        "an empty `reason`."
                    )
                if status != "covered" and not reason.strip():
                    raise ValueError(
                        f"Malformed coverage item {index}: partial/missing items "
                        "must include a non-empty `reason`."
                    )

            validated.append(
                {
                    "source_item": source_item,
                    "meaningfully_present": meaningfully_present,
                    "fully_present": fully_present,
                    "status": status,
                    "reason": reason,
                }
            )
        return validated

    @staticmethod
    def _build_items(raw_items: list[dict]) -> list[dict]:
        """Adds stable IDs/scores and removes normalized-exact duplicates."""
        seen: set[str] = set()
        items: list[dict] = []
        for raw in raw_items:
            normalized = _normalize(raw["source_item"])
            if normalized in seen:
                continue
            seen.add(normalized)
            items.append(
                {
                    "id": f"S{len(items) + 1}",
                    "source_item": raw["source_item"],
                    "meaningfully_present": raw["meaningfully_present"],
                    "fully_present": raw["fully_present"],
                    "status": raw["status"],
                    "item_score": coverage_status_score(raw["status"]),
                    "reason": raw["reason"],
                }
            )
        return items

    def _base_details(self, items: list[dict], total_ms: float) -> dict:
        return {
            "final_item_count": len(items),
            "covered_count": sum(i["status"] == "covered" for i in items),
            "partial_count": sum(i["status"] == "partial" for i in items),
            "missing_count": sum(i["status"] == "missing" for i in items),
            "judge_call_count": 1,
            "total_ms": total_ms,
            "verbose": self._verbose,
        }

    def _result(self, items: list[dict], total_ms: float) -> EvaluationResult:
        details = self._base_details(items, total_ms)
        if self._verbose:
            details["items"] = items

        tracing.set_current_span_attributes(
            {
                "coverage.item_count": details["final_item_count"],
                "coverage.judge_call_count": 1,
                "coverage.verbose": self._verbose,
                "coverage.total_ms": total_ms,
            }
        )
        score = calculate_coverage(items)
        count = details["final_item_count"]
        covered = details["covered_count"]
        partial = details["partial_count"]
        missing = details["missing_count"]
        return EvaluationResult(
            metric=self.name,
            score=score,
            label=coverage_label(score),
            explanation=(
                f"{covered} of {count} source items are fully covered; "
                f"{partial} partial and {missing} missing."
            ),
            details=details,
        )

    def _not_applicable(self, total_ms: float) -> EvaluationResult:
        details = self._base_details([], total_ms)
        if self._verbose:
            details["items"] = []
        tracing.set_current_span_attributes(
            {
                "coverage.item_count": 0,
                "coverage.judge_call_count": 1,
                "coverage.verbose": self._verbose,
                "coverage.total_ms": total_ms,
            }
        )
        return EvaluationResult(
            metric=self.name,
            score=None,
            label="not_applicable",
            explanation=(
                "No important source items were identified in the supplied context."
            ),
            details=details,
        )
