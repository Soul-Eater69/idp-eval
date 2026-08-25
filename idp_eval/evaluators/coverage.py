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
from typing import Literal

from idp_eval import tracing
from idp_eval.models import EvaluationCase, EvaluationResult, Evaluator
from idp_eval.prompts.coverage import (
    COVERAGE_SCHEMA_NONE,
    COVERAGE_SCHEMA_OVERALL,
    COVERAGE_SCHEMA_PER_ITEM,
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
            This controls exposed detail only; ``reason_mode`` controls reason
            generation and the visible explanation.
        max_items: Optional positive maximum number of material source items.
            This is an upper bound, never a target or padding instruction.
        reason_mode: ``overall`` returns one semantic explanation plus internal
            reasons for failures; ``per_item`` requires reasons for every item;
            ``none`` returns no semantic reasons or explanation.
    """

    name = "coverage"
    required_fields = ("context", "output")

    def __init__(
        self,
        llm=None,
        verbose: bool = False,
        max_items: int | None = None,
        reason_mode: Literal["overall", "per_item", "none"] = "overall",
    ):
        if (
            isinstance(max_items, bool)
            or (max_items is not None and not isinstance(max_items, int))
            or (isinstance(max_items, int) and max_items < 1)
        ):
            raise ValueError(
                "max_items must be None or a positive integer, got "
                f"{max_items!r}."
            )
        if reason_mode not in {"overall", "per_item", "none"}:
            raise ValueError(
                "reason_mode must be one of 'overall', 'per_item', or 'none', "
                f"got {reason_mode!r}."
            )
        self._llm = llm
        self._verbose = verbose
        self._max_items = max_items
        self._reason_mode = reason_mode

    def resume_signature(self) -> dict:
        return {
            "contract_version": 3,
            "verbose": self._verbose,
            "max_items": self._max_items,
            "reason_mode": self._reason_mode,
            "judge": self.judge_resume_signature(self._llm),
        }

    def evaluate(self, case: EvaluationCase) -> EvaluationResult:
        """Evaluates one case with exactly one judge call."""
        self.validate_case(case)
        llm = self._require_judge()
        started = time.monotonic()
        prompt, schema = self._prompt_and_schema(case)
        with tracing.judge_span(
            "coverage.evaluate",
            {"idp_eval.metric": self.name, "idp_eval.stage": "evaluate"},
        ):
            response = llm.generate_object(prompt=prompt, schema=schema)
        return self._result_from_response(response, _elapsed_ms(started))

    def _prompt_and_schema(self, case: EvaluationCase) -> tuple[list[dict], dict]:
        prompt = render_coverage_prompt(
            context=render_value(case.context),
            output=render_value(case.output),
            reason_mode=self._reason_mode,
            max_items=self._max_items,
        )
        schema = {
            "overall": COVERAGE_SCHEMA_OVERALL,
            "per_item": COVERAGE_SCHEMA_PER_ITEM,
            "none": COVERAGE_SCHEMA_NONE,
        }[self._reason_mode]
        return prompt, schema

    def _result_from_response(
        self, response: object, total_ms: float
    ) -> EvaluationResult:
        raw_items, overall_reason = self._validate_response(response)
        items = self._build_items(raw_items)
        if not items:
            return self._not_applicable(total_ms, overall_reason)
        return self._result(items, total_ms, overall_reason)

    async def a_evaluate(
        self, case: EvaluationCase, *, judge_limiter: asyncio.Semaphore
    ) -> EvaluationResult:
        """Evaluates asynchronously using the judge's native async method."""
        self.validate_case(case)
        llm = self._require_judge()
        started = time.monotonic()
        prompt, schema = self._prompt_and_schema(case)
        async with judge_limiter:
            with tracing.judge_span(
                "coverage.evaluate",
                {"idp_eval.metric": self.name, "idp_eval.stage": "evaluate"},
            ):
                async_generate = getattr(
                    llm, "async_generate_object", None
                )
                if callable(async_generate):
                    response = await async_generate(prompt=prompt, schema=schema)
                else:
                    response = await asyncio.to_thread(
                        llm.generate_object,
                        prompt=prompt,
                        schema=schema,
                    )
        return self._result_from_response(response, _elapsed_ms(started))

    def _validate_response(
        self, response: object
    ) -> tuple[list[dict], str | None]:
        if not isinstance(response, dict):
            raise ValueError("Malformed coverage response: expected an object.")
        expected_top = (
            {"items"}
            if self._reason_mode == "none"
            else {"items", "overall_reason"}
        )
        if set(response) != expected_top:
            raise ValueError(
                "Malformed coverage response: expected exactly "
                f"{sorted(expected_top)}."
            )
        raw_items = response["items"]
        if not isinstance(raw_items, list):
            raise ValueError("Malformed coverage response: `items` must be a list.")
        if self._max_items is not None and len(raw_items) > self._max_items:
            raise ValueError(
                "Malformed coverage response: `items` exceeds configured "
                f"max_items={self._max_items}."
            )

        overall_reason = response.get("overall_reason")
        if self._reason_mode == "none":
            overall_reason = None
        elif not isinstance(overall_reason, str) or not overall_reason.strip():
            raise ValueError(
                "Malformed coverage response: `overall_reason` must be a "
                "non-empty string."
            )

        validated: list[dict] = []
        base_keys = {
            "source_item",
            "meaningfully_present",
            "fully_present",
        }
        for index, item in enumerate(raw_items, start=1):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Malformed coverage item {index}: expected an object."
                )
            item_keys = set(item)
            if self._reason_mode == "none":
                valid_keys = item_keys == base_keys
            elif self._reason_mode == "per_item":
                valid_keys = item_keys == base_keys | {"reason"}
            else:
                valid_keys = item_keys in (base_keys, base_keys | {"reason"})
            if not valid_keys:
                raise ValueError(
                    f"Malformed coverage item {index}: unexpected or missing "
                    "fields for the configured reason_mode."
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

            reason = item.get("reason")
            if reason is not None and not isinstance(reason, str):
                raise ValueError(
                    f"Malformed coverage item {index}: `reason` must be a string."
                )
            if self._reason_mode == "overall":
                if status == "covered" and reason not in (None, ""):
                    raise ValueError(
                        f"Malformed coverage item {index}: covered items must use "
                        "an omitted or empty `reason` in overall mode."
                    )
                if status != "covered" and (
                    reason is None or not reason.strip()
                ):
                    raise ValueError(
                        f"Malformed coverage item {index}: partial/missing items "
                        "must include a non-empty `reason`."
                    )
            elif self._reason_mode == "per_item" and (
                reason is None or not reason.strip()
            ):
                raise ValueError(
                    f"Malformed coverage item {index}: every item must include "
                    "a non-empty `reason` in per_item mode."
                )

            value = {
                "source_item": source_item,
                "meaningfully_present": meaningfully_present,
                "fully_present": fully_present,
                "status": status,
            }
            if self._reason_mode != "none":
                value["reason"] = reason or ""
            validated.append(value)
        return validated, overall_reason

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
            item = {
                "id": f"S{len(items) + 1}",
                "source_item": raw["source_item"],
                "meaningfully_present": raw["meaningfully_present"],
                "fully_present": raw["fully_present"],
                "status": raw["status"],
                "item_score": coverage_status_score(raw["status"]),
            }
            if "reason" in raw:
                item["reason"] = raw["reason"]
            items.append(item)
        return items

    def _base_details(self, items: list[dict], total_ms: float) -> dict:
        return {
            "final_item_count": len(items),
            "max_items": self._max_items,
            "evaluated_items": len(items),
            "covered_count": sum(i["status"] == "covered" for i in items),
            "partial_count": sum(i["status"] == "partial" for i in items),
            "missing_count": sum(i["status"] == "missing" for i in items),
            "judge_call_count": 1,
            "total_ms": total_ms,
            "verbose": self._verbose,
            "reason_mode": self._reason_mode,
        }

    def _result(
        self,
        items: list[dict],
        total_ms: float,
        overall_reason: str | None,
    ) -> EvaluationResult:
        details = self._base_details(items, total_ms)
        if self._verbose:
            details["items"] = items

        tracing.set_current_span_attributes(
            {
                "coverage.item_count": details["final_item_count"],
                "coverage.judge_call_count": 1,
                "coverage.verbose": self._verbose,
                "coverage.reason_mode": self._reason_mode,
                "coverage.total_ms": total_ms,
            }
        )
        score = calculate_coverage(items)
        return EvaluationResult(
            metric=self.name,
            score=score,
            label=coverage_label(score),
            explanation=overall_reason,
            details=details,
        )

    def _not_applicable(
        self, total_ms: float, overall_reason: str | None
    ) -> EvaluationResult:
        details = self._base_details([], total_ms)
        if self._verbose:
            details["items"] = []
        tracing.set_current_span_attributes(
            {
                "coverage.item_count": 0,
                "coverage.judge_call_count": 1,
                "coverage.verbose": self._verbose,
                "coverage.reason_mode": self._reason_mode,
                "coverage.total_ms": total_ms,
            }
        )
        return EvaluationResult(
            metric=self.name,
            score=None,
            label="not_applicable",
            explanation=overall_reason,
            details=details,
        )
