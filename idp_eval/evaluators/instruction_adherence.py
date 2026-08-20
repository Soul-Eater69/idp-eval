"""One-call holistic instruction-adherence evaluator.

The evaluator sends only the complete rendered ``instructions`` and ``output``
to one structured judge call. The judge identifies materially distinct,
checkable instructions and classifies each as followed or violated. Python
validates and deduplicates the response, assigns stable IDs, and calculates the
normalized score.
"""

from __future__ import annotations

import asyncio
import time

from idp_eval import tracing
from idp_eval.models import EvaluationCase, EvaluationResult, Evaluator
from idp_eval.prompts.instruction_adherence import (
    INSTRUCTION_ADHERENCE_SCHEMA_COMPACT,
    INSTRUCTION_ADHERENCE_SCHEMA_VERBOSE,
    render_instruction_adherence_prompt,
)
from idp_eval.rendering import is_empty_value, render_value
from idp_eval.scoring import (
    calculate_instruction_adherence,
    instruction_adherence_label,
)


def _normalize_instruction(text: str) -> str:
    """Normalizes instruction text for exact-match deduplication."""
    return " ".join(text.lower().split())


def _elapsed_ms(started: float) -> float:
    return (time.monotonic() - started) * 1000.0


class InstructionAdherenceEvaluator(Evaluator):
    """Measures adherence to explicit instructions with one judge call.

    Args:
        llm: Judge exposing ``generate_object(prompt, schema) -> dict``.
        verbose: Include the item-level audit trail in ``result.details``.
            Compact mode returns counts and timing only. Scoring is identical.
    """

    name = "instruction_adherence"
    required_fields = ("instructions", "output")

    def __init__(self, llm, verbose: bool = False):
        self._llm = llm
        self._verbose = verbose

    def evaluate(self, case: EvaluationCase) -> EvaluationResult:
        """Evaluates one case with exactly one structured judge call."""
        self.validate_case(case)
        started = time.monotonic()
        prompt, schema = self._prompt_and_schema(case)
        with tracing.judge_span(
            "instruction_adherence.evaluate",
            {"idp_eval.metric": self.name, "idp_eval.stage": "evaluate"},
        ):
            response = self._llm.generate_object(prompt=prompt, schema=schema)
        return self._result_from_response(response, _elapsed_ms(started))

    def _prompt_and_schema(self, case: EvaluationCase) -> tuple[list[dict], dict]:
        context = (
            None if is_empty_value(case.context) else render_value(case.context)
        )
        prompt = render_instruction_adherence_prompt(
            instructions=render_value(case.instructions),
            output=render_value(case.output),
            context=context,
            verbose=self._verbose,
        )
        schema = (
            INSTRUCTION_ADHERENCE_SCHEMA_VERBOSE
            if self._verbose
            else INSTRUCTION_ADHERENCE_SCHEMA_COMPACT
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
        """Evaluates asynchronously through the native or bridged judge path."""
        self.validate_case(case)
        started = time.monotonic()
        prompt, schema = self._prompt_and_schema(case)
        async with judge_limiter:
            with tracing.judge_span(
                "instruction_adherence.evaluate",
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
        if not isinstance(response, dict):
            raise ValueError(
                "Malformed instruction-adherence response: expected an object."
            )
        if set(response) != {"instructions"}:
            raise ValueError(
                "Malformed instruction-adherence response: expected only an "
                "`instructions` list."
            )
        raw_items = response["instructions"]
        if not isinstance(raw_items, list):
            raise ValueError(
                "Malformed instruction-adherence response: `instructions` must "
                "be a list."
            )

        required_keys = (
            {"instruction", "status", "reason"}
            if self._verbose
            else {"instruction", "status"}
        )
        validated: list[dict] = []
        for index, item in enumerate(raw_items, start=1):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Malformed instruction item {index}: expected an object."
                )
            if set(item) != required_keys:
                raise ValueError(
                    f"Malformed instruction item {index}: expected exactly "
                    f"{sorted(required_keys)}."
                )
            instruction = item.get("instruction")
            status = item.get("status")
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError(
                    f"Malformed instruction item {index}: `instruction` must be "
                    "a non-empty string."
                )
            if status not in {"followed", "violated"}:
                raise ValueError(
                    f"Unknown instruction-adherence status: {status!r}"
                )

            reason = item.get("reason", "")
            if not isinstance(reason, str):
                raise ValueError(
                    f"Malformed instruction item {index}: `reason` must be a "
                    "string."
                )
            if self._verbose:
                if status == "followed" and reason != "":
                    raise ValueError(
                        f"Malformed instruction item {index}: followed items "
                        "must use an empty `reason`."
                    )
                if status == "violated" and not reason.strip():
                    raise ValueError(
                        f"Malformed instruction item {index}: violated items "
                        "must include a non-empty `reason`."
                    )

            validated.append(
                {
                    "instruction": " ".join(instruction.split()),
                    "status": status,
                    "reason": reason,
                }
            )
        return validated

    @staticmethod
    def _build_items(raw_items: list[dict]) -> list[dict]:
        """Deduplicates items and adds stable IDs and deterministic scores."""
        seen: set[str] = set()
        items: list[dict] = []
        for raw in raw_items:
            normalized = _normalize_instruction(raw["instruction"])
            if normalized in seen:
                continue
            seen.add(normalized)
            items.append(
                {
                    "id": f"I{len(items) + 1}",
                    "instruction": raw["instruction"],
                    "status": raw["status"],
                    "score": 1.0 if raw["status"] == "followed" else 0.0,
                    "reason": raw["reason"],
                }
            )
        return items

    def _base_details(self, items: list[dict], total_ms: float) -> dict:
        return {
            "instruction_count": len(items),
            "followed_count": sum(i["status"] == "followed" for i in items),
            "violated_count": sum(i["status"] == "violated" for i in items),
            "judge_call_count": 1,
            "total_ms": total_ms,
            "verbose": self._verbose,
        }

    def _set_trace_attributes(self, count: int, total_ms: float) -> None:
        tracing.set_current_span_attributes(
            {
                "instruction_adherence.instruction_count": count,
                "instruction_adherence.judge_call_count": 1,
                "instruction_adherence.verbose": self._verbose,
                "instruction_adherence.total_ms": total_ms,
            }
        )

    def _result(self, items: list[dict], total_ms: float) -> EvaluationResult:
        details = self._base_details(items, total_ms)
        if self._verbose:
            details["instructions"] = items
        self._set_trace_attributes(len(items), total_ms)

        score = calculate_instruction_adherence(items)
        return EvaluationResult(
            metric=self.name,
            score=score,
            label=instruction_adherence_label(score),
            explanation=(
                f"{details['followed_count']} of {len(items)} instructions were "
                f"followed; {details['violated_count']} were violated."
            ),
            details=details,
        )

    def _not_applicable(self, total_ms: float) -> EvaluationResult:
        details = self._base_details([], total_ms)
        if self._verbose:
            details["instructions"] = []
        self._set_trace_attributes(0, total_ms)
        return EvaluationResult(
            metric=self.name,
            score=None,
            label="not_applicable",
            explanation=(
                "No meaningful checkable instructions were identified in the "
                "supplied instructions."
            ),
            details=details,
        )
