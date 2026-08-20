"""One-call claim-level faithfulness evaluator."""

from __future__ import annotations

import asyncio
import time

from idp_eval import tracing
from idp_eval.models import EvaluationCase, EvaluationResult, Evaluator
from idp_eval.prompts.faithfulness import (
    FAITHFULNESS_SCHEMA_COMPACT,
    FAITHFULNESS_SCHEMA_VERBOSE,
    render_faithfulness_prompt,
)
from idp_eval.rendering import render_value
from idp_eval.scoring import (
    calculate_faithfulness,
    faithfulness_label,
    faithfulness_status_score,
)


def _normalize_claim(text: str) -> str:
    return " ".join(text.lower().split())


def _elapsed_ms(started: float) -> float:
    return (time.monotonic() - started) * 1000.0


class FaithfulnessEvaluator(Evaluator):
    """Measures support for factual output claims with one judge call."""

    name = "faithfulness"
    required_fields = ("context", "output")

    def __init__(self, llm, verbose: bool = False):
        self._llm = llm
        self._verbose = verbose

    def evaluate(self, case: EvaluationCase) -> EvaluationResult:
        """Evaluates one case with exactly one structured judge call."""
        self.validate_case(case)
        started = time.monotonic()
        prompt, schema = self._prompt_and_schema(case)
        with tracing.judge_span(
            "faithfulness.evaluate",
            {"idp_eval.metric": self.name, "idp_eval.stage": "evaluate"},
        ):
            response = self._llm.generate_object(prompt=prompt, schema=schema)
        return self._result_from_response(response, _elapsed_ms(started))

    async def a_evaluate(
        self, case: EvaluationCase, *, judge_limiter: asyncio.Semaphore
    ) -> EvaluationResult:
        """Uses native async generation or the shared-limiter thread bridge."""
        self.validate_case(case)
        started = time.monotonic()
        prompt, schema = self._prompt_and_schema(case)
        async with judge_limiter:
            with tracing.judge_span(
                "faithfulness.evaluate",
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

    def _prompt_and_schema(self, case: EvaluationCase) -> tuple[list[dict], dict]:
        prompt = render_faithfulness_prompt(
            context=render_value(case.context),
            output=render_value(case.output),
            verbose=self._verbose,
        )
        schema = (
            FAITHFULNESS_SCHEMA_VERBOSE
            if self._verbose
            else FAITHFULNESS_SCHEMA_COMPACT
        )
        return prompt, schema

    def _result_from_response(
        self, response: object, total_ms: float
    ) -> EvaluationResult:
        claims = self._build_claims(self._validate_response(response))
        if not claims:
            return self._not_applicable(total_ms)
        return self._result(claims, total_ms)

    def _validate_response(self, response: object) -> list[dict]:
        if not isinstance(response, dict):
            raise ValueError("Malformed faithfulness response: expected an object.")
        if set(response) != {"claims"}:
            raise ValueError(
                "Malformed faithfulness response: expected only a `claims` list."
            )
        raw_claims = response["claims"]
        if not isinstance(raw_claims, list):
            raise ValueError(
                "Malformed faithfulness response: `claims` must be a list."
            )
        required_keys = (
            {"claim", "status", "reason"}
            if self._verbose
            else {"claim", "status"}
        )
        validated = []
        for index, item in enumerate(raw_claims, start=1):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Malformed faithfulness claim {index}: expected an object."
                )
            if set(item) != required_keys:
                raise ValueError(
                    f"Malformed faithfulness claim {index}: expected exactly "
                    f"{sorted(required_keys)}."
                )
            claim = item.get("claim")
            status = item.get("status")
            if not isinstance(claim, str) or not claim.strip():
                raise ValueError(
                    f"Malformed faithfulness claim {index}: `claim` must be a "
                    "non-empty string."
                )
            if status not in {"supported", "unsupported"}:
                raise ValueError(f"Unknown faithfulness status: {status!r}")
            reason = item.get("reason", "")
            if not isinstance(reason, str):
                raise ValueError(
                    f"Malformed faithfulness claim {index}: `reason` must be a "
                    "string."
                )
            if self._verbose and status == "unsupported" and not reason.strip():
                raise ValueError(
                    f"Malformed faithfulness claim {index}: unsupported claims "
                    "must include a non-empty `reason`."
                )
            validated.append(
                {
                    "claim": " ".join(claim.split()),
                    "status": status,
                    "reason": reason,
                }
            )
        return validated

    @staticmethod
    def _build_claims(raw_claims: list[dict]) -> list[dict]:
        seen: set[str] = set()
        claims = []
        for raw in raw_claims:
            normalized = _normalize_claim(raw["claim"])
            if normalized in seen:
                continue
            seen.add(normalized)
            claims.append(
                {
                    "id": f"F{len(claims) + 1}",
                    "claim": raw["claim"],
                    "status": raw["status"],
                    "item_score": faithfulness_status_score(raw["status"]),
                    "reason": raw["reason"],
                }
            )
        return claims

    def _base_details(self, claims: list[dict], total_ms: float) -> dict:
        return {
            "claim_count": len(claims),
            "supported_count": sum(c["status"] == "supported" for c in claims),
            "unsupported_count": sum(
                c["status"] == "unsupported" for c in claims
            ),
            "judge_call_count": 1,
            "total_ms": total_ms,
            "verbose": self._verbose,
        }

    def _set_trace_attributes(self, count: int, total_ms: float) -> None:
        tracing.set_current_span_attributes(
            {
                "faithfulness.claim_count": count,
                "faithfulness.judge_call_count": 1,
                "faithfulness.verbose": self._verbose,
                "faithfulness.total_ms": total_ms,
            }
        )

    def _result(self, claims: list[dict], total_ms: float) -> EvaluationResult:
        details = self._base_details(claims, total_ms)
        if self._verbose:
            details["claims"] = claims
        self._set_trace_attributes(len(claims), total_ms)
        score = calculate_faithfulness(claims)
        unsupported = details["unsupported_count"]
        unsupported_verb = "was" if unsupported == 1 else "were"
        return EvaluationResult(
            metric=self.name,
            score=score,
            label=faithfulness_label(score),
            explanation=(
                f"{details['supported_count']} of {len(claims)} factual claims "
                f"were supported; {unsupported} {unsupported_verb} unsupported."
            ),
            details=details,
        )

    def _not_applicable(self, total_ms: float) -> EvaluationResult:
        details = self._base_details([], total_ms)
        if self._verbose:
            details["claims"] = []
        self._set_trace_attributes(0, total_ms)
        return EvaluationResult(
            metric=self.name,
            score=None,
            label="not_applicable",
            explanation="No checkable factual claims were identified in the output.",
            details=details,
        )
