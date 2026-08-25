"""One-call claim-level faithfulness evaluator."""

from __future__ import annotations

import asyncio
import time
from typing import Literal

from idp_eval import tracing
from idp_eval.models import EvaluationCase, EvaluationResult, Evaluator
from idp_eval.prompts.faithfulness import (
    FAITHFULNESS_SCHEMA_NONE,
    FAITHFULNESS_SCHEMA_OVERALL,
    FAITHFULNESS_SCHEMA_PER_ITEM,
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
    """Measures support for factual output claims with one judge call.

    ``max_items`` is an optional positive upper bound, not a required count;
    fewer real claims remain fewer evaluated claims.
    """

    name = "faithfulness"
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
            "contract_version": 4,
            "verbose": self._verbose,
            "max_items": self._max_items,
            "reason_mode": self._reason_mode,
            "judge": self.judge_resume_signature(self._llm),
        }

    def evaluate(self, case: EvaluationCase) -> EvaluationResult:
        """Evaluates one case with exactly one structured judge call."""
        self.validate_case(case)
        llm = self._require_judge()
        started = time.monotonic()
        prompt, schema = self._prompt_and_schema(case)
        with tracing.judge_span(
            "faithfulness.evaluate",
            {"idp_eval.metric": self.name, "idp_eval.stage": "evaluate"},
        ):
            response = llm.generate_object(prompt=prompt, schema=schema)
        return self._result_from_response(response, _elapsed_ms(started))

    async def a_evaluate(
        self, case: EvaluationCase, *, judge_limiter: asyncio.Semaphore
    ) -> EvaluationResult:
        """Uses native async generation or the shared-limiter thread bridge."""
        self.validate_case(case)
        llm = self._require_judge()
        started = time.monotonic()
        prompt, schema = self._prompt_and_schema(case)
        async with judge_limiter:
            with tracing.judge_span(
                "faithfulness.evaluate",
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

    def _prompt_and_schema(self, case: EvaluationCase) -> tuple[list[dict], dict]:
        prompt = render_faithfulness_prompt(
            context=render_value(case.context),
            output=render_value(case.output),
            reason_mode=self._reason_mode,
            max_items=self._max_items,
        )
        schema = {
            "overall": FAITHFULNESS_SCHEMA_OVERALL,
            "per_item": FAITHFULNESS_SCHEMA_PER_ITEM,
            "none": FAITHFULNESS_SCHEMA_NONE,
        }[self._reason_mode]
        return prompt, schema

    def _result_from_response(
        self, response: object, total_ms: float
    ) -> EvaluationResult:
        raw_claims, overall_reason = self._validate_response(response)
        claims = self._build_claims(raw_claims)
        if not claims:
            return self._not_applicable(total_ms, overall_reason)
        return self._result(claims, total_ms, overall_reason)

    def _validate_response(
        self, response: object
    ) -> tuple[list[dict], str | None]:
        if not isinstance(response, dict):
            raise ValueError("Malformed faithfulness response: expected an object.")
        expected_top = (
            {"claims"}
            if self._reason_mode == "none"
            else {"claims", "overall_reason"}
        )
        if set(response) != expected_top:
            raise ValueError(
                "Malformed faithfulness response: expected exactly "
                f"{sorted(expected_top)}."
            )
        raw_claims = response["claims"]
        if not isinstance(raw_claims, list):
            raise ValueError(
                "Malformed faithfulness response: `claims` must be a list."
            )
        if self._max_items is not None and len(raw_claims) > self._max_items:
            raise ValueError(
                "Malformed faithfulness response: `claims` exceeds configured "
                f"max_items={self._max_items}."
            )

        overall_reason = response.get("overall_reason")
        if self._reason_mode == "none":
            overall_reason = None
        elif not isinstance(overall_reason, str) or not overall_reason.strip():
            raise ValueError(
                "Malformed faithfulness response: `overall_reason` must be a "
                "non-empty string."
            )

        base_keys = {"claim", "status"}
        validated = []
        for index, item in enumerate(raw_claims, start=1):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Malformed faithfulness claim {index}: expected an object."
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
                    f"Malformed faithfulness claim {index}: unexpected or "
                    "missing fields for the configured reason_mode."
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
            reason = item.get("reason")
            if reason is not None and not isinstance(reason, str):
                raise ValueError(
                    f"Malformed faithfulness claim {index}: `reason` must be a "
                    "string."
                )
            if self._reason_mode == "overall":
                if status == "supported" and reason not in (None, ""):
                    raise ValueError(
                        f"Malformed faithfulness claim {index}: supported claims "
                        "must use an omitted or empty `reason` in overall mode."
                    )
                if status == "unsupported" and (
                    reason is None or not reason.strip()
                ):
                    raise ValueError(
                        f"Malformed faithfulness claim {index}: unsupported "
                        "claims must include a non-empty `reason`."
                    )
            elif self._reason_mode == "per_item" and (
                reason is None or not reason.strip()
            ):
                raise ValueError(
                    f"Malformed faithfulness claim {index}: every claim must "
                    "include a non-empty `reason` in per_item mode."
                )
            value = {
                "claim": " ".join(claim.split()),
                "status": status,
            }
            if self._reason_mode != "none":
                value["reason"] = reason or ""
            validated.append(value)
        return validated, overall_reason

    @staticmethod
    def _build_claims(raw_claims: list[dict]) -> list[dict]:
        seen: dict[str, str] = {}
        claims = []
        for raw in raw_claims:
            normalized = _normalize_claim(raw["claim"])
            if normalized in seen:
                if seen[normalized] != raw["status"]:
                    raise ValueError(
                        "Malformed faithfulness response: duplicate normalized "
                        "claim has conflicting classifications."
                    )
                continue
            seen[normalized] = raw["status"]
            claim = {
                "id": f"F{len(claims) + 1}",
                "claim": raw["claim"],
                "status": raw["status"],
                "item_score": faithfulness_status_score(raw["status"]),
            }
            if "reason" in raw:
                claim["reason"] = raw["reason"]
            claims.append(claim)
        return claims

    def _base_details(self, claims: list[dict], total_ms: float) -> dict:
        return {
            "claim_count": len(claims),
            "max_items": self._max_items,
            "evaluated_claims": len(claims),
            "supported_count": sum(c["status"] == "supported" for c in claims),
            "unsupported_count": sum(
                c["status"] == "unsupported" for c in claims
            ),
            "judge_call_count": 1,
            "total_ms": total_ms,
            "verbose": self._verbose,
            "reason_mode": self._reason_mode,
        }

    def _set_trace_attributes(self, count: int, total_ms: float) -> None:
        tracing.set_current_span_attributes(
            {
                "faithfulness.claim_count": count,
                "faithfulness.judge_call_count": 1,
                "faithfulness.verbose": self._verbose,
                "faithfulness.reason_mode": self._reason_mode,
                "faithfulness.total_ms": total_ms,
            }
        )

    def _result(
        self,
        claims: list[dict],
        total_ms: float,
        overall_reason: str | None,
    ) -> EvaluationResult:
        details = self._base_details(claims, total_ms)
        if self._verbose:
            details["claims"] = claims
        self._set_trace_attributes(len(claims), total_ms)
        score = calculate_faithfulness(claims)
        return EvaluationResult(
            metric=self.name,
            score=score,
            label=faithfulness_label(score),
            explanation=overall_reason,
            details=details,
        )

    def _not_applicable(
        self, total_ms: float, overall_reason: str | None
    ) -> EvaluationResult:
        details = self._base_details([], total_ms)
        if self._verbose:
            details["claims"] = []
        self._set_trace_attributes(0, total_ms)
        return EvaluationResult(
            metric=self.name,
            score=None,
            label="not_applicable",
            explanation=overall_reason,
            details=details,
        )
