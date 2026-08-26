"""One-call claim-level detector for historical few-shot content leakage."""

from __future__ import annotations

import asyncio
import time
from typing import Literal

from idp_eval import tracing
from idp_eval.models import EvaluationCase, EvaluationResult, Evaluator
from idp_eval.prompts.few_shot_content_leakage import (
    FEW_SHOT_CONTENT_LEAKAGE_SCHEMA_NONE,
    FEW_SHOT_CONTENT_LEAKAGE_SCHEMA_OVERALL,
    FEW_SHOT_CONTENT_LEAKAGE_SCHEMA_PER_ITEM,
    render_few_shot_content_leakage_prompt,
)
from idp_eval.rendering import render_value
from idp_eval.scoring import (
    calculate_few_shot_content_leakage,
    classify_few_shot_source,
    few_shot_content_leakage_label,
    few_shot_item_leakage_score,
)


def _normalize_claim(text: str) -> str:
    return " ".join(text.lower().split())


def _elapsed_ms(started: float) -> float:
    return (time.monotonic() - started) * 1000.0


class FewShotContentLeakageEvaluator(Evaluator):
    """Measures output claims supported only by historical few-shot examples.

    Current ``context`` is authoritative. ``retrieved_documents`` are historical
    examples used only as non-authoritative comparison material.
    """

    name = "few_shot_content_leakage"
    required_fields = ("context", "retrieved_documents", "output")

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
            "contract_version": 1,
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
            "few_shot_content_leakage.evaluate",
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
                "few_shot_content_leakage.evaluate",
                {"idp_eval.metric": self.name, "idp_eval.stage": "evaluate"},
            ):
                async_generate = getattr(llm, "async_generate_object", None)
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
        prompt = render_few_shot_content_leakage_prompt(
            context=render_value(case.context),
            retrieved_documents=render_value(case.retrieved_documents),
            output=render_value(case.output),
            reason_mode=self._reason_mode,
            max_items=self._max_items,
        )
        schema = {
            "overall": FEW_SHOT_CONTENT_LEAKAGE_SCHEMA_OVERALL,
            "per_item": FEW_SHOT_CONTENT_LEAKAGE_SCHEMA_PER_ITEM,
            "none": FEW_SHOT_CONTENT_LEAKAGE_SCHEMA_NONE,
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
        prefix = "Malformed few-shot content leakage response"
        if not isinstance(response, dict):
            raise ValueError(f"{prefix}: expected an object.")
        expected_top = (
            {"claims"}
            if self._reason_mode == "none"
            else {"claims", "overall_reason"}
        )
        if set(response) != expected_top:
            raise ValueError(
                f"{prefix}: expected exactly {sorted(expected_top)}."
            )
        raw_claims = response["claims"]
        if not isinstance(raw_claims, list):
            raise ValueError(f"{prefix}: `claims` must be a list.")
        if self._max_items is not None and len(raw_claims) > self._max_items:
            raise ValueError(
                f"{prefix}: `claims` exceeds configured "
                f"max_items={self._max_items}."
            )

        overall_reason = response.get("overall_reason")
        if self._reason_mode == "none":
            overall_reason = None
        elif not isinstance(overall_reason, str) or not overall_reason.strip():
            raise ValueError(
                f"{prefix}: `overall_reason` must be a non-empty string."
            )

        base_keys = {"claim", "theme_supported", "example_supported"}
        expected_item_keys = (
            base_keys
            if self._reason_mode == "none"
            else base_keys | {"reason"}
        )
        validated: list[dict] = []
        for index, item in enumerate(raw_claims, start=1):
            item_prefix = f"Malformed few-shot content leakage claim {index}"
            if not isinstance(item, dict):
                raise ValueError(f"{item_prefix}: expected an object.")
            if set(item) != expected_item_keys:
                raise ValueError(
                    f"{item_prefix}: unexpected or missing fields for the "
                    "configured reason_mode."
                )
            claim = item.get("claim")
            if not isinstance(claim, str) or not claim.strip():
                raise ValueError(
                    f"{item_prefix}: `claim` must be a non-empty string."
                )
            theme_supported = item.get("theme_supported")
            example_supported = item.get("example_supported")
            if not isinstance(theme_supported, bool):
                raise ValueError(
                    f"{item_prefix}: `theme_supported` must be a boolean."
                )
            if not isinstance(example_supported, bool):
                raise ValueError(
                    f"{item_prefix}: `example_supported` must be a boolean."
                )

            reason = item.get("reason")
            if reason is not None and not isinstance(reason, str):
                raise ValueError(f"{item_prefix}: `reason` must be a string.")
            if self._reason_mode == "overall":
                if theme_supported and reason != "":
                    raise ValueError(
                        f"{item_prefix}: claims supported by current context "
                        "must use an empty `reason` in overall mode."
                    )
                if not theme_supported and (
                    reason is None or not reason.strip()
                ):
                    raise ValueError(
                        f"{item_prefix}: example-only/unsupported claims must "
                        "include a non-empty `reason`."
                    )
            elif self._reason_mode == "per_item" and (
                reason is None or not reason.strip()
            ):
                raise ValueError(
                    f"{item_prefix}: every claim must include a non-empty "
                    "`reason` in per_item mode."
                )

            value = {
                "claim": " ".join(claim.split()),
                "theme_supported": theme_supported,
                "example_supported": example_supported,
            }
            if self._reason_mode != "none":
                value["reason"] = reason or ""
            validated.append(value)
        return validated, overall_reason

    @staticmethod
    def _build_claims(raw_claims: list[dict]) -> list[dict]:
        seen: dict[str, tuple[bool, bool]] = {}
        claims: list[dict] = []
        for raw in raw_claims:
            normalized = _normalize_claim(raw["claim"])
            support = (raw["theme_supported"], raw["example_supported"])
            if normalized in seen:
                if seen[normalized] != support:
                    raise ValueError(
                        "Malformed few-shot content leakage response: duplicate "
                        "normalized claim has conflicting support booleans."
                    )
                continue
            seen[normalized] = support
            classification = classify_few_shot_source(*support)
            claim = {
                "id": f"FS{len(claims) + 1}",
                "claim": raw["claim"],
                "theme_supported": support[0],
                "example_supported": support[1],
                "classification": classification,
                "item_leakage_score": few_shot_item_leakage_score(
                    classification
                ),
            }
            if "reason" in raw:
                claim["reason"] = raw["reason"]
            claims.append(claim)
        return claims

    def _base_details(self, claims: list[dict], total_ms: float) -> dict:
        return {
            "claim_count": len(claims),
            "evaluated_claims": len(claims),
            "max_items": self._max_items,
            "theme_only_count": sum(
                c["classification"] == "theme_only" for c in claims
            ),
            "theme_and_examples_count": sum(
                c["classification"] == "theme_and_examples" for c in claims
            ),
            "example_only_count": sum(
                c["classification"] == "example_only" for c in claims
            ),
            "unsupported_count": sum(
                c["classification"] == "unsupported" for c in claims
            ),
            "judge_call_count": 1,
            "total_ms": total_ms,
            "verbose": self._verbose,
            "reason_mode": self._reason_mode,
        }

    def _set_trace_attributes(self, claims: list[dict], total_ms: float) -> None:
        tracing.set_current_span_attributes(
            {
                "few_shot_content_leakage.claim_count": len(claims),
                "few_shot_content_leakage.example_only_count": sum(
                    c["classification"] == "example_only" for c in claims
                ),
                "few_shot_content_leakage.judge_call_count": 1,
                "few_shot_content_leakage.verbose": self._verbose,
                "few_shot_content_leakage.reason_mode": self._reason_mode,
                "few_shot_content_leakage.total_ms": total_ms,
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
        self._set_trace_attributes(claims, total_ms)
        score = calculate_few_shot_content_leakage(claims)
        return EvaluationResult(
            metric=self.name,
            score=score,
            label=few_shot_content_leakage_label(score),
            explanation=overall_reason,
            details=details,
        )

    def _not_applicable(
        self, total_ms: float, overall_reason: str | None
    ) -> EvaluationResult:
        details = self._base_details([], total_ms)
        if self._verbose:
            details["claims"] = []
        self._set_trace_attributes([], total_ms)
        return EvaluationResult(
            metric=self.name,
            score=None,
            label="not_applicable",
            explanation=overall_reason,
            details=details,
        )
