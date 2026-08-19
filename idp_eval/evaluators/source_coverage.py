"""Whole-source ``CoverageEvaluator`` (metric ``coverage``) with two modes.

Question: "How much materially important information from the source (context) is
represented in the output?" Direction: ``context + output``. No ``input`` required.

Modes (the algorithm choice lives in ``details["mode"]``, not the metric name):

- ``mode="dag"`` (default, recommended): output-isolated denominator. Stage 1 sees
  the CONTEXT only and extracts ~10 consolidated source items; Stage 2 classifies
  ALL of those fixed items against the OUTPUT in one call. ``judge_call_count`` is
  2 (1 extract + 1 classify); a very large item set (> 20) is split into safety
  batches only to reduce gateway-timeout risk.
- ``mode="g_eval"``: one call sees CONTEXT + OUTPUT and identifies + classifies the
  source items together. ``judge_call_count`` is 1. Lower latency / fewer calls,
  but the output is visible while identifying items.

Both modes share the source-item rubric and the deterministic covered/partial/
missing = 1.0/0.5/0.0 scoring; the judge never returns a numeric coverage score.

``SourceCoverageEvaluator`` is a deprecated thin alias kept for backward
compatibility (see bottom of this module).
"""

from __future__ import annotations

import asyncio
import time

from idp_eval import tracing
from idp_eval.evaluators.coverage_base import (
    _TwoStageCoverageEvaluator,
    _dedup,
    _ms,
    _normalize,
    _split_batches,
)
from idp_eval.models import EvaluationCase, EvaluationResult
from idp_eval.prompts.coverage import (
    COVERAGE_EXTRACT_SCHEMA,
    COVERAGE_GEVAL_SCHEMA_COMPACT,
    COVERAGE_GEVAL_SCHEMA_VERBOSE,
    render_coverage_extract_prompt,
    render_coverage_geval_prompt,
)
from idp_eval.rendering import render_value
from idp_eval.scoring import (
    calculate_coverage,
    coverage_label,
    coverage_status_from_binary,
    coverage_status_score,
)

# Semantic extraction target (prompt-side only; never truncated in Python).
DEFAULT_TARGET_ITEMS = 10
# Operational safety batching for unusually large extracted item sets (NOT the
# normal path): classify all items in one call up to the threshold, else split.
CLASSIFICATION_BATCH_THRESHOLD = 20
SAFETY_CLASSIFICATION_BATCH_SIZE = 10

_VALID_MODES = ("dag", "g_eval")


class CoverageEvaluator(_TwoStageCoverageEvaluator):
    """Whole-source coverage with a ``dag`` (default) or ``g_eval`` mode.

    Direction: ``context + output``. Higher is better.
    """

    name = "coverage"
    required_fields = ("context", "output")
    _item_key = "source_item"
    _total_key = "final_item_count"
    _id_prefix = "S"
    _explanation_unit = "source items"
    _not_applicable_reason = (
        "No important source items were identified in the supplied context."
    )

    def __init__(self, llm, *, mode: str = "dag", verbose: bool = False):
        """Args:
            llm: judge exposing ``generate_object(prompt, schema) -> dict``.
            mode: ``"dag"`` (default; output-isolated two-stage) or ``"g_eval"``
                (one-call, lower latency).
            verbose: include per-item reasons for partial/missing items and the
                detailed ``items`` array in ``details``. Scoring is unaffected.

        Raises:
            ValueError: If ``mode`` is not one of ``"dag"`` / ``"g_eval"``.
        """
        if mode not in _VALID_MODES:
            raise ValueError(
                f"mode must be one of {_VALID_MODES}, got {mode!r}."
            )
        self._mode = mode
        super().__init__(llm, verbose=verbose)

    # --- entry points (dispatch on mode) ------------------------------------

    def evaluate(self, case: EvaluationCase) -> EvaluationResult:
        self.validate_case(case)
        if self._mode == "g_eval":
            return self._evaluate_g_eval(case)
        return super().evaluate(case)  # DAG two-stage

    async def a_evaluate(
        self, case: EvaluationCase, *, judge_limiter: asyncio.Semaphore
    ) -> EvaluationResult:
        self.validate_case(case)
        if self._mode == "g_eval":
            async with judge_limiter:
                return await asyncio.to_thread(self._evaluate_g_eval, case)
        return await super().a_evaluate(case, judge_limiter=judge_limiter)

    # --- grouped shared extraction (DAG only) -------------------------------

    @property
    def uses_shared_extraction(self) -> bool:
        """DAG mode can extract one shared context's items once per group."""
        return self._mode == "dag"

    def extract_items(self, case: EvaluationCase) -> list[dict]:
        """Runs the DAG Stage-1 extraction once (for grouped reuse)."""
        return self._extract_requirements(case)

    async def a_extract_items(
        self, case: EvaluationCase, judge_limiter: asyncio.Semaphore
    ) -> list[dict]:
        """Async DAG Stage-1 extraction once, under the shared limiter."""
        return await self._aextract(case, judge_limiter)

    def evaluate_with_items(
        self, case: EvaluationCase, items: list[dict]
    ) -> EvaluationResult:
        """Classifies pre-extracted (group-shared) items against this output.

        The extraction was done once for the group, so this output's
        ``judge_call_count`` reflects only its own classification call(s), and
        ``details["shared_extraction"]`` is ``True``.
        """
        self.validate_case(case)
        if not items:
            return self._grouped_not_applicable()
        started = time.monotonic()
        judgments, batch_count = self._run_classify(case, items)
        return self._grouped_result(
            items, judgments, batch_count, classify_ms=_ms(started), total_ms=_ms(started)
        )

    async def a_evaluate_with_items(
        self, case: EvaluationCase, items: list[dict], *, judge_limiter: asyncio.Semaphore
    ) -> EvaluationResult:
        self.validate_case(case)
        if not items:
            return self._grouped_not_applicable()
        started = time.monotonic()
        judgments, batch_count = await self._arun_classify(case, items, judge_limiter)
        return self._grouped_result(
            items, judgments, batch_count, classify_ms=_ms(started), total_ms=_ms(started)
        )

    def _grouped_result(self, items, judgments, batch_count, *, classify_ms, total_ms):
        built = self._build_items(items, judgments)
        result = self._coverage_result(
            built, judge_call_count=batch_count, batch_count=batch_count,
            extract_ms=0.0, classify_ms=classify_ms, total_ms=total_ms,
        )
        result.details["shared_extraction"] = True
        result.details["classification_calls"] = batch_count
        return result

    def _grouped_not_applicable(self) -> EvaluationResult:
        result = self._coverage_not_applicable(judge_call_count=0, batch_count=0)
        result.details["shared_extraction"] = True
        result.details["classification_calls"] = 0
        return result

    # --- DAG Stage 1: whole-source extraction (context only) ----------------

    def _classify_input(self, case: EvaluationCase) -> str:
        """Whole-source coverage is task-agnostic: no task text to the classifier."""
        return ""

    def _extract_requirements(self, case: EvaluationCase) -> list[dict]:
        prompt = render_coverage_extract_prompt(context=render_value(case.context))
        with tracing.judge_span(
            f"{self.name}.extract",
            {"idp_eval.metric": self.name, "idp_eval.stage": "extract"},
        ):
            response = self._llm.generate_object(
                prompt=prompt, schema=COVERAGE_EXTRACT_SCHEMA
            )
        raw = _dedup(response.get("items", []), "source_item")
        return [
            {"id": f"{self._id_prefix}{i}", "source_item": item["source_item"]}
            for i, item in enumerate(raw, start=1)
        ]

    def _plan_batches(self, requirements: list[dict]) -> list[list[dict]]:
        """One classify call normally; safety-batch only very large item sets."""
        if len(requirements) <= CLASSIFICATION_BATCH_THRESHOLD:
            return [list(requirements)]
        return _split_batches(requirements, SAFETY_CLASSIFICATION_BATCH_SIZE)

    # --- result shaping (coverage-specific; used by both modes) -------------

    def _build_scored_result(
        self, items, *, extract_ms, classify_ms, total_ms, batch_count
    ) -> EvaluationResult:
        # DAG: 1 extract + one classify per batch.
        return self._coverage_result(
            items, judge_call_count=1 + batch_count, batch_count=batch_count,
            extract_ms=extract_ms, classify_ms=classify_ms, total_ms=total_ms,
        )

    def _not_applicable_result(self) -> EvaluationResult:
        return self._coverage_not_applicable(judge_call_count=1, batch_count=0)

    def _coverage_result(
        self, items, *, judge_call_count, batch_count, extract_ms, classify_ms, total_ms
    ) -> EvaluationResult:
        score = calculate_coverage(items)
        covered = sum(i["status"] == "covered" for i in items)
        partial = sum(i["status"] == "partial" for i in items)
        missing = sum(i["status"] == "missing" for i in items)
        count = len(items)

        tracing.set_current_span_attributes({
            "coverage.mode": self._mode,
            "coverage.item_count": count,
            "coverage.judge_call_count": judge_call_count,
            "coverage.batch_count": batch_count,
            "coverage.verbose": self._verbose,
            "coverage.extract_ms": extract_ms,
            "coverage.classify_ms": classify_ms,
            "coverage.total_ms": total_ms,
        })

        details = {
            "mode": self._mode,
            "final_item_count": count,
            "covered_count": covered,
            "partial_count": partial,
            "missing_count": missing,
            "judge_call_count": judge_call_count,
            "batch_count": batch_count,
            "extract_ms": extract_ms,
            "classify_ms": classify_ms,
            "total_ms": total_ms,
            "verbose": self._verbose,
        }
        # Compact mode keeps details lean; the full item array is verbose-only.
        if self._verbose:
            details["items"] = items

        return EvaluationResult(
            metric=self.name,
            score=score,
            label=coverage_label(score),
            explanation=self._compact_explanation(count, covered, partial, missing),
            details=details,
        )

    def _coverage_not_applicable(
        self, *, judge_call_count, batch_count
    ) -> EvaluationResult:
        return EvaluationResult(
            metric=self.name,
            score=None,
            label="not_applicable",
            explanation=self._not_applicable_reason,
            details={
                "mode": self._mode,
                "final_item_count": 0,
                "covered_count": 0,
                "partial_count": 0,
                "missing_count": 0,
                "judge_call_count": judge_call_count,
                "batch_count": batch_count,
                "verbose": self._verbose,
            },
        )

    # --- G-Eval: one-call identify + classify -------------------------------

    def _evaluate_g_eval(self, case: EvaluationCase) -> EvaluationResult:
        started = time.monotonic()
        prompt = render_coverage_geval_prompt(
            context=render_value(case.context),
            output=render_value(case.output),
            verbose=self._verbose,
        )
        schema = (
            COVERAGE_GEVAL_SCHEMA_VERBOSE
            if self._verbose
            else COVERAGE_GEVAL_SCHEMA_COMPACT
        )
        t0 = time.monotonic()
        with tracing.judge_span(
            f"{self.name}.evaluate",
            {"idp_eval.metric": self.name, "idp_eval.stage": "evaluate"},
        ):
            response = self._llm.generate_object(prompt=prompt, schema=schema)
        classify_ms = _ms(t0)

        items = self._build_geval_items(self._validate_geval_items(response))
        if not items:
            return self._coverage_not_applicable(judge_call_count=1, batch_count=1)
        return self._coverage_result(
            items, judge_call_count=1, batch_count=1,
            extract_ms=0.0, classify_ms=classify_ms, total_ms=_ms(started),
        )

    def _validate_geval_items(self, response) -> list[dict]:
        if not isinstance(response, dict) or "items" not in response:
            raise ValueError("Malformed coverage response: missing `items`.")
        raw = response["items"]
        if not isinstance(raw, list):
            raise ValueError("Malformed coverage response: `items` must be a list.")
        validated: list[dict] = []
        for index, item in enumerate(raw, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"Malformed coverage item {index}: expected an object.")
            si = item.get("source_item")
            present = item.get("meaningfully_present")
            full = item.get("fully_present")
            if not isinstance(si, str) or not si.strip():
                raise ValueError(
                    f"Malformed coverage item {index}: `source_item` must be a "
                    "non-empty string."
                )
            if not isinstance(present, bool) or not isinstance(full, bool):
                raise ValueError(
                    f"Malformed coverage item {index}: meaningfully_present/"
                    "fully_present must be booleans."
                )
            coverage_status_from_binary(present, full)  # rejects impossible combos
            reason = item.get("reason", "")
            reason = "" if reason is None else reason
            if not isinstance(reason, str):
                raise ValueError(f"Malformed coverage item {index}: `reason` must be a string.")
            validated.append({
                "source_item": si, "meaningfully_present": present,
                "fully_present": full, "reason": reason,
            })
        return validated

    def _build_geval_items(self, raw_items: list[dict]) -> list[dict]:
        seen: set[str] = set()
        items: list[dict] = []
        for raw in raw_items:
            norm = _normalize(raw["source_item"])
            if norm in seen:
                continue
            seen.add(norm)
            status = coverage_status_from_binary(
                raw["meaningfully_present"], raw["fully_present"]
            )
            items.append({
                "id": f"{self._id_prefix}{len(items) + 1}",
                "source_item": raw["source_item"],
                "meaningfully_present": raw["meaningfully_present"],
                "fully_present": raw["fully_present"],
                "status": status,
                "score": coverage_status_score(status),
                "reason": raw["reason"],
            })
        return items


class SourceCoverageEvaluator(CoverageEvaluator):
    """Deprecated alias for :class:`CoverageEvaluator` (whole-source coverage).

    Kept for backward compatibility. Defaults to ``mode="g_eval"`` (its historical
    one-call behavior) and reports metric name ``source_coverage`` with
    ``source_coverage.*`` spans. Prefer ``CoverageEvaluator(judge, mode=...)``.
    """

    name = "source_coverage"

    def __init__(self, llm, *, mode: str = "g_eval", verbose: bool = False):
        import warnings

        warnings.warn(
            "SourceCoverageEvaluator is deprecated; use "
            "CoverageEvaluator(judge, mode=...).",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(llm, mode=mode, verbose=verbose)
