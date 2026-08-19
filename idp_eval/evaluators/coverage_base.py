"""Internal two-stage pipeline for task coverage.

Not part of the public API. ``TaskCoverageEvaluator`` uses this base for its
output-isolated extraction, fixed-denominator classification, id-integrity
validation, status derivation, batching, scoring, and result shaping.

Performance/robustness (the metric semantics are unchanged — covered/partial/
missing = 1.0/0.5/0.0, deterministic Python mean, fixed output-isolated
denominator):

- **Compact Stage-2 output by default** (``verbose=False``): the judge returns
  only booleans per item, no per-item prose, so responses stay small and fast.
  ``verbose=True`` additionally requests short reasons for partial/missing items;
  it changes diagnostics only, never the booleans/status/score/label.
- **Adaptive Stage-2 batching**: a large denominator is split into batches of
  ``classification_batch_size`` (default 12) so no single classification request
  carries the whole denominator. This is an *operational request-size control*
  (distinct from the semantic denominator/item count); it reduces the risk of an
  individual classification request exceeding the gateway timeout, but is a
  mitigation, not a guarantee (latency also depends on model load, input/output
  length, and gateway conditions). Batches are merged deterministically by stable
  id; batch boundaries never affect the score. Batching can increase the total
  request count for large denominators (1 extraction + one classification per
  batch) — that is intentional: smaller, parallelizable requests over minimum
  call count.
- **Async path**: :meth:`a_evaluate` runs the (dependent) extract->classify
  stages for one case, with independent Stage-2 batches overlapping under a
  shared, caller-provided concurrency limiter. The sync :meth:`evaluate` remains
  serial.
- **Timing/count diagnostics** in ``details`` and on trace attributes.
"""

from __future__ import annotations

import asyncio
import json
import time

from idp_eval import tracing
from idp_eval.models import EvaluationCase, EvaluationResult, Evaluator
from idp_eval.prompts.coverage_classify import (
    COVERAGE_CLASSIFY_SCHEMA_COMPACT,
    COVERAGE_CLASSIFY_SCHEMA_VERBOSE,
    render_coverage_classify_prompt,
)
from idp_eval.rendering import render_value
from idp_eval.scoring import (
    calculate_coverage,
    coverage_label,
    coverage_status_from_binary,
    coverage_status_score,
)

# Default Stage-2 batch size (operational request-size control, NOT a semantic
# item cap): small enough to reduce the risk of a single classification request
# exceeding the gateway timeout, large enough to avoid excessive calls.
DEFAULT_CLASSIFICATION_BATCH_SIZE = 12

# Soft diagnostic threshold: a denominator above this is flagged (never truncated
# and never a scoring cutoff).
LARGE_DENOMINATOR_THRESHOLD = 20


def _normalize(text: str) -> str:
    """Normalizes item text for exact-match deduplication."""
    return " ".join(text.lower().split())


def _dedup(items: list[dict], key: str) -> list[dict]:
    """Removes normalized-exact duplicates by ``key``, keeping first occurrence."""
    seen: set[str] = set()
    deduped: list[dict] = []
    for item in items:
        norm = _normalize(item[key])
        if norm in seen:
            continue
        seen.add(norm)
        deduped.append(item)
    return deduped


def _split_batches(items: list[dict], size: int) -> list[list[dict]]:
    """Splits ``items`` into contiguous batches of at most ``size`` (order kept)."""
    if size <= 0 or len(items) <= size:
        return [list(items)]
    return [items[i : i + size] for i in range(0, len(items), size)]


def _ms(start: float) -> float:
    """Elapsed milliseconds since a monotonic ``start``, rounded."""
    return round((time.monotonic() - start) * 1000, 1)


class _TwoStageCoverageEvaluator(Evaluator):
    """Base for two-stage coverage: extract items, then classify them vs output.

    Stage 1 (subclass-specific scope) yields a fixed, id-tagged item set; Stage 2
    (shared) returns two booleans per item; Python derives
    covered/partial/missing (1.0/0.5/0.0) and averages. The judge never emits a
    numeric score. Not exported.

    Subclasses configure the class attributes and implement
    :meth:`_extract_requirements`.
    """

    # Configured by subclasses (this base is abstract and never instantiated).
    name = "task_coverage"
    _item_key = "requirement"          # per-item text field in details["items"]
    _total_key = "total_requirements"  # details count key
    _id_prefix = "r"
    _coverage_label = "Coverage"       # explanation prefix
    _explanation_unit = "task-relevant requirements"
    _not_applicable_reason = (
        "No task-relevant requirements were identified in the supplied context."
    )

    def __init__(
        self,
        llm,
        *,
        verbose: bool = False,
        classification_batch_size: int = DEFAULT_CLASSIFICATION_BATCH_SIZE,
    ):
        """Args:
            llm: judge exposing ``generate_object(prompt, schema) -> dict`` (the
                same judge is reused for both stages; Phoenix's ``LLM`` fits).
            verbose: when ``True``, Stage 2 also returns short reasons for
                partial/missing items. Diagnostics only — score/label/statuses
                are identical to ``verbose=False``.
            classification_batch_size: max items per Stage-2 classification call
                (default 12; an operational request-size control, not a semantic
                cap). Batches are merged by stable id; the denominator and score
                are unaffected by batching.

        Raises:
            ValueError: If ``classification_batch_size`` is not a positive integer.
        """
        if (
            isinstance(classification_batch_size, bool)
            or not isinstance(classification_batch_size, int)
            or classification_batch_size < 1
        ):
            raise ValueError(
                "classification_batch_size must be a positive integer, got "
                f"{classification_batch_size!r}."
            )
        self._llm = llm
        self._verbose = verbose
        self._classification_batch_size = classification_batch_size

    # --- sync entry point ---------------------------------------------------

    def evaluate(self, case: EvaluationCase) -> EvaluationResult:
        """Evaluates coverage for one case (serial two-stage, adaptive batching).

        Required-field validation runs first, so a missing required field fails
        before any judge call — identically to the framework entry point.
        """
        self.validate_case(case)
        started = time.monotonic()

        t0 = time.monotonic()
        requirements = self._extract_requirements(case)
        extract_ms = _ms(t0)

        if not requirements:
            self._trace_not_applicable(extract_ms)
            return self._not_applicable_result()

        t1 = time.monotonic()
        judgments, batch_count = self._run_classify(case, requirements)
        classify_ms = _ms(t1)

        items = self._build_items(requirements, judgments)
        return self._build_scored_result(
            items,
            extract_ms=extract_ms,
            classify_ms=classify_ms,
            total_ms=_ms(started),
            batch_count=batch_count,
        )

    # --- async entry point --------------------------------------------------

    async def a_evaluate(
        self, case: EvaluationCase, *, judge_limiter: asyncio.Semaphore
    ) -> EvaluationResult:
        """Async coverage for one case; independent Stage-2 batches may overlap.

        Extraction and classification for a single case stay dependent (extract
        then classify). ``judge_limiter`` is the shared, global concurrency cap
        for judge calls; it is acquired around each real judge call so total
        in-flight gateway requests never exceed it.
        """
        self.validate_case(case)
        started = time.monotonic()

        t0 = time.monotonic()
        requirements = await self._aextract(case, judge_limiter)
        extract_ms = _ms(t0)

        if not requirements:
            self._trace_not_applicable(extract_ms)
            return self._not_applicable_result()

        t1 = time.monotonic()
        judgments, batch_count = await self._arun_classify(
            case, requirements, judge_limiter
        )
        classify_ms = _ms(t1)

        items = self._build_items(requirements, judgments)
        return self._build_scored_result(
            items,
            extract_ms=extract_ms,
            classify_ms=classify_ms,
            total_ms=_ms(started),
            batch_count=batch_count,
        )

    # --- Stage 1 (subclass-specific) ----------------------------------------

    def _extract_requirements(self, case: EvaluationCase) -> list[dict]:
        """Stage 1: returns id-tagged items ``[{"id", <item_key>: text}]``."""
        raise NotImplementedError

    async def _aextract(
        self, case: EvaluationCase, limiter: asyncio.Semaphore
    ) -> list[dict]:
        """Runs the (sync) extraction in a worker thread under the limiter.

        The sync extractor's span/prompt are reused unchanged; OTel context is
        copied into the thread, so the extract span nests under the case root.
        """
        async with limiter:
            return await asyncio.to_thread(self._extract_requirements, case)

    # --- Stage 2 (shared) ---------------------------------------------------

    def _classify_input(self, case: EvaluationCase) -> str:
        """Task text passed to the classifier for semantic clarification.

        Task coverage passes ``input`` for semantic clarification.
        """
        return case.input

    def _classify_one_batch(
        self,
        batch: list[dict],
        *,
        rendered_input: str,
        rendered_output: str,
        span_name: str,
        batch_index: int | None,
        batch_count: int,
    ) -> list[dict]:
        """Classifies one batch of the fixed item set in a single judge call.

        The classifier prompt is item-key agnostic: items are serialized as
        ``{"id", "requirement"}`` regardless of the internal field name, and only
        ids are returned (requirement text is never echoed back).
        """
        payload = [
            {"id": r["id"], "requirement": r[self._item_key]} for r in batch
        ]
        prompt = render_coverage_classify_prompt(
            input_text=rendered_input,
            requirements_json=json.dumps(payload, ensure_ascii=False),
            output=rendered_output,
            verbose=self._verbose,
        )
        schema = (
            COVERAGE_CLASSIFY_SCHEMA_VERBOSE
            if self._verbose
            else COVERAGE_CLASSIFY_SCHEMA_COMPACT
        )
        attributes = {"idp_eval.metric": self.name, "idp_eval.stage": "classify"}
        if batch_index is not None:
            attributes["coverage.batch_index"] = batch_index
            attributes["coverage.batch_count"] = batch_count
        with tracing.judge_span(span_name, attributes):
            response = self._llm.generate_object(prompt=prompt, schema=schema)
        return response.get("requirements", [])

    def _plan_batches(self, requirements: list[dict]) -> list[list[dict]]:
        """Splits the fixed item set into Stage-2 classification batches.

        Default: fixed-size batches of ``classification_batch_size``. Subclasses
        (e.g. whole-source coverage) may override to keep a single call for normal
        sizes and only split unusually large item sets.
        """
        return _split_batches(requirements, self._classification_batch_size)

    def _run_classify(
        self, case: EvaluationCase, requirements: list[dict]
    ) -> tuple[list[dict], int]:
        """Serial Stage 2: classify each batch, return merged judgments + count.

        A single batch keeps the historical trace shape (one ``{name}.classify``
        span directly wrapping the model call). Multiple batches nest
        ``{name}.classify.batch`` child spans under one ``{name}.classify`` span.
        """
        rendered_input = render_value(self._classify_input(case))
        rendered_output = render_value(case.output)
        batches = self._plan_batches(requirements)
        batch_count = len(batches)

        if batch_count == 1:
            judgments = self._classify_one_batch(
                batches[0],
                rendered_input=rendered_input,
                rendered_output=rendered_output,
                span_name=f"{self.name}.classify",
                batch_index=None,
                batch_count=1,
            )
            return judgments, 1

        judgments: list[dict] = []
        with tracing.judge_span(
            f"{self.name}.classify",
            {
                "idp_eval.metric": self.name,
                "idp_eval.stage": "classify",
                "coverage.batch_count": batch_count,
            },
        ):
            for index, batch in enumerate(batches):
                judgments.extend(
                    self._classify_one_batch(
                        batch,
                        rendered_input=rendered_input,
                        rendered_output=rendered_output,
                        span_name=f"{self.name}.classify.batch",
                        batch_index=index,
                        batch_count=batch_count,
                    )
                )
        return judgments, batch_count

    async def _arun_classify(
        self,
        case: EvaluationCase,
        requirements: list[dict],
        limiter: asyncio.Semaphore,
    ) -> tuple[list[dict], int]:
        """Async Stage 2: independent batches overlap under the shared limiter."""
        rendered_input = render_value(self._classify_input(case))
        rendered_output = render_value(case.output)
        batches = self._plan_batches(requirements)
        batch_count = len(batches)

        if batch_count == 1:
            async with limiter:
                judgments = await asyncio.to_thread(
                    self._classify_one_batch,
                    batches[0],
                    rendered_input=rendered_input,
                    rendered_output=rendered_output,
                    span_name=f"{self.name}.classify",
                    batch_index=None,
                    batch_count=1,
                )
            return judgments, 1

        async def run_batch(index: int, batch: list[dict]) -> list[dict]:
            async with limiter:
                return await asyncio.to_thread(
                    self._classify_one_batch,
                    batch,
                    rendered_input=rendered_input,
                    rendered_output=rendered_output,
                    span_name=f"{self.name}.classify.batch",
                    batch_index=index,
                    batch_count=batch_count,
                )

        with tracing.judge_span(
            f"{self.name}.classify",
            {
                "idp_eval.metric": self.name,
                "idp_eval.stage": "classify",
                "coverage.batch_count": batch_count,
            },
        ):
            batch_results = await asyncio.gather(
                *(run_batch(i, b) for i, b in enumerate(batches))
            )

        merged: list[dict] = []
        for result in batch_results:
            merged.extend(result)
        return merged, batch_count

    def _build_items(
        self, requirements: list[dict], classifications: list[dict]
    ) -> list[dict]:
        """Validates id integrity and derives per-item status + score.

        The merged classification must cover exactly the extracted ids — once
        each, no unknown ids, none missing — even when produced across multiple
        Stage-2 batches. Results are reconstructed in the original extraction
        order regardless of batch/return order.

        Raises:
            ValueError: On duplicate, unknown, or missing ids, non-boolean
                fields, or a logically inconsistent binary combination.
        """
        by_id: dict[str, dict] = {}
        for entry in classifications:
            cid = entry["id"]
            if cid in by_id:
                raise ValueError(
                    f"Duplicate requirement id in classification: {cid!r}"
                )
            by_id[cid] = entry

        expected = {req["id"] for req in requirements}
        unknown = sorted(set(by_id) - expected)
        if unknown:
            raise ValueError(
                f"Unknown requirement id(s) in classification: {unknown}"
            )
        missing = sorted(expected - set(by_id))
        if missing:
            raise ValueError(
                f"Missing classification for requirement id(s): {missing}"
            )

        items: list[dict] = []
        for req in requirements:  # original extraction order
            entry = by_id[req["id"]]
            present = entry["meaningfully_present"]
            full = entry["fully_present"]
            if not isinstance(present, bool) or not isinstance(full, bool):
                raise ValueError(
                    f"Non-boolean classification for {req['id']!r}: "
                    "meaningfully_present/fully_present must be booleans."
                )
            status = coverage_status_from_binary(present, full)
            items.append(
                {
                    "id": req["id"],
                    self._item_key: req[self._item_key],
                    "meaningfully_present": present,
                    "fully_present": full,
                    "status": status,
                    "score": coverage_status_score(status),
                    "reason": entry.get("reason", ""),
                }
            )
        return items

    # --- result shaping (shared by sync and async) --------------------------

    def _build_scored_result(
        self,
        items: list[dict],
        *,
        extract_ms: float,
        classify_ms: float,
        total_ms: float,
        batch_count: int,
    ) -> EvaluationResult:
        """Builds the scored result + diagnostics; sets trace summary attributes."""
        score = calculate_coverage(items)
        covered = [i for i in items if i["status"] == "covered"]
        partial = [i for i in items if i["status"] == "partial"]
        missing = [i for i in items if i["status"] == "missing"]
        item_count = len(items)
        large_denominator = item_count > LARGE_DENOMINATOR_THRESHOLD
        # Deterministic: 1 extraction call + one classification call per batch.
        judge_call_count = 1 + batch_count

        explanation = self._compact_explanation(
            item_count, len(covered), len(partial), len(missing)
        )

        tracing.set_current_span_attributes(
            {
                "coverage.item_count": item_count,
                "coverage.batch_count": batch_count,
                "coverage.batch_size": self._classification_batch_size,
                "coverage.judge_call_count": judge_call_count,
                "coverage.verbose": self._verbose,
                "coverage.large_denominator": large_denominator,
                "coverage.extract_ms": extract_ms,
                "coverage.classify_ms": classify_ms,
                "coverage.total_ms": total_ms,
            }
        )

        return EvaluationResult(
            metric=self.name,
            score=score,
            label=coverage_label(score),
            explanation=explanation,
            details={
                self._total_key: item_count,
                "covered_count": len(covered),
                "partial_count": len(partial),
                "missing_count": len(missing),
                "items": items,
                # Diagnostics (do not affect scoring; surfaced for auditing).
                "final_item_count": item_count,
                "batch_count": batch_count,
                "batch_size": self._classification_batch_size,
                "judge_call_count": judge_call_count,
                "verbose": self._verbose,
                "large_denominator": large_denominator,
                "extract_ms": extract_ms,
                "classify_ms": classify_ms,
                "total_ms": total_ms,
            },
        )

    def _compact_explanation(
        self, item_count: int, covered: int, partial: int, missing: int
    ) -> str:
        """Deterministic Python summary of coverage (no LLM call).

        This is a computed summary, not a judge rationale. ``verbose=True`` adds
        item-level semantic reasons from the judge for partial/missing items; it
        does not change this overall text's determinism.
        """
        unit = self._explanation_unit
        if covered == item_count:
            return f"All {item_count} {unit} are fully represented."
        if missing == item_count:
            return f"None of the {item_count} {unit} are represented."
        return (
            f"{covered} of {item_count} {unit} are fully covered; "
            f"{partial} partial and {missing} missing."
        )

    def _not_applicable_result(self) -> EvaluationResult:
        """Builds the not-applicable result (empty extraction, no Stage 2).

        Exactly one judge call was made (the extraction), so
        ``judge_call_count`` is 1; no classification batches ran.
        """
        return EvaluationResult(
            metric=self.name,
            score=None,
            label="not_applicable",
            explanation=self._not_applicable_reason,
            details={
                self._total_key: 0,
                "covered_count": 0,
                "partial_count": 0,
                "missing_count": 0,
                "items": [],
                "final_item_count": 0,
                "batch_count": 0,
                "judge_call_count": 1,
            },
        )

    def _trace_not_applicable(self, extract_ms: float) -> None:
        """Records compact diagnostics for the empty-extraction path."""
        tracing.set_current_span_attributes(
            {
                "coverage.item_count": 0,
                "coverage.batch_count": 0,
                "coverage.batch_size": self._classification_batch_size,
                "coverage.judge_call_count": 1,
                "coverage.verbose": self._verbose,
                "coverage.large_denominator": False,
                "coverage.extract_ms": extract_ms,
                "coverage.classify_ms": 0.0,
            }
        )
