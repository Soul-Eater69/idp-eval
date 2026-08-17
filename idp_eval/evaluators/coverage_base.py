"""Shared internals for the two-stage coverage evaluators.

Not part of the public API. ``TaskCoverageEvaluator`` (task-scoped) and
``SourceCoverageEvaluator`` (whole-source) share this base so the classification
prompt, id-integrity validation, status derivation, scoring, empty-extraction
handling, and result shaping exist in exactly one place. Only Stage 1 extraction
(scope) and a few labels differ between subclasses.
"""

from __future__ import annotations

import json

from idp_eval import tracing
from idp_eval.models import EvaluationCase, EvaluationResult, Evaluator
from idp_eval.prompts.coverage_classify import (
    COVERAGE_CLASSIFY_SCHEMA,
    render_coverage_classify_prompt,
)
from idp_eval.scoring import (
    calculate_coverage,
    coverage_status_from_binary,
    coverage_status_score,
    score_to_label,
)


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

    def __init__(self, llm):
        """Args: llm: judge exposing ``generate_object(prompt, schema) -> dict``.

        The same judge is reused for both stages (Phoenix's ``LLM`` satisfies it).
        """
        self._llm = llm

    def evaluate(self, case: EvaluationCase) -> EvaluationResult:
        """Evaluates coverage for one case using (at most) two judge calls."""
        requirements = self._extract_requirements(case)

        # No items: skip Stage 2, never divide by zero, and never report perfect
        # coverage for a failure to identify what should be covered.
        if not requirements:
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
                },
            )

        classifications = self._classify_requirements(case, requirements)
        items = self._build_items(requirements, classifications)
        score = calculate_coverage(items)

        covered = [i for i in items if i["status"] == "covered"]
        partial = [i for i in items if i["status"] == "partial"]
        missing = [i for i in items if i["status"] == "missing"]

        percentage = f"{round(score * 100, 1):g}"
        explanation = (
            f"{self._coverage_label} was {percentage}% across {len(items)} "
            f"{self._explanation_unit}: {len(covered)} covered, "
            f"{len(partial)} partial, and {len(missing)} missing."
        )

        return EvaluationResult(
            metric=self.name,
            score=score,
            label=score_to_label(score),
            explanation=explanation,
            details={
                self._total_key: len(items),
                "covered_count": len(covered),
                "partial_count": len(partial),
                "missing_count": len(missing),
                "items": items,
            },
        )

    # --- Stage 1 (subclass-specific) ----------------------------------------

    def _extract_requirements(self, case: EvaluationCase) -> list[dict]:
        """Stage 1: returns id-tagged items ``[{"id", <item_key>: text}]``."""
        raise NotImplementedError

    # --- Stage 2 (shared) ---------------------------------------------------

    def _classify_input(self, case: EvaluationCase) -> str:
        """Task text passed to the classifier for semantic clarification.

        Task-scoped coverage passes ``input``; source coverage overrides this to
        ``""`` because it is task-agnostic.
        """
        return case.input

    def _classify_requirements(
        self, case: EvaluationCase, requirements: list[dict]
    ) -> list[dict]:
        """Stage 2: batched binary classification of the fixed item set.

        The shared classifier prompt is item-key agnostic: items are serialized
        as ``{"id", "requirement"}`` regardless of the internal field name.
        """
        payload = [
            {"id": r["id"], "requirement": r[self._item_key]} for r in requirements
        ]
        requirements_json = json.dumps(payload, ensure_ascii=False)
        prompt = render_coverage_classify_prompt(
            input_text=self._classify_input(case),
            requirements_json=requirements_json,
            output=case.output,
        )
        with tracing.judge_span(
            f"{self.name}.classify",
            {"idp_eval.metric": self.name, "idp_eval.stage": "classify"},
        ):
            response = self._llm.generate_object(
                prompt=prompt,
                schema=COVERAGE_CLASSIFY_SCHEMA,
            )
        return response.get("requirements", [])

    def _build_items(
        self, requirements: list[dict], classifications: list[dict]
    ) -> list[dict]:
        """Validates id integrity and derives per-item status + score.

        The classification must cover exactly the extracted ids — once each, no
        unknown ids, none missing. Results are reconstructed in the original
        extraction order regardless of the order returned.

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
