"""Coverage evaluator (two-stage, QAG-style).

Coverage measures how completely the generated output represents the
task-relevant information in the supplied context. It runs two isolated judge
calls per case:

    STAGE 1 - extraction:  input + context            -> atomic requirements
    STAGE 2 - classification: requirements + output   -> two binary judgments/req

Python then derives covered/partial/missing from the booleans, maps them to
1.0/0.5/0.0, and averages. Stage 1 never sees the output (so extraction cannot be
biased by it), and Stage 2 cannot change the requirement set (so the denominator
is fixed). The judge never emits a numeric score.

Cost: exactly two judge calls for a non-applicable-or-scored evaluation, and one
call when extraction finds no requirements. It is two calls total, never one call
per requirement.

Coverage answers "did the output OMIT important task-relevant information?" The
"did the output ADD unsupported information?" question is faithfulness's job;
coverage never performs hallucination detection.

Known limitation (v1): deduplication is normalized-exact only, so semantic
near-duplicate requirements may remain distinct and each count in the denominator.
"""

from __future__ import annotations

import json

from idp_eval.models import EvaluationCase, EvaluationResult, Evaluator
from idp_eval.prompts.coverage_classify import (
    COVERAGE_CLASSIFY_SCHEMA,
    render_coverage_classify_prompt,
)
from idp_eval.prompts.coverage_extract import (
    COVERAGE_EXTRACT_SCHEMA,
    render_coverage_extract_prompt,
)
from idp_eval.scoring import (
    calculate_coverage,
    coverage_status_from_binary,
    coverage_status_score,
    score_to_label,
)


def _normalize_requirement(text: str) -> str:
    """Normalizes requirement text for exact-match deduplication."""
    return " ".join(text.lower().split())


def _dedup_requirements(requirements: list[dict]) -> list[dict]:
    """Removes normalized-exact duplicate requirements, keeping first occurrence.

    Args:
        requirements: Parsed extraction entries (each with ``requirement``).

    Returns:
        The requirements with normalized-exact duplicates removed.
    """
    seen: set[str] = set()
    deduped: list[dict] = []
    for req in requirements:
        key = _normalize_requirement(req["requirement"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(req)
    return deduped


class CoverageEvaluator(Evaluator):
    """Two-stage semantic coverage of task-relevant context.

    Direction: ``input + context -> output``. Higher score is better.
    """

    name = "coverage"

    def __init__(self, llm):
        """Initializes the evaluator.

        Args:
            llm: A judge object exposing
                ``generate_object(prompt, schema: dict) -> dict`` where ``prompt``
                is a Phoenix-style message list. The same judge is reused for both
                stages. Phoenix's ``LLM`` satisfies this contract.
        """
        self._llm = llm

    def evaluate(self, case: EvaluationCase) -> EvaluationResult:
        """Evaluates coverage for a single case using two judge calls."""
        requirements = self._extract_requirements(case)

        # No task-relevant requirements: skip Stage 2, do not divide by zero, and
        # do not report perfect coverage for a failure to identify requirements.
        if not requirements:
            return EvaluationResult(
                metric=self.name,
                score=None,
                label="not_applicable",
                explanation=(
                    "No task-relevant requirements were identified in the "
                    "supplied context."
                ),
                details={
                    "total_requirements": 0,
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
            f"Coverage was {percentage}% across {len(items)} task-relevant "
            f"requirements: {len(covered)} covered, {len(partial)} partial, "
            f"and {len(missing)} missing."
        )

        return EvaluationResult(
            metric=self.name,
            score=score,
            label=score_to_label(score),
            explanation=explanation,
            details={
                "total_requirements": len(items),
                "covered_count": len(covered),
                "partial_count": len(partial),
                "missing_count": len(missing),
                "items": items,
            },
        )

    def _extract_requirements(self, case: EvaluationCase) -> list[dict]:
        """Stage 1: derives deduplicated, id-tagged requirements (no output).

        Returns:
            A list of ``{"id": "r<n>", "requirement": str}`` in extraction order.
        """
        prompt = render_coverage_extract_prompt(
            input_text=case.input,
            context=case.context,
        )
        response = self._llm.generate_object(
            prompt=prompt,
            schema=COVERAGE_EXTRACT_SCHEMA,
        )
        raw = _dedup_requirements(response.get("requirements", []))
        return [
            {"id": f"r{index}", "requirement": req["requirement"]}
            for index, req in enumerate(raw, start=1)
        ]

    def _classify_requirements(
        self, case: EvaluationCase, requirements: list[dict]
    ) -> list[dict]:
        """Stage 2: batched binary classification of the fixed requirement set."""
        requirements_json = json.dumps(requirements, ensure_ascii=False)
        prompt = render_coverage_classify_prompt(
            input_text=case.input,
            requirements_json=requirements_json,
            output=case.output,
        )
        response = self._llm.generate_object(
            prompt=prompt,
            schema=COVERAGE_CLASSIFY_SCHEMA,
        )
        return response.get("requirements", [])

    @staticmethod
    def _build_items(
        requirements: list[dict], classifications: list[dict]
    ) -> list[dict]:
        """Validates id integrity and derives per-requirement status + score.

        The classification must cover exactly the extracted requirement ids —
        once each, no unknown ids, none missing. Results are reconstructed in the
        original extraction order regardless of the order returned.

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
                    "requirement": req["requirement"],
                    "meaningfully_present": present,
                    "fully_present": full,
                    "status": status,
                    "score": coverage_status_score(status),
                    "reason": entry.get("reason", ""),
                }
            )
        return items
