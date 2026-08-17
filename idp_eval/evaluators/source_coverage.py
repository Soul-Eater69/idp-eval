"""Whole-source coverage evaluator.

``SourceCoverageEvaluator`` (metric ``source_coverage``) measures how much of the
important information in the source/context is represented in the output,
independent of any task:

    STAGE 1 - extraction:  context                   -> important source items
    STAGE 2 - classification: items + output          -> two binary judgments/item

Direction: ``source (context) -> output``. Unlike task coverage, Stage 1 receives
only the context (no input/task, no instructions, no output), so the whole source
defines what should be covered. Typical uses: summarization, document
compression, and generic source-to-output transformations.

Shares
:class:`~idp_eval.evaluators.coverage_base._TwoStageCoverageEvaluator` for the
classification prompt, id validation, scoring, and result shaping; only Stage 1
scope and result-field naming differ.
"""

from __future__ import annotations

from idp_eval import tracing
from idp_eval.evaluators.coverage_base import _TwoStageCoverageEvaluator, _dedup
from idp_eval.models import EvaluationCase
from idp_eval.prompts.source_coverage_extract import (
    SOURCE_COVERAGE_EXTRACT_SCHEMA,
    render_source_coverage_extract_prompt,
)


class SourceCoverageEvaluator(_TwoStageCoverageEvaluator):
    """Coverage of the important information in the whole source (context).

    Direction: ``context -> output``. The task/input is not used to determine
    relevance. Higher score is better.
    """

    name = "source_coverage"
    _item_key = "source_item"
    _total_key = "total_items"
    _id_prefix = "s"
    _coverage_label = "Source coverage"
    _explanation_unit = "source items"
    _not_applicable_reason = (
        "No important source items were identified in the supplied context."
    )

    def _extract_requirements(self, case: EvaluationCase) -> list[dict]:
        """Stage 1: important source items from context only (no task/output)."""
        prompt = render_source_coverage_extract_prompt(context=case.context)
        with tracing.judge_span(
            f"{self.name}.extract",
            {"idp_eval.metric": self.name, "idp_eval.stage": "extract"},
        ):
            response = self._llm.generate_object(
                prompt=prompt,
                schema=SOURCE_COVERAGE_EXTRACT_SCHEMA,
            )
        raw = _dedup(response.get("source_items", []), "source_item")
        return [
            {"id": f"{self._id_prefix}{index}", "source_item": item["source_item"]}
            for index, item in enumerate(raw, start=1)
        ]

    def _classify_input(self, case: EvaluationCase) -> str:
        """Source coverage is task-agnostic: never pass the task to the classifier."""
        return ""
