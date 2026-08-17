"""Task-scoped coverage evaluator.

``TaskCoverageEvaluator`` (metric ``task_coverage``) measures how much of the
source information *relevant to the requested task* is represented in the output:

    STAGE 1 - extraction:  input + context          -> task-relevant items
    STAGE 2 - classification: items + output         -> two binary judgments/item

It shares :class:`~idp_eval.evaluators.coverage_base._TwoStageCoverageEvaluator`
with :class:`~idp_eval.evaluators.source_coverage.SourceCoverageEvaluator` for
classification, id validation, scoring, and result shaping. Stage 1 never sees
the output; Stage 2 cannot change the item set. Coverage answers "did the output
OMIT important task-relevant information?" — faithfulness owns unsupported
additions.
"""

from __future__ import annotations

from idp_eval import tracing
from idp_eval.evaluators.coverage_base import _TwoStageCoverageEvaluator, _dedup
from idp_eval.models import EvaluationCase
from idp_eval.prompts.coverage_extract import (
    COVERAGE_EXTRACT_SCHEMA,
    render_coverage_extract_prompt,
)


class TaskCoverageEvaluator(_TwoStageCoverageEvaluator):
    """Coverage of the source information relevant to the requested task.

    Direction: ``input + context -> output``. ``input`` scopes which parts of the
    context are relevant. Higher score is better.
    """

    name = "task_coverage"
    _item_key = "requirement"
    _total_key = "total_requirements"
    _id_prefix = "r"
    _coverage_label = "Task coverage"
    _explanation_unit = "task-relevant requirements"
    _not_applicable_reason = (
        "No task-relevant requirements were identified in the supplied context."
    )

    def _extract_requirements(self, case: EvaluationCase) -> list[dict]:
        """Stage 1: task-relevant requirements from input + context (no output)."""
        prompt = render_coverage_extract_prompt(
            input_text=case.input,
            context=case.context,
        )
        with tracing.judge_span(
            f"{self.name}.extract",
            {"idp_eval.metric": self.name, "idp_eval.stage": "extract"},
        ):
            response = self._llm.generate_object(
                prompt=prompt,
                schema=COVERAGE_EXTRACT_SCHEMA,
            )
        raw = _dedup(response.get("requirements", []), "requirement")
        return [
            {"id": f"{self._id_prefix}{index}", "requirement": req["requirement"]}
            for index, req in enumerate(raw, start=1)
        ]
