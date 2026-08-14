"""Coverage evaluator.

Recall-style, auditable coverage:

    input + context -> derive task-relevant atomic requirements
                    -> classify each requirement against output
                    -> covered / partial / missing
                    -> deterministic numeric mapping (Python)
                    -> aggregate 0.0-1.0 score

The judge only performs semantic decomposition and item-level classification.
Python maps ``covered/partial/missing`` to ``1.0/0.5/0.0`` and averages them, so
the aggregate score is reproducible and every requirement is explained.

Coverage answers "did the output OMIT important task-relevant information?" The
complementary "did the output ADD unsupported information?" question is handled
by faithfulness; coverage never performs hallucination detection.
"""

from __future__ import annotations

from idp_eval.models import EvaluationCase, EvaluationResult, Evaluator
from idp_eval.prompts.coverage import COVERAGE_SCHEMA, render_coverage_prompt
from idp_eval.scoring import (
    calculate_coverage,
    coverage_status_score,
    score_to_label,
)


class CoverageEvaluator(Evaluator):
    """Semantic coverage of task-relevant context.

    Answers: how completely does the output represent the task-relevant
    information in the context? Direction: ``input + context -> output``. Higher
    score is better.
    """

    name = "coverage"

    def __init__(self, llm):
        """Initializes the evaluator.

        Args:
            llm: A judge object exposing
                ``generate_object(prompt, schema: dict) -> dict`` where ``prompt``
                is a Phoenix-style message list (``[{"role", "content"}, ...]``).
                Phoenix's ``LLM`` satisfies this contract.
        """
        self._llm = llm

    def evaluate(self, case: EvaluationCase) -> EvaluationResult:
        """Evaluates coverage for a single case."""
        prompt = render_coverage_prompt(
            input_text=case.input,
            context=case.context,
            output=case.output,
        )

        response = self._llm.generate_object(
            prompt=prompt,
            schema=COVERAGE_SCHEMA,
        )
        requirements = response.get("requirements", [])

        # Attach the deterministic per-item score in Python (never the LLM).
        # Unknown statuses raise a clear ValueError via coverage_status_score.
        items = [
            {
                "requirement": req.get("requirement", ""),
                "status": req["status"],
                "score": coverage_status_score(req["status"]),
                "reason": req.get("reason", ""),
            }
            for req in requirements
        ]

        score = calculate_coverage(requirements)

        covered = [i for i in items if i["status"] == "covered"]
        partial = [i for i in items if i["status"] == "partial"]
        missing = [i for i in items if i["status"] == "missing"]
        total = len(items)

        if total == 0:
            explanation = (
                "No task-relevant requirements were identified in the supplied "
                "context."
            )
        else:
            percentage = f"{round(score * 100, 1):g}"
            explanation = (
                f"Coverage was {percentage}% across {total} task-relevant "
                f"requirements: {len(covered)} covered, {len(partial)} partial, "
                f"and {len(missing)} missing."
            )

        return EvaluationResult(
            metric=self.name,
            score=score,
            label=score_to_label(score),
            explanation=explanation,
            details={
                "total_requirements": total,
                "covered_count": len(covered),
                "partial_count": len(partial),
                "missing_count": len(missing),
                "items": items,
            },
        )
