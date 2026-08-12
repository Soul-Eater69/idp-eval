"""Coverage evaluator.

The judge identifies the material source items in the context that are relevant
to the requested task and classifies each as covered, partial, or missing in the
output. Python calculates the weighted coverage score. Higher is better.

Coverage answers the "did the output OMIT important relevant information?"
question. The complementary "did the output ADD unsupported information?"
question is handled by faithfulness.
"""

from __future__ import annotations

from idp_eval.models import EvaluationCase, EvaluationResult, Evaluator
from idp_eval.prompts.coverage import COVERAGE_PROMPT, COVERAGE_SCHEMA
from idp_eval.scoring import calculate_coverage, score_to_label


class CoverageEvaluator(Evaluator):
    """Semantic coverage of task-relevant context.

    Answers: how much of the material, task-relevant information from the
    context is represented in the output? Direction: ``context -> output``.
    Higher score is better.
    """

    name = "coverage"

    def __init__(self, llm):
        """Initializes the evaluator.

        Args:
            llm: A judge object exposing
                ``generate_object(prompt: str, schema: dict) -> dict``. Phoenix's
                ``LLM`` satisfies this contract.
        """
        self._llm = llm

    def evaluate(self, case: EvaluationCase) -> EvaluationResult:
        """Evaluates coverage for a single case."""
        prompt = COVERAGE_PROMPT.format(
            input=case.input,
            context=case.context,
            output=case.output,
        )

        response = self._llm.generate_object(
            prompt=prompt,
            schema=COVERAGE_SCHEMA,
        )
        items = response.get("items", [])

        score = calculate_coverage(items)
        missing_items = [
            item["source_item"]
            for item in items
            if item["status"] == "missing"
        ]

        return EvaluationResult(
            metric=self.name,
            score=score,
            label=score_to_label(score),
            explanation=(
                f"{len(missing_items)} of {len(items)} relevant source items "
                "are missing from the output."
            ),
            details={
                "missing_items": missing_items,
                "items": items,
            },
        )
