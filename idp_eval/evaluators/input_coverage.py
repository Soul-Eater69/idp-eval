"""Input coverage evaluator.

The judge identifies the important source items in the context and classifies
each as covered, partial, or missing in the output. Python calculates the
weighted coverage score. Higher is better.
"""

from __future__ import annotations

from idp_eval.models import EvaluationCase, EvaluationResult, Evaluator
from idp_eval.prompts.input_coverage import (
    INPUT_COVERAGE_PROMPT,
    INPUT_COVERAGE_SCHEMA,
)
from idp_eval.scoring import calculate_coverage, score_to_label


class InputCoverageEvaluator(Evaluator):
    """Semantic source coverage.

    Answers: how much important information from the context is represented in
    the output? Direction: ``context -> output``. Higher score is better.
    """

    name = "input_coverage"

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
        prompt = INPUT_COVERAGE_PROMPT.format(
            input=case.input,
            context=case.context,
            output=case.output,
        )

        response = self._llm.generate_object(
            prompt=prompt,
            schema=INPUT_COVERAGE_SCHEMA,
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
                f"{len(missing_items)} of {len(items)} important source items "
                "are missing from the output."
            ),
            details={
                "missing_items": missing_items,
                "items": items,
            },
        )
