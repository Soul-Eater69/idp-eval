"""Faithfulness evaluator.

For v1 this wraps Phoenix's existing ``FaithfulnessEvaluator`` so we get a
baseline grounding metric without writing our own implementation.
"""

from __future__ import annotations

from idp_eval.models import EvaluationCase, EvaluationResult, Evaluator


class FaithfulnessMetric(Evaluator):
    """Holistic grounding evaluation backed by Phoenix.

    Answers: is the generated output grounded in the provided context?
    Direction: ``output -> context``.
    """

    name = "faithfulness"

    def __init__(self, llm):
        """Initializes the metric.

        Args:
            llm: A Phoenix ``LLM`` (or compatible) judge object.
        """
        # Imported lazily so the rest of the framework (models, scoring) can be
        # used and tested without Phoenix installed.
        from phoenix.evals.metrics import FaithfulnessEvaluator

        self._evaluator = FaithfulnessEvaluator(llm=llm)

    def evaluate(self, case: EvaluationCase) -> EvaluationResult:
        """Evaluates grounding for a single case."""
        result = self._evaluator.evaluate(
            {
                "input": case.input,
                "context": case.context,
                "output": case.output,
            }
        )[0]

        return EvaluationResult(
            metric=self.name,
            score=result.score,
            label=result.label,
            explanation=result.explanation,
            details=None,
        )
