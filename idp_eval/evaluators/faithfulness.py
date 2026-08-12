"""Faithfulness evaluator.

For v1 this wraps Phoenix's existing ``FaithfulnessEvaluator`` so we get a
grounding metric without writing our own implementation.

Faithfulness is the metric we use to evaluate whether the generated output
contains hallucinated / unsupported information relative to the authoritative
context. Hallucination is the failure/problem being measured; faithfulness is
the metric that detects it. There is deliberately no separate top-level
``hallucination`` metric in v1. Higher is better.
"""

from __future__ import annotations

from idp_eval.models import EvaluationCase, EvaluationResult, Evaluator


class FaithfulnessMetric(Evaluator):
    """Grounding evaluation backed by Phoenix.

    Answers: is the generated output grounded in the provided context, or did it
    ADD unsupported (hallucinated) information? Direction: ``output -> context``.
    Higher is better.
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
