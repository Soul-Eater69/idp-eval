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

from idp_eval import tracing
from idp_eval.models import EvaluationCase, EvaluationResult, Evaluator


class FaithfulnessEvaluator(Evaluator):
    """Grounding evaluation backed by Phoenix.

    A thin adapter around Phoenix's built-in ``FaithfulnessEvaluator`` (imported
    here as ``PhoenixFaithfulnessEvaluator``) that returns our common
    ``EvaluationResult``.

    Answers: is the generated output grounded in the provided context, or did it
    ADD unsupported (hallucinated) information? Direction: ``output -> context``.
    Higher is better.
    """

    name = "faithfulness"

    def __init__(self, llm):
        """Initializes the evaluator.

        Args:
            llm: A Phoenix ``LLM`` (or compatible) judge object.
        """
        # Imported lazily so the rest of the framework (models, scoring) can be
        # used and tested without Phoenix installed. Aliased to avoid colliding
        # with our own class of the same name.
        from phoenix.evals.metrics import (
            FaithfulnessEvaluator as PhoenixFaithfulnessEvaluator,
        )

        self._evaluator = PhoenixFaithfulnessEvaluator(llm=llm)

    def evaluate(self, case: EvaluationCase) -> EvaluationResult:
        """Evaluates grounding for a single case."""
        with tracing.judge_span(
            "faithfulness.evaluate", {"idp_eval.metric": self.name}
        ):
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
