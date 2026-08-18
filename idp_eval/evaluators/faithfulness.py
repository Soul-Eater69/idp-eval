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
from idp_eval.rendering import render_value


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
    required_fields = ("context", "output")

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
        """Evaluates grounding for a single case.

        Validates required fields first, so a missing ``context`` / ``output``
        fails before the Phoenix judge call — consistent with the framework.
        """
        self.validate_case(case)
        with tracing.judge_span(
            "faithfulness.evaluate", {"idp_eval.metric": self.name}
        ):
            result = self._evaluator.evaluate(
                {
                    "input": render_value(case.input),
                    "context": render_value(case.context),
                    "output": render_value(case.output),
                }
            )[0]

        return EvaluationResult(
            metric=self.name,
            score=result.score,
            label=result.label,
            explanation=result.explanation,
            details=None,
        )
