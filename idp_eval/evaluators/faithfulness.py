"""Faithfulness measures whether output is supported by authoritative context."""

from __future__ import annotations

from idp_eval import tracing
from idp_eval.models import EvaluationCase, EvaluationResult, Evaluator
from idp_eval.rendering import render_value


class FaithfulnessEvaluator(Evaluator):
    """Thin Phoenix adapter for the ``output -> context`` grounding metric."""

    name = "faithfulness"
    required_fields = ("context", "output")

    def __init__(self, llm):
        # Imported lazily so the rest of the framework (models, scoring) can be
        # used and tested without Phoenix installed. Aliased to avoid colliding
        # with our own class of the same name.
        from phoenix.evals.metrics import (
            FaithfulnessEvaluator as PhoenixFaithfulnessEvaluator,
        )

        self._evaluator = PhoenixFaithfulnessEvaluator(llm=llm)

    def evaluate(self, case: EvaluationCase) -> EvaluationResult:
        """Evaluates whether output claims are supported by context."""
        self.validate_case(case)
        with tracing.judge_span(
            "faithfulness.evaluate",
            {"idp_eval.metric": self.name, "idp_eval.stage": "evaluate"},
        ):
            result = self._evaluator.evaluate(
                {
                    # Phoenix Evals 3.4 requires an input string structurally and
                    # accepts empty text. Keep it neutral so case.input cannot
                    # influence faithfulness semantics.
                    "input": "",
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
