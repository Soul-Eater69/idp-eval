"""Hallucination evaluator.

The judge classifies each claim in the output as supported, unsupported, or
contradicted. Python then calculates the unsupported-claim ratio.
"""

from __future__ import annotations

from idp_eval.models import EvaluationCase, EvaluationResult, Evaluator
from idp_eval.prompts.hallucination import (
    HALLUCINATION_PROMPT,
    HALLUCINATION_SCHEMA,
)
from idp_eval.scoring import (
    HALLUCINATED_STATUSES,
    calculate_hallucination_score,
    score_to_label,
)


class HallucinationEvaluator(Evaluator):
    """Detailed hallucination detection.

    Answers: which claims in the output are not supported by the context?
    Lower score is better.
    """

    name = "hallucination"

    def __init__(self, llm):
        """Initializes the evaluator.

        Args:
            llm: A judge object exposing
                ``generate_object(prompt: str, schema: dict) -> dict``. Phoenix's
                ``LLM`` satisfies this contract.
        """
        self._llm = llm

    def evaluate(self, case: EvaluationCase) -> EvaluationResult:
        """Evaluates hallucination for a single case."""
        prompt = HALLUCINATION_PROMPT.format(
            input=case.input,
            context=case.context,
            output=case.output,
        )

        response = self._llm.generate_object(
            prompt=prompt,
            schema=HALLUCINATION_SCHEMA,
        )
        claims = response.get("claims", [])

        score = calculate_hallucination_score(claims)
        unsupported_claims = [
            claim["claim"]
            for claim in claims
            if claim["status"] in HALLUCINATED_STATUSES
        ]

        # Lower hallucination is better, so invert before bucketing the label.
        label = score_to_label(1.0 - score)

        return EvaluationResult(
            metric=self.name,
            score=score,
            label=label,
            explanation=(
                f"{len(unsupported_claims)} of {len(claims)} claims are not "
                "supported by the context."
            ),
            details={
                "unsupported_claims": unsupported_claims,
                "claims": claims,
            },
        )
