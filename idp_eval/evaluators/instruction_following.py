"""Instruction-following evaluator.

The judge decomposes the explicit instructions supplied in ``EvaluationCase.input``
into atomic instructions and classifies each as followed, partial, or violated in
the output. Python calculates the score. Higher is better.

For this metric, ``input`` contains ONLY the instructions to evaluate (not the
full generation prompt). ``context`` is optional supporting information consulted
only when an instruction requires it. Direction: ``instructions -> output``.

When no meaningful instructions are supplied, the metric is not applicable: it
returns ``score=None`` with ``label="not_applicable"`` rather than inventing
instructions or reporting a perfect score.
"""

from __future__ import annotations

from idp_eval.models import EvaluationCase, EvaluationResult, Evaluator
from idp_eval.prompts.instruction_following import (
    INSTRUCTION_FOLLOWING_SCHEMA,
    render_instruction_following_prompt,
)
from idp_eval.scoring import calculate_instruction_following, score_to_label


class InstructionFollowingEvaluator(Evaluator):
    """Semantic instruction-following judgment.

    Answers: how well does the output obey the explicit instructions provided in
    ``input``? Direction: ``instructions -> output``. Higher score is better.
    """

    name = "instruction_following"

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
        """Evaluates instruction following for a single case."""
        # No instructions supplied: the metric does not apply. Short-circuit
        # before calling the judge so it cannot invent instructions.
        if not case.input or not case.input.strip():
            return self._not_applicable(
                "No instructions were supplied in input."
            )

        prompt = render_instruction_following_prompt(
            input_text=case.input,
            context=case.context,
            output=case.output,
        )

        response = self._llm.generate_object(
            prompt=prompt,
            schema=INSTRUCTION_FOLLOWING_SCHEMA,
        )
        instructions = response.get("instructions", [])

        # The judge found no meaningful instructions to evaluate.
        if not instructions:
            return self._not_applicable(
                "No meaningful instructions were found in input."
            )

        score = calculate_instruction_following(instructions)
        violated = [
            i["instruction"] for i in instructions if i["status"] == "violated"
        ]
        partial = [
            i["instruction"] for i in instructions if i["status"] == "partial"
        ]

        return EvaluationResult(
            metric=self.name,
            score=score,
            label=score_to_label(score),
            explanation=(
                f"{len(violated)} violated and {len(partial)} partially "
                f"followed of {len(instructions)} instructions."
            ),
            details={
                "violated_instructions": violated,
                "partial_instructions": partial,
                "instructions": instructions,
            },
        )

    def _not_applicable(self, explanation: str) -> EvaluationResult:
        """Builds a not-applicable result for the empty-instruction case."""
        return EvaluationResult(
            metric=self.name,
            score=None,
            label="not_applicable",
            explanation=explanation,
            details={"instructions": []},
        )
