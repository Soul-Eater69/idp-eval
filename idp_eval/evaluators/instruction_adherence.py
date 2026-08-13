"""Instruction-adherence evaluator.

The judge decomposes the explicit instructions supplied in
``EvaluationCase.instructions`` into atomic instructions and classifies each as
followed, partial, violated, or not_applicable in the output. Python calculates
the score. Higher is better.

This metric reads ``case.instructions`` (the dedicated instruction field), never
``case.input``. ``context`` is optional supporting information consulted only
when an instruction requires it. Direction: ``instructions -> output``.

``not_applicable`` instructions (e.g. an untriggered conditional) are excluded
from scoring entirely. The metric returns ``score=None`` with
``label="not_applicable"`` when there is nothing applicable to evaluate. Three
distinct situations produce that result, distinguished by their explanation:
    - no instructions supplied on the case;
    - the judge found no meaningful instructions;
    - every supplied instruction was not applicable to this case.
None of these is treated as a perfect (1.0) or failing (0.0) score.
"""

from __future__ import annotations

from idp_eval.models import EvaluationCase, EvaluationResult, Evaluator
from idp_eval.prompts.instruction_adherence import (
    INSTRUCTION_ADHERENCE_SCHEMA,
    render_instruction_adherence_prompt,
)
from idp_eval.scoring import calculate_instruction_adherence, score_to_label


class InstructionAdherenceEvaluator(Evaluator):
    """Semantic instruction-adherence judgment.

    Answers: how well does the output obey the explicit instructions provided in
    ``instructions``? Direction: ``instructions -> output``. Higher score is
    better.
    """

    name = "instruction_adherence"

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
        """Evaluates instruction adherence for a single case."""
        # No instructions supplied: the metric does not apply. Short-circuit
        # before calling the judge so it cannot invent instructions. This reads
        # the dedicated instructions field, never falling back to input.
        if not case.instructions or not case.instructions.strip():
            return self._not_applicable(
                "No instructions were supplied on the case.", instructions=[]
            )

        prompt = render_instruction_adherence_prompt(
            instructions=case.instructions,
            context=case.context,
            output=case.output,
        )

        response = self._llm.generate_object(
            prompt=prompt,
            schema=INSTRUCTION_ADHERENCE_SCHEMA,
        )
        instructions = response.get("instructions", [])

        # The judge found no meaningful instructions to evaluate.
        if not instructions:
            return self._not_applicable(
                "No meaningful instructions were found.", instructions=[]
            )

        applicable = [
            i for i in instructions if i["status"] != "not_applicable"
        ]

        # Every supplied instruction turned out not to apply to this case.
        if not applicable:
            return self._not_applicable(
                "All supplied instructions were not applicable to this case.",
                instructions=instructions,
            )

        score = calculate_instruction_adherence(instructions)
        violated = [i["instruction"] for i in instructions if i["status"] == "violated"]
        partial = [i["instruction"] for i in instructions if i["status"] == "partial"]
        followed = [i["instruction"] for i in instructions if i["status"] == "followed"]
        not_applicable = [
            i["instruction"] for i in instructions if i["status"] == "not_applicable"
        ]

        return EvaluationResult(
            metric=self.name,
            score=score,
            label=score_to_label(score),
            explanation=(
                f"{len(violated)} violated, {len(partial)} partial, and "
                f"{len(followed)} followed of {len(applicable)} applicable "
                f"instructions. {len(not_applicable)} additional instruction(s) "
                "were not applicable."
            ),
            details={
                "violated_instructions": violated,
                "partial_instructions": partial,
                "not_applicable_instructions": not_applicable,
                "instructions": instructions,
            },
        )

    def _not_applicable(
        self, explanation: str, instructions: list[dict]
    ) -> EvaluationResult:
        """Builds a not-applicable result (nothing applicable to score)."""
        return EvaluationResult(
            metric=self.name,
            score=None,
            label="not_applicable",
            explanation=explanation,
            details={"instructions": instructions},
        )
