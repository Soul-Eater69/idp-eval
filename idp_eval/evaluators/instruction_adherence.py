"""Two-stage, binary instruction-adherence evaluator.

Instruction adherence uses only ``EvaluationCase.instructions`` as its semantic
source. Stage 1 extracts distinct checkable instructions; Stage 2 classifies the
fixed instruction set against the generated output as followed or violated.
Python calculates the score as the fraction followed. The judge never emits a
numeric score.

The metric does not read ``case.input`` or ``case.context`` and never falls back
to either field. Missing instructions or an empty extraction produce
``score=None`` with ``label="not_applicable"``.
"""

from __future__ import annotations

import json

from idp_eval import tracing
from idp_eval.models import EvaluationCase, EvaluationResult, Evaluator
from idp_eval.prompts.instruction_adherence_classify import (
    INSTRUCTION_ADHERENCE_CLASSIFY_SCHEMA,
    render_instruction_adherence_classify_prompt,
)
from idp_eval.prompts.instruction_adherence_extract import (
    INSTRUCTION_ADHERENCE_EXTRACT_SCHEMA,
    render_instruction_adherence_extract_prompt,
)
from idp_eval.scoring import calculate_instruction_adherence, score_to_label


def _normalize_instruction(text: str) -> str:
    """Normalizes instruction text for exact-match deduplication."""
    return " ".join(text.lower().split())


def _dedup_instructions(instructions: list[dict]) -> list[dict]:
    """Removes normalized-exact duplicates while keeping the first entry."""
    seen: set[str] = set()
    deduped: list[dict] = []
    for entry in instructions:
        if not isinstance(entry, dict):
            raise ValueError("Extracted instruction entry must be an object.")
        text = entry.get("instruction")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                "Extracted instruction must contain a non-empty string."
            )
        cleaned = " ".join(text.split())
        key = _normalize_instruction(cleaned)
        if key in seen:
            continue
        seen.add(key)
        deduped.append({"instruction": cleaned})
    return deduped


class InstructionAdherenceEvaluator(Evaluator):
    """Two-stage binary judgment of explicit instruction adherence.

    Direction: ``instructions -> output``. Higher score is better.
    """

    name = "instruction_adherence"

    def __init__(self, llm):
        """Initializes the evaluator with the shared structured-output judge."""
        self._llm = llm

    def evaluate(self, case: EvaluationCase) -> EvaluationResult:
        """Evaluates instruction adherence for one case using up to two calls."""
        if not case.instructions or not case.instructions.strip():
            return self._not_applicable("No instructions were supplied on the case.")

        instructions = self._extract_instructions(case.instructions)
        if not instructions:
            return self._not_applicable("No meaningful instructions were found.")

        judgments = self._classify_instructions(instructions, case.output)
        items = self._build_items(instructions, judgments)
        score = calculate_instruction_adherence(items)
        followed_count = sum(item["status"] == "followed" for item in items)
        violated_count = len(items) - followed_count

        return EvaluationResult(
            metric=self.name,
            score=score,
            label=score_to_label(score),
            explanation=(
                f"{followed_count} of {len(items)} instructions were followed; "
                f"{violated_count} were violated."
            ),
            details={
                "instruction_count": len(items),
                "followed_count": followed_count,
                "violated_count": violated_count,
                "instructions": items,
            },
        )

    def _extract_instructions(self, instructions: str) -> list[dict]:
        """Stage 1: extracts deduplicated instructions and assigns stable IDs."""
        prompt = render_instruction_adherence_extract_prompt(instructions)
        with tracing.judge_span(
            "instruction_adherence.extract",
            {"idp_eval.metric": self.name, "idp_eval.stage": "extract"},
        ):
            response = self._llm.generate_object(
                prompt=prompt,
                schema=INSTRUCTION_ADHERENCE_EXTRACT_SCHEMA,
            )

        if not isinstance(response, dict):
            raise ValueError("Instruction extraction response must be an object.")
        raw = response.get("instructions")
        if not isinstance(raw, list):
            raise ValueError(
                "Instruction extraction response must contain an instructions list."
            )
        deduped = _dedup_instructions(raw)
        return [
            {"id": f"I{index}", "instruction": entry["instruction"]}
            for index, entry in enumerate(deduped, start=1)
        ]

    def _classify_instructions(
        self, instructions: list[dict], output: str
    ) -> list[dict]:
        """Stage 2: classifies all fixed instructions in one judge call."""
        prompt = render_instruction_adherence_classify_prompt(
            instructions_json=json.dumps(instructions, ensure_ascii=False),
            output=output,
        )
        with tracing.judge_span(
            "instruction_adherence.classify",
            {"idp_eval.metric": self.name, "idp_eval.stage": "classify"},
        ):
            response = self._llm.generate_object(
                prompt=prompt,
                schema=INSTRUCTION_ADHERENCE_CLASSIFY_SCHEMA,
            )

        if not isinstance(response, dict):
            raise ValueError("Instruction classification response must be an object.")
        answers = response.get("answers")
        if not isinstance(answers, list):
            raise ValueError(
                "Instruction classification response must contain an answers list."
            )
        return answers

    @staticmethod
    def _build_items(
        instructions: list[dict], judgments: list[dict]
    ) -> list[dict]:
        """Validates exact ID correspondence and builds scored audit items."""
        by_id: dict[str, dict] = {}
        for judgment in judgments:
            if not isinstance(judgment, dict):
                raise ValueError("Instruction judgment must be an object.")
            instruction_id = judgment.get("id")
            if not isinstance(instruction_id, str):
                raise ValueError("Instruction judgment must contain a string id.")
            if instruction_id in by_id:
                raise ValueError(
                    f"Duplicate instruction id in classification: {instruction_id!r}"
                )
            by_id[instruction_id] = judgment

        expected = {entry["id"] for entry in instructions}
        unknown = sorted(set(by_id) - expected)
        if unknown:
            raise ValueError(
                f"Unknown instruction id(s) in classification: {unknown}"
            )
        missing = sorted(expected - set(by_id))
        if missing:
            raise ValueError(
                f"Missing classification for instruction id(s): {missing}"
            )

        items: list[dict] = []
        for instruction in instructions:
            judgment = by_id[instruction["id"]]
            status = judgment.get("status")
            if status not in {"followed", "violated"}:
                raise ValueError(
                    f"Unknown instruction-adherence status: {status!r}"
                )
            reason = judgment.get("reason")
            if not isinstance(reason, str):
                raise ValueError(
                    f"Instruction judgment reason for {instruction['id']!r} "
                    "must be a string."
                )
            items.append(
                {
                    "id": instruction["id"],
                    "instruction": instruction["instruction"],
                    "status": status,
                    "score": 1.0 if status == "followed" else 0.0,
                    "reason": reason,
                }
            )
        return items

    def _not_applicable(self, explanation: str) -> EvaluationResult:
        """Builds a metric-level result when there is nothing to score."""
        return EvaluationResult(
            metric=self.name,
            score=None,
            label="not_applicable",
            explanation=explanation,
            details={
                "instruction_count": 0,
                "followed_count": 0,
                "violated_count": 0,
                "instructions": [],
            },
        )
