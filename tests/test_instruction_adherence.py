"""Instruction-adherence evaluator tests using a fake judge (no real LLM)."""

import copy

import pytest

from idp_eval import (
    EvaluationCase,
    EvaluationFramework,
    InstructionAdherenceEvaluator,
)
from idp_eval.models import EvaluationResult
from idp_eval.prompts.instruction_adherence import (
    INSTRUCTION_ADHERENCE_PROMPT,
    INSTRUCTION_ADHERENCE_SCHEMA,
    render_instruction_adherence_prompt,
)
from idp_eval.scoring import calculate_instruction_adherence


class FakeJudge:
    """Judge stub returning a canned structured response."""

    def __init__(self, response: dict):
        self.response = response
        self.calls: list[dict] = []

    def generate_object(self, prompt, schema: dict) -> dict:
        self.calls.append({"prompt": prompt, "schema": schema})
        return self.response


def _judge(*statuses: str) -> FakeJudge:
    return FakeJudge(
        {
            "instructions": [
                {"instruction": f"instr {i}", "status": s, "reason": "r"}
                for i, s in enumerate(statuses)
            ]
        }
    )


CASE = EvaluationCase(
    input="Use exactly 3 bullet points.\nDo not mention customer names.",
    context="",
    output="- a\n- b\n- c",
)


# --- scoring / status behavior ----------------------------------------------


def test_all_followed():
    result = InstructionAdherenceEvaluator(llm=_judge("followed", "followed")).evaluate(
        CASE
    )
    assert result.metric == "instruction_adherence"
    assert result.score == 1.0
    assert result.label == "high"


def test_one_violated():
    result = InstructionAdherenceEvaluator(
        llm=_judge("followed", "followed", "followed", "violated")
    ).evaluate(CASE)
    assert result.score == 0.75
    assert result.details["violated_instructions"] == ["instr 3"]


def test_partial():
    result = InstructionAdherenceEvaluator(llm=_judge("partial")).evaluate(CASE)
    assert result.score == 0.5
    assert result.details["partial_instructions"] == ["instr 0"]


def test_mixed():
    result = InstructionAdherenceEvaluator(
        llm=_judge("followed", "followed", "partial", "violated")
    ).evaluate(CASE)
    assert result.score == 0.625


def test_returns_evaluation_result():
    result = InstructionAdherenceEvaluator(llm=_judge("followed")).evaluate(CASE)
    assert isinstance(result, EvaluationResult)


# --- empty / not-applicable behavior ----------------------------------------


def test_no_instructions_blank_input_is_not_applicable():
    judge = _judge("followed")  # would score 1.0 if it were ever called
    case = EvaluationCase(input="   ", context="", output="whatever")
    result = InstructionAdherenceEvaluator(llm=judge).evaluate(case)
    assert result.score is None
    assert result.label == "not_applicable"
    # The judge must not be consulted when there are no instructions.
    assert judge.calls == []


def test_judge_finds_no_instructions_is_not_applicable():
    judge = FakeJudge({"instructions": []})
    result = InstructionAdherenceEvaluator(llm=judge).evaluate(CASE)
    assert result.score is None
    assert result.label == "not_applicable"
    assert result.explanation == "No meaningful instructions were found in input."


def test_not_applicable_excluded_from_denominator():
    # 2 applicable (followed, violated) -> 0.5; the not_applicable one is ignored.
    result = InstructionAdherenceEvaluator(
        llm=_judge("followed", "violated", "not_applicable")
    ).evaluate(CASE)
    assert result.score == 0.5
    assert result.details["not_applicable_instructions"] == ["instr 2"]


def test_mixed_with_not_applicable():
    # followed, followed, partial over 3 applicable = 2.5/3; one not_applicable.
    result = InstructionAdherenceEvaluator(
        llm=_judge("followed", "followed", "partial", "not_applicable")
    ).evaluate(CASE)
    assert result.score == 2.5 / 3


def test_all_not_applicable_is_metric_not_applicable():
    judge = _judge("not_applicable", "not_applicable")
    result = InstructionAdherenceEvaluator(llm=judge).evaluate(CASE)
    assert result.score is None
    assert result.label == "not_applicable"
    assert result.explanation == (
        "All supplied instructions were not applicable to this case."
    )
    # The full instruction list is still surfaced for debugging.
    assert len(result.details["instructions"]) == 2


# --- instruction kinds ------------------------------------------------------


def test_negative_instruction_violated():
    judge = FakeJudge(
        {
            "instructions": [
                {
                    "instruction": "Do not mention customer names.",
                    "status": "violated",
                    "reason": "Output names 'Acme Corp'.",
                }
            ]
        }
    )
    result = InstructionAdherenceEvaluator(llm=judge).evaluate(CASE)
    assert result.score == 0.0
    assert result.details["violated_instructions"] == ["Do not mention customer names."]


def test_formatting_count_instruction_followed():
    judge = FakeJudge(
        {
            "instructions": [
                {
                    "instruction": "Use exactly 3 bullet points.",
                    "status": "followed",
                    "reason": "Exactly three bullets present.",
                }
            ]
        }
    )
    result = InstructionAdherenceEvaluator(llm=judge).evaluate(CASE)
    assert result.score == 1.0


def test_semantic_instruction_partial():
    judge = FakeJudge(
        {
            "instructions": [
                {
                    "instruction": "Use a professional tone.",
                    "status": "partial",
                    "reason": "Mostly professional but one casual aside.",
                }
            ]
        }
    )
    result = InstructionAdherenceEvaluator(llm=judge).evaluate(CASE)
    assert result.score == 0.5


def test_conditional_instruction_condition_applies():
    # The condition depends on context; verify context reaches the judge prompt
    # so it can decide whether the instruction applies. Here it applies.
    case = EvaluationCase(
        input="If the account is inactive, include a warning.",
        context="Account status: inactive.",
        output="Warning: this account is inactive.",
    )
    judge = FakeJudge(
        {
            "instructions": [
                {
                    "instruction": "If the account is inactive, include a warning.",
                    "status": "followed",
                    "reason": "Account is inactive and a warning is present.",
                }
            ]
        }
    )
    result = InstructionAdherenceEvaluator(llm=judge).evaluate(case)
    assert result.score == 1.0
    user_content = judge.calls[0]["prompt"][1]["content"]
    assert "Account status: inactive." in user_content


def test_conditional_instruction_condition_does_not_apply():
    # Condition is false (account active) -> the single instruction is
    # not_applicable, so the metric has nothing applicable to score.
    case = EvaluationCase(
        input="If the account is inactive, include a warning.",
        context="Account status: active.",
        output="Your account summary is ready.",
    )
    judge = FakeJudge(
        {
            "instructions": [
                {
                    "instruction": "If the account is inactive, include a warning.",
                    "status": "not_applicable",
                    "reason": "The account is active, so the condition does not apply.",
                }
            ]
        }
    )
    result = InstructionAdherenceEvaluator(llm=judge).evaluate(case)
    assert result.score is None
    assert result.label == "not_applicable"


# --- prompt / schema structure ----------------------------------------------


def test_schema_requires_reason_and_allows_not_applicable():
    item = INSTRUCTION_ADHERENCE_SCHEMA["properties"]["instructions"]["items"]
    assert item["required"] == ["instruction", "status", "reason"]
    assert item["properties"]["status"]["enum"] == [
        "followed",
        "partial",
        "violated",
        "not_applicable",
    ]


def test_scoring_helper_raises_without_applicable():
    with pytest.raises(ValueError):
        calculate_instruction_adherence([{"status": "not_applicable"}])


# --- prompt structure -------------------------------------------------------


def test_prompt_is_message_list_and_scoped():
    judge = _judge("followed")
    InstructionAdherenceEvaluator(llm=judge).evaluate(CASE)
    prompt = judge.calls[0]["prompt"]
    assert [m["role"] for m in prompt] == ["system", "user"]
    system = prompt[0]["content"]
    assert "Instruction Adherence measures" in system
    user = prompt[1]["content"]
    assert "[INSTRUCTIONS]" in user and CASE.input in user
    assert "[OUTPUT]" in user and CASE.output in user


def test_render_does_not_mutate_global_template():
    before = copy.deepcopy(INSTRUCTION_ADHERENCE_PROMPT)
    render_instruction_adherence_prompt(input_text="a", context="b", output="c")
    assert INSTRUCTION_ADHERENCE_PROMPT == before
    assert "{input}" in INSTRUCTION_ADHERENCE_PROMPT[1]["content"]


# --- framework integration --------------------------------------------------


def test_framework_runs_only_instruction_adherence():
    framework = EvaluationFramework(
        evaluators=[InstructionAdherenceEvaluator(llm=_judge("followed"))]
    )
    results = framework.evaluate(CASE, metrics=["instruction_adherence"])
    assert set(results) == {"instruction_adherence"}
    assert results["instruction_adherence"].score == 1.0
