"""Offline tests for two-stage binary instruction adherence."""

import copy

import pytest

from idp_eval import (
    EvaluationCase,
    EvaluationFramework,
    InstructionAdherenceEvaluator,
)
from idp_eval.models import EvaluationResult
from idp_eval.prompts.instruction_adherence_classify import (
    INSTRUCTION_ADHERENCE_CLASSIFY_PROMPT,
    INSTRUCTION_ADHERENCE_CLASSIFY_SCHEMA,
    render_instruction_adherence_classify_prompt,
)
from idp_eval.prompts.instruction_adherence_extract import (
    INSTRUCTION_ADHERENCE_EXTRACT_PROMPT,
    INSTRUCTION_ADHERENCE_EXTRACT_SCHEMA,
    render_instruction_adherence_extract_prompt,
)


class ScriptedJudge:
    """Returns queued structured responses or exceptions without network calls."""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def generate_object(self, prompt, schema: dict) -> dict:
        self.calls.append({"prompt": prompt, "schema": schema})
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _extraction(*instructions: str) -> dict:
    return {
        "instructions": [
            {"instruction": instruction} for instruction in instructions
        ]
    }


def _classification(*statuses: str) -> dict:
    return {
        "answers": [
            {"id": f"I{index}", "status": status, "reason": "reason"}
            for index, status in enumerate(statuses, start=1)
        ]
    }


def _judge(instructions: list[str], statuses: list[str]) -> ScriptedJudge:
    return ScriptedJudge(_extraction(*instructions), _classification(*statuses))


CASE = EvaluationCase(
    input="UNIQUE TASK INPUT THAT MUST NOT REACH THIS METRIC",
    instructions=(
        "Return JSON. Include exactly 3 recommendations. Do not mention pricing."
    ),
    context="UNIQUE CONTEXT THAT MUST NOT REACH THIS METRIC",
    output='{"recommendations": ["a", "b", "c"]}',
)


def test_all_followed_uses_two_calls_and_returns_audit_details():
    judge = _judge(
        ["Return JSON.", "Include exactly 3 recommendations."],
        ["followed", "followed"],
    )
    result = InstructionAdherenceEvaluator(judge).evaluate(CASE)

    assert isinstance(result, EvaluationResult)
    assert result.metric == "instruction_adherence"
    assert result.score == 1.0
    assert result.label == "fully_followed"
    assert result.details == {
        "instruction_count": 2,
        "followed_count": 2,
        "violated_count": 0,
        "instructions": [
            {
                "id": "I1",
                "instruction": "Return JSON.",
                "status": "followed",
                "score": 1.0,
                "reason": "reason",
            },
            {
                "id": "I2",
                "instruction": "Include exactly 3 recommendations.",
                "status": "followed",
                "score": 1.0,
                "reason": "reason",
            },
        ],
    }
    assert len(judge.calls) == 2


def test_mixed_binary_score_is_fraction_followed():
    judge = _judge(
        ["Return JSON.", "Include exactly 3 items.", "Do not mention pricing."],
        ["followed", "followed", "violated"],
    )
    result = InstructionAdherenceEvaluator(judge).evaluate(CASE)
    assert result.score == pytest.approx(2 / 3)
    # 2/3 followed with one violation is "violations_present", never "high".
    assert result.label == "violations_present"
    assert result.details["followed_count"] == 2
    assert result.details["violated_count"] == 1


def test_all_violated_scores_zero():
    result = InstructionAdherenceEvaluator(
        _judge(["Return JSON.", "Respond in Spanish."], ["violated", "violated"])
    ).evaluate(CASE)
    assert result.score == 0.0
    assert result.label == "violated"


@pytest.mark.parametrize("instructions", [None, "", "   \n\t"])
def test_no_instructions_is_not_applicable_without_judge_call(instructions):
    judge = ScriptedJudge(AssertionError("judge must not be called"))
    case = EvaluationCase(
        input="Return exactly 3 bullet points.",
        instructions=instructions,
        context="context",
        output="- one",
    )
    result = InstructionAdherenceEvaluator(judge).evaluate(case)
    assert result.score is None
    assert result.label == "not_applicable"
    assert result.details == {
        "instruction_count": 0,
        "followed_count": 0,
        "violated_count": 0,
        "instructions": [],
    }
    assert judge.calls == []


def test_empty_extraction_is_not_applicable_and_skips_classification():
    judge = ScriptedJudge(_extraction())
    result = InstructionAdherenceEvaluator(judge).evaluate(CASE)
    assert result.score is None
    assert result.label == "not_applicable"
    assert result.explanation == "No meaningful instructions were found."
    assert len(judge.calls) == 1


def test_compound_instruction_can_split_into_independent_items():
    judge = _judge(
        ["Return JSON.", "Do not mention internal IDs."],
        ["followed", "violated"],
    )
    case = EvaluationCase(
        input="ignored",
        instructions="Return JSON and do not mention internal IDs.",
        context="ignored",
        output='{"internal_id": 7}',
    )
    result = InstructionAdherenceEvaluator(judge).evaluate(case)
    assert [item["id"] for item in result.details["instructions"]] == ["I1", "I2"]
    assert result.score == 0.5


def test_material_qualifier_is_preserved_in_fixed_classification_set():
    judge = _judge(["Provide exactly 5 items."], ["violated"])
    InstructionAdherenceEvaluator(judge).evaluate(CASE)
    classify_user = judge.calls[1]["prompt"][1]["content"]
    assert "Provide exactly 5 items." in classify_user
    assert "exactly 5" in classify_user


def test_normalized_exact_duplicates_keep_first_and_receive_one_id():
    judge = ScriptedJudge(
        _extraction(
            "Return JSON.",
            "  return   json.  ",
            "RETURN JSON.",
        ),
        _classification("followed"),
    )
    result = InstructionAdherenceEvaluator(judge).evaluate(CASE)
    assert result.details["instruction_count"] == 1
    assert result.details["instructions"][0]["instruction"] == "Return JSON."
    assert '"id": "I1"' in judge.calls[1]["prompt"][1]["content"]


def test_reordered_classifications_are_reconstructed_in_extraction_order():
    judge = ScriptedJudge(
        _extraction("First.", "Second."),
        {
            "answers": [
                {"id": "I2", "status": "violated", "reason": "second"},
                {"id": "I1", "status": "followed", "reason": "first"},
            ]
        },
    )
    result = InstructionAdherenceEvaluator(judge).evaluate(CASE)
    assert [item["id"] for item in result.details["instructions"]] == ["I1", "I2"]


@pytest.mark.parametrize(
    ("answers", "message"),
    [
        (
            [{"id": "I1", "status": "followed", "reason": "r"}],
            "Missing classification",
        ),
        (
            [
                {"id": "I1", "status": "followed", "reason": "r"},
                {"id": "I1", "status": "violated", "reason": "r"},
            ],
            "Duplicate instruction id",
        ),
        (
            [
                {"id": "I1", "status": "followed", "reason": "r"},
                {"id": "I9", "status": "violated", "reason": "r"},
            ],
            "Unknown instruction id",
        ),
    ],
)
def test_stage_two_requires_exact_instruction_ids(answers, message):
    judge = ScriptedJudge(_extraction("First.", "Second."), {"answers": answers})
    with pytest.raises(ValueError, match=message):
        InstructionAdherenceEvaluator(judge).evaluate(CASE)


@pytest.mark.parametrize("status", ["partial", "sometimes", "not_applicable"])
def test_non_binary_status_fails_clearly(status):
    judge = ScriptedJudge(
        _extraction("Return JSON."),
        {"answers": [{"id": "I1", "status": status, "reason": "r"}]},
    )
    with pytest.raises(ValueError, match="Unknown instruction-adherence status"):
        InstructionAdherenceEvaluator(judge).evaluate(CASE)


@pytest.mark.parametrize(
    "response",
    [
        None,
        {},
        {"instructions": "not a list"},
        {"instructions": ["bad"]},
        _extraction(""),
    ],
)
def test_malformed_extraction_fails_clearly(response):
    with pytest.raises(ValueError):
        InstructionAdherenceEvaluator(ScriptedJudge(response)).evaluate(CASE)


@pytest.mark.parametrize(
    "response",
    [None, {}, {"answers": "not a list"}, {"answers": ["bad"]}],
)
def test_malformed_classification_fails_clearly(response):
    judge = ScriptedJudge(_extraction("Return JSON."), response)
    with pytest.raises(ValueError):
        InstructionAdherenceEvaluator(judge).evaluate(CASE)


@pytest.mark.parametrize("failure_stage", ["extract", "classify"])
def test_judge_failure_propagates(failure_stage):
    error = RuntimeError("judge unavailable")
    judge = (
        ScriptedJudge(error)
        if failure_stage == "extract"
        else ScriptedJudge(_extraction("Return JSON."), error)
    )
    with pytest.raises(RuntimeError, match="judge unavailable"):
        InstructionAdherenceEvaluator(judge).evaluate(CASE)


def test_extraction_prompt_contains_only_instruction_case_data():
    judge = _judge(["Return JSON."], ["followed"])
    InstructionAdherenceEvaluator(judge).evaluate(CASE)
    user = judge.calls[0]["prompt"][1]["content"]
    assert CASE.instructions in user
    assert CASE.input not in user
    assert CASE.context not in user
    assert CASE.output not in user


def test_classification_prompt_contains_only_fixed_instructions_and_output():
    judge = _judge(["Return JSON."], ["followed"])
    InstructionAdherenceEvaluator(judge).evaluate(CASE)
    user = judge.calls[1]["prompt"][1]["content"]
    assert '"instruction": "Return JSON."' in user
    assert CASE.output in user
    assert CASE.input not in user
    assert CASE.context not in user


def test_prompt_schemas_are_minimal_and_binary():
    extract_item = INSTRUCTION_ADHERENCE_EXTRACT_SCHEMA["properties"][
        "instructions"
    ]["items"]
    classify_item = INSTRUCTION_ADHERENCE_CLASSIFY_SCHEMA["properties"]["answers"][
        "items"
    ]
    assert extract_item["required"] == ["instruction"]
    assert classify_item["required"] == ["id", "status", "reason"]
    assert classify_item["properties"]["status"]["enum"] == [
        "followed",
        "violated",
    ]


def test_extraction_rubric_documents_split_qualifiers_and_generic_scope():
    system = INSTRUCTION_ADHERENCE_EXTRACT_PROMPT[0]["content"]
    lowered = system.lower()
    normalized = " ".join(lowered.split())
    assert "independently be followed or violated" in lowered
    assert "exact counts" in lowered
    assert "do not invent unstated instructions" in normalized
    assert all(word not in lowered for word in ("jira", "theme", "epic"))


def test_prompt_renderers_do_not_mutate_global_templates():
    extract_before = copy.deepcopy(INSTRUCTION_ADHERENCE_EXTRACT_PROMPT)
    classify_before = copy.deepcopy(INSTRUCTION_ADHERENCE_CLASSIFY_PROMPT)
    render_instruction_adherence_extract_prompt("Return JSON.")
    render_instruction_adherence_classify_prompt("[]", "output")
    assert INSTRUCTION_ADHERENCE_EXTRACT_PROMPT == extract_before
    assert INSTRUCTION_ADHERENCE_CLASSIFY_PROMPT == classify_before
    assert "{instructions}" in INSTRUCTION_ADHERENCE_EXTRACT_PROMPT[1]["content"]


def test_framework_usage_is_unchanged():
    framework = EvaluationFramework(
        evaluators=[
            InstructionAdherenceEvaluator(
                _judge(["Return JSON."], ["followed"])
            )
        ]
    )
    results = framework.evaluate(CASE, metrics=["instruction_adherence"])
    assert set(results) == {"instruction_adherence"}
    assert results["instruction_adherence"].score == 1.0
