"""Offline tests for one-call holistic instruction adherence."""

import asyncio
import copy
import threading
import time

import pytest

from idp_eval import (
    EvaluationCase,
    EvaluationFramework,
    InstructionAdherenceEvaluator,
)
from idp_eval.models import EvaluationResult
from idp_eval.prompts.instruction_adherence import (
    INSTRUCTION_ADHERENCE_PROMPT_COMPACT_V1,
    INSTRUCTION_ADHERENCE_PROMPT_VERBOSE_V1,
    INSTRUCTION_ADHERENCE_SCHEMA_COMPACT,
    INSTRUCTION_ADHERENCE_SCHEMA_VERBOSE,
    render_instruction_adherence_prompt,
)


class ScriptedJudge:
    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def generate_object(self, prompt, schema: dict) -> dict:
        self.calls.append({"prompt": prompt, "schema": schema})
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class NativeAsyncJudge:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def generate_object(self, prompt, schema):
        raise AssertionError("sync path must not be called")

    async def async_generate_object(self, prompt, schema):
        self.calls += 1
        await asyncio.sleep(0)
        return self.response


class ConcurrencyJudge:
    def __init__(self):
        self.calls = 0
        self.current = 0
        self.max_concurrent = 0
        self._lock = threading.Lock()

    def generate_object(self, prompt, schema):
        with self._lock:
            self.calls += 1
            self.current += 1
            self.max_concurrent = max(self.max_concurrent, self.current)
        try:
            time.sleep(0.03)
            return _response(("Use concise language.", "followed"))
        finally:
            with self._lock:
                self.current -= 1


def _response(*items, verbose=False):
    rows = []
    for item in items:
        instruction, status, *reason = item
        row = {"instruction": instruction, "status": status}
        if verbose:
            row["reason"] = reason[0] if reason else ""
        rows.append(row)
    return {"instructions": rows}


CASE = EvaluationCase(
    input="UNIQUE INPUT THAT MUST NOT REACH THIS METRIC",
    instructions={
        "count": "Return exactly 3 recommendations",
        "constraints": ["Use JSON", "Do not mention pricing"],
    },
    context="UNIQUE CONTEXT THAT MUST NOT REACH THIS METRIC",
    output={"recommendations": ["a", "b", "c"]},
    metadata={"secret_marker": "UNIQUE METADATA THAT MUST NOT REACH THIS METRIC"},
)


def test_compact_one_call_scores_and_returns_compact_details():
    judge = ScriptedJudge(
        _response(
            ("Return exactly 3 recommendations.", "followed"),
            ("Use JSON.", "followed"),
            ("Do not mention pricing.", "followed"),
        )
    )
    result = InstructionAdherenceEvaluator(judge).evaluate(CASE)

    assert isinstance(result, EvaluationResult)
    assert result.metric == "instruction_adherence"
    assert result.score == 1.0
    assert result.label == "fully_followed"
    assert result.details["instruction_count"] == 3
    assert result.details["followed_count"] == 3
    assert result.details["violated_count"] == 0
    assert result.details["judge_call_count"] == 1
    assert result.details["verbose"] is False
    assert result.details["total_ms"] >= 0
    assert "instructions" not in result.details
    assert len(judge.calls) == 1


def test_verbose_mode_returns_stable_ids_scores_and_reasons():
    judge = ScriptedJudge(
        _response(
            ("Return JSON.", "followed", ""),
            ("Do not mention pricing.", "violated", "Pricing is present."),
            verbose=True,
        )
    )
    result = InstructionAdherenceEvaluator(judge, verbose=True).evaluate(CASE)
    assert result.score == 0.5
    assert result.label == "violations_present"
    assert result.details["instructions"] == [
        {
            "id": "I1",
            "instruction": "Return JSON.",
            "status": "followed",
            "score": 1.0,
            "reason": "",
        },
        {
            "id": "I2",
            "instruction": "Do not mention pricing.",
            "status": "violated",
            "score": 0.0,
            "reason": "Pricing is present.",
        },
    ]


@pytest.mark.parametrize(
    ("count", "status", "expected"),
    [(3, "followed", 1.0), (2, "violated", 0.0)],
)
def test_exactly_three_structured_epics_are_judged_in_one_call(
    count, status, expected
):
    case = EvaluationCase(
        instructions="Generate exactly 3 epics.",
        output=[{"title": str(index)} for index in range(count)],
    )
    judge = ScriptedJudge(_response(("Generate exactly 3 epics.", status)))
    result = InstructionAdherenceEvaluator(judge).evaluate(case)
    assert result.score == expected
    assert result.details["judge_call_count"] == 1
    assert len(judge.calls) == 1


def test_compound_instruction_decomposes_to_four_binary_items():
    response = _response(
        ("Generate exactly 3 epics.", "followed"),
        ("Every epic must include a title.", "followed"),
        ("Every epic must include a description.", "followed"),
        ("Every epic must include success criteria.", "violated"),
    )
    case = EvaluationCase(
        instructions=(
            "Generate exactly 3 epics with a title, description, and success "
            "criteria."
        ),
        output=[{"title": "A", "description": "B"}],
    )
    result = InstructionAdherenceEvaluator(ScriptedJudge(response)).evaluate(case)
    assert result.score == 0.75
    assert result.details["instruction_count"] == 4


def test_universal_constraint_violation_has_audit_reason():
    response = _response(
        (
            "Every recommendation must contain rationale.",
            "violated",
            "Recommendation 2 does not contain rationale.",
        ),
        verbose=True,
    )
    case = EvaluationCase(
        instructions="Every recommendation must contain rationale.",
        output=[{"text": "A", "rationale": "why"}, {"text": "B"}],
    )
    result = InstructionAdherenceEvaluator(
        ScriptedJudge(response), verbose=True
    ).evaluate(case)
    assert result.score == 0.0
    assert result.details["instructions"][0]["reason"] == (
        "Recommendation 2 does not contain rationale."
    )


@pytest.mark.parametrize(
    ("instruction", "output", "status"),
    [
        ("Generate at least 3 options.", [1, 2, 3], "followed"),
        ("Return no more than 5 records.", list(range(6)), "violated"),
        ("Each record must include an ID.", [{"id": 1}], "followed"),
        ("Do not expose internal IDs.", "Internal ID: 7", "violated"),
        ("Keep each summary under 100 words.", "Short summary.", "followed"),
        ("Sort by descending revenue.", [10, 20], "violated"),
        ("Return JSON.", {"ok": True}, "followed"),
        ("Use a professional tone.", "Thank you for your consideration.", "followed"),
    ],
)
def test_generic_semantic_constraints_use_the_same_one_call_path(
    instruction, output, status
):
    judge = ScriptedJudge(_response((instruction, status)))
    result = InstructionAdherenceEvaluator(judge).evaluate(
        EvaluationCase(instructions=instruction, output=output)
    )
    assert result.score == (1.0 if status == "followed" else 0.0)
    assert len(judge.calls) == 1


def test_normalized_exact_duplicates_keep_first_and_receive_one_id():
    response = _response(
        ("Return JSON.", "followed", ""),
        ("  return   json.  ", "followed", ""),
        ("RETURN JSON.", "followed", ""),
        verbose=True,
    )
    result = InstructionAdherenceEvaluator(
        ScriptedJudge(response), verbose=True
    ).evaluate(CASE)
    assert result.details["instruction_count"] == 1
    assert result.details["instructions"][0]["id"] == "I1"
    assert result.details["instructions"][0]["instruction"] == "Return JSON."


@pytest.mark.parametrize("verbose", [False, True])
def test_empty_judge_instruction_list_is_not_applicable_after_one_call(verbose):
    judge = ScriptedJudge({"instructions": []})
    result = InstructionAdherenceEvaluator(judge, verbose=verbose).evaluate(CASE)
    assert result.score is None
    assert result.label == "not_applicable"
    assert result.details["instruction_count"] == 0
    assert result.details["judge_call_count"] == 1
    assert result.details["verbose"] is verbose
    assert (result.details.get("instructions") == []) is verbose
    assert len(judge.calls) == 1


@pytest.mark.parametrize("instructions", [None, "", "   \n\t", [], {}])
def test_missing_instructions_fails_before_judge_call(instructions):
    judge = ScriptedJudge(AssertionError("must not call"))
    case = EvaluationCase(instructions=instructions, output="output")
    with pytest.raises(ValueError, match="requires non-empty `instructions`"):
        InstructionAdherenceEvaluator(judge).evaluate(case)
    assert judge.calls == []


def test_missing_output_fails_before_judge_call():
    judge = ScriptedJudge(AssertionError("must not call"))
    with pytest.raises(ValueError, match="requires non-empty `output`"):
        InstructionAdherenceEvaluator(judge).evaluate(
            EvaluationCase(instructions="Return JSON.", output="")
        )
    assert judge.calls == []


@pytest.mark.parametrize(
    "response",
    [
        None,
        {},
        {"instructions": [], "extra": True},
        {"instructions": "bad"},
        {"instructions": ["bad"]},
        _response(("", "followed")),
        _response(("Return JSON.", "partial")),
        {"instructions": [{"instruction": "Return JSON.", "status": True}]},
        {
            "instructions": [
                {"instruction": "Return JSON.", "status": "followed", "extra": 1}
            ]
        },
    ],
)
def test_malformed_compact_response_fails_clearly(response):
    with pytest.raises(ValueError):
        InstructionAdherenceEvaluator(ScriptedJudge(response)).evaluate(CASE)


@pytest.mark.parametrize(
    "response",
    [
        _response(("Return JSON.", "followed", "unneeded"), verbose=True),
        _response(("Do not mention pricing.", "violated", ""), verbose=True),
        {
            "instructions": [
                {
                    "instruction": "Return JSON.",
                    "status": "followed",
                    "reason": None,
                }
            ]
        },
        {"instructions": [{"instruction": "Return JSON.", "status": "followed"}]},
    ],
)
def test_verbose_reason_contract_is_strict(response):
    with pytest.raises(ValueError):
        InstructionAdherenceEvaluator(
            ScriptedJudge(response), verbose=True
        ).evaluate(CASE)


def test_prompt_contains_only_rendered_instructions_and_complete_output():
    judge = ScriptedJudge(_response(("Use JSON.", "followed")))
    InstructionAdherenceEvaluator(judge).evaluate(CASE)
    user = judge.calls[0]["prompt"][1]["content"]
    assert "[BEGIN DATA]" in user and "[END DATA]" in user
    assert "[INSTRUCTIONS]" in user and "[OUTPUT]" in user
    assert "Return exactly 3 recommendations" in user
    assert "Recommendations:" in user
    assert CASE.input not in user
    assert CASE.context not in user
    assert CASE.metadata["secret_marker"] not in user


def test_structured_instructions_and_list_of_dicts_remain_one_case_and_call():
    case = EvaluationCase(
        instructions={
            "count": "Generate exactly 2 records",
            "fields": ["Each record has an ID", "Each record has a summary"],
        },
        output=[{"id": 1, "summary": "a"}, {"id": 2, "summary": "b"}],
    )
    judge = ScriptedJudge(_response(("Generate exactly 2 records.", "followed")))
    InstructionAdherenceEvaluator(judge).evaluate(case)
    user = judge.calls[0]["prompt"][1]["content"]
    assert "Count: Generate exactly 2 records" in user
    assert user.count("- Id:") == 2
    assert len(judge.calls) == 1


def test_schemas_are_strict_binary_and_reason_is_verbose_only():
    compact_item = INSTRUCTION_ADHERENCE_SCHEMA_COMPACT["properties"][
        "instructions"
    ]["items"]
    verbose_item = INSTRUCTION_ADHERENCE_SCHEMA_VERBOSE["properties"][
        "instructions"
    ]["items"]
    assert compact_item["required"] == ["instruction", "status"]
    assert verbose_item["required"] == ["instruction", "status", "reason"]
    assert "reason" not in compact_item["properties"]
    assert verbose_item["properties"]["reason"] == {"type": "string"}
    assert compact_item["properties"]["status"]["enum"] == [
        "followed",
        "violated",
    ]
    for schema in (
        INSTRUCTION_ADHERENCE_SCHEMA_COMPACT,
        INSTRUCTION_ADHERENCE_SCHEMA_VERBOSE,
    ):
        assert schema["additionalProperties"] is False
        assert schema["properties"]["instructions"]["items"][
            "additionalProperties"
        ] is False


def test_prompt_contract_is_generic_strict_and_not_domain_specific():
    system = INSTRUCTION_ADHERENCE_PROMPT_COMPACT_V1[0]["content"].lower()
    normalized = " ".join(system.split())
    assert "independently be followed or violated" in normalized
    assert '"each", "every", and "all"' in normalized
    assert "exact/at-least/maximum counts" in normalized
    assert "prohibition" in normalized
    assert all(word not in normalized for word in ("jira", "theme", "epic"))


def test_prompt_renderer_does_not_mutate_global_templates():
    compact = copy.deepcopy(INSTRUCTION_ADHERENCE_PROMPT_COMPACT_V1)
    verbose = copy.deepcopy(INSTRUCTION_ADHERENCE_PROMPT_VERBOSE_V1)
    render_instruction_adherence_prompt("Return JSON.", "{}")
    render_instruction_adherence_prompt("Return JSON.", "{}", verbose=True)
    assert INSTRUCTION_ADHERENCE_PROMPT_COMPACT_V1 == compact
    assert INSTRUCTION_ADHERENCE_PROMPT_VERBOSE_V1 == verbose
    assert "{instructions}" in INSTRUCTION_ADHERENCE_PROMPT_COMPACT_V1[1]["content"]


def test_sync_judge_failure_propagates_without_retry():
    judge = ScriptedJudge(RuntimeError("judge unavailable"))
    with pytest.raises(RuntimeError, match="judge unavailable"):
        InstructionAdherenceEvaluator(judge).evaluate(CASE)
    assert len(judge.calls) == 1


def test_native_async_path_calls_async_judge_once():
    judge = NativeAsyncJudge(_response(("Use concise language.", "followed")))
    result = asyncio.run(
        InstructionAdherenceEvaluator(judge).a_evaluate(
            CASE, judge_limiter=asyncio.Semaphore(2)
        )
    )
    assert result.score == 1.0
    assert judge.calls == 1


def test_async_fallback_calls_sync_judge_once():
    judge = ScriptedJudge(_response(("Use concise language.", "followed")))
    result = asyncio.run(
        InstructionAdherenceEvaluator(judge).a_evaluate(
            CASE, judge_limiter=asyncio.Semaphore(1)
        )
    )
    assert result.score == 1.0
    assert len(judge.calls) == 1


def test_async_framework_semaphore_bounds_instruction_judge_calls():
    judge = ConcurrencyJudge()
    framework = EvaluationFramework(
        evaluators=[InstructionAdherenceEvaluator(judge)]
    )
    cases = [
        EvaluationCase(
            instructions="Use concise language.", output=f"output {index}"
        )
        for index in range(5)
    ]
    results = asyncio.run(framework.a_evaluate_many(cases, max_concurrency=2))
    assert [row["instruction_adherence"].score for row in results] == [1.0] * 5
    assert judge.calls == 5
    assert judge.max_concurrent == 2


def test_framework_usage_is_unchanged():
    framework = EvaluationFramework(
        evaluators=[
            InstructionAdherenceEvaluator(
                ScriptedJudge(_response(("Use JSON.", "followed")))
            )
        ]
    )
    result = framework.evaluate(CASE, metrics=["instruction_adherence"])
    assert result["instruction_adherence"].score == 1.0
