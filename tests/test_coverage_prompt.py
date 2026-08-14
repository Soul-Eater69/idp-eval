"""Tests for the coverage prompt structure and rendering (no LLM needed)."""

import copy

from idp_eval.prompts.coverage import (
    COVERAGE_PROMPT,
    COVERAGE_PROMPT_V2,
    COVERAGE_SCHEMA,
    render_coverage_prompt,
)


def test_prompt_is_message_list_with_system_and_user():
    assert COVERAGE_PROMPT is COVERAGE_PROMPT_V2
    roles = [message["role"] for message in COVERAGE_PROMPT]
    assert roles == ["system", "user"]


def test_system_message_holds_rubric_not_data():
    system = COVERAGE_PROMPT[0]["content"]
    assert "How much of the task-relevant information" in system
    # Recall-style decomposition + classification vocabulary is present.
    assert "atomic requirements" in system
    assert "covered" in system and "partial" in system and "missing" in system
    # Rubric explicitly separates coverage from faithfulness.
    assert "faithfulness" in system.lower()
    # The system message must not carry the data placeholders.
    assert "{input}" not in system
    assert "{context}" not in system
    assert "{output}" not in system


def test_render_fills_user_sections():
    messages = render_coverage_prompt(
        input_text="TASK_TEXT",
        context="CONTEXT_TEXT",
        output="OUTPUT_TEXT",
    )
    user = messages[1]["content"]
    assert "[INPUT]\nTASK_TEXT" in user
    assert "[CONTEXT]\nCONTEXT_TEXT" in user
    assert "[OUTPUT]\nOUTPUT_TEXT" in user
    assert "[BEGIN DATA]" in user and "[END DATA]" in user


def test_render_does_not_mutate_global_template():
    before = copy.deepcopy(COVERAGE_PROMPT)
    render_coverage_prompt(input_text="a", context="b", output="c")
    assert COVERAGE_PROMPT == before
    assert "{input}" in COVERAGE_PROMPT[1]["content"]


def test_render_returns_fresh_object_each_call():
    first = render_coverage_prompt(input_text="a", context="b", output="c")
    second = render_coverage_prompt(input_text="a", context="b", output="c")
    assert first == second
    assert first is not second
    assert first[0] is not COVERAGE_PROMPT[0]


def test_system_rubric_forbids_numeric_scores():
    system = COVERAGE_PROMPT[0]["content"]
    assert "Do NOT return any aggregate score" in system


def test_schema_uses_requirements_with_required_reason():
    req_items = COVERAGE_SCHEMA["properties"]["requirements"]["items"]
    assert set(req_items["properties"]) == {"requirement", "status", "reason"}
    assert req_items["required"] == ["requirement", "status", "reason"]
    assert req_items["properties"]["status"]["enum"] == [
        "covered",
        "partial",
        "missing",
    ]
