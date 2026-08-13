"""Tests for the coverage prompt structure and rendering (no LLM needed)."""

import copy

from idp_eval.prompts.coverage import (
    COVERAGE_PROMPT,
    COVERAGE_PROMPT_V1,
    COVERAGE_SCHEMA,
    render_coverage_prompt,
)


def test_prompt_is_message_list_with_system_and_user():
    assert COVERAGE_PROMPT is COVERAGE_PROMPT_V1
    roles = [message["role"] for message in COVERAGE_PROMPT]
    assert roles == ["system", "user"]


def test_system_message_holds_rubric_not_data():
    system = COVERAGE_PROMPT[0]["content"]
    assert "Coverage measures how much of the material information" in system
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
    # Data lands in the correct labelled sections.
    assert "[INPUT]\nTASK_TEXT" in user
    assert "[CONTEXT]\nCONTEXT_TEXT" in user
    assert "[OUTPUT]\nOUTPUT_TEXT" in user
    assert "[BEGIN DATA]" in user and "[END DATA]" in user


def test_render_does_not_mutate_global_template():
    before = copy.deepcopy(COVERAGE_PROMPT)
    render_coverage_prompt(input_text="a", context="b", output="c")
    # The module-level template is untouched: still has placeholders.
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
    assert "Do NOT return any numeric score" in system


def test_schema_allows_optional_reason():
    item_props = COVERAGE_SCHEMA["properties"]["items"]["items"]["properties"]
    assert set(item_props) == {"source_item", "status", "reason"}
    required = COVERAGE_SCHEMA["properties"]["items"]["items"]["required"]
    assert "reason" not in required
