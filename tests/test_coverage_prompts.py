"""Tests for the two-stage coverage prompts (no LLM needed)."""

import copy

from idp_eval.prompts.coverage_classify import (
    COVERAGE_CLASSIFY_PROMPT,
    COVERAGE_CLASSIFY_PROMPT_V1,
    COVERAGE_CLASSIFY_SCHEMA,
    render_coverage_classify_prompt,
)
from idp_eval.prompts.coverage_extract import (
    COVERAGE_EXTRACT_PROMPT,
    COVERAGE_EXTRACT_PROMPT_V1,
    COVERAGE_EXTRACT_SCHEMA,
    render_coverage_extract_prompt,
)


# --- Stage 1: extraction ----------------------------------------------------


def test_extract_prompt_is_message_list():
    assert COVERAGE_EXTRACT_PROMPT is COVERAGE_EXTRACT_PROMPT_V1
    assert [m["role"] for m in COVERAGE_EXTRACT_PROMPT] == ["system", "user"]


def test_extract_system_forbids_grading_and_output():
    # Normalize whitespace so line wrapping doesn't affect substring checks.
    system = " ".join(COVERAGE_EXTRACT_PROMPT[0]["content"].split())
    assert "NOT given any generated output" in system
    assert "must NOT grade" in system
    assert "atomic" in system
    # Extraction must not ask for a numeric verdict.
    assert "score" in system  # only as part of "Do NOT return any status, score..."
    assert "Do NOT return any status, score, percentage" in system


def test_extract_render_has_input_context_not_output():
    messages = render_coverage_extract_prompt(
        input_text="TASK_TEXT", context="CONTEXT_TEXT"
    )
    user = messages[1]["content"]
    assert "[INPUT]\nTASK_TEXT" in user
    assert "[CONTEXT]\nCONTEXT_TEXT" in user
    assert "[OUTPUT]" not in user


def test_extract_render_does_not_mutate_template():
    before = copy.deepcopy(COVERAGE_EXTRACT_PROMPT)
    render_coverage_extract_prompt(input_text="a", context="b")
    assert COVERAGE_EXTRACT_PROMPT == before
    assert "{input}" in COVERAGE_EXTRACT_PROMPT[1]["content"]


def test_extract_schema_shape():
    item = COVERAGE_EXTRACT_SCHEMA["properties"]["requirements"]["items"]
    assert set(item["properties"]) == {"requirement"}
    assert item["required"] == ["requirement"]


# --- Stage 2: classification ------------------------------------------------


def test_classify_prompt_is_message_list():
    assert COVERAGE_CLASSIFY_PROMPT is COVERAGE_CLASSIFY_PROMPT_V1
    assert [m["role"] for m in COVERAGE_CLASSIFY_PROMPT] == ["system", "user"]


def test_classify_system_uses_binary_and_forbids_edits():
    system = COVERAGE_CLASSIFY_PROMPT[0]["content"]
    assert "meaningfully_present" in system and "fully_present" in system
    assert "DO NOT add, remove, merge, split, or rewrite" in system
    # Consistency rule stated.
    assert "fully_present must also be false" in system


def test_classify_render_has_requirements_output_not_context():
    messages = render_coverage_classify_prompt(
        input_text="TASK",
        requirements_json='[{"id": "r1", "requirement": "REQ_TEXT"}]',
        output="OUTPUT_TEXT",
    )
    user = messages[1]["content"]
    assert "REQ_TEXT" in user
    assert "[OUTPUT]\nOUTPUT_TEXT" in user
    assert "[CONTEXT]" not in user


def test_classify_render_does_not_mutate_template():
    before = copy.deepcopy(COVERAGE_CLASSIFY_PROMPT)
    render_coverage_classify_prompt(
        input_text="a", requirements_json="[]", output="c"
    )
    assert COVERAGE_CLASSIFY_PROMPT == before


def test_classify_schema_requires_booleans_and_id():
    item = COVERAGE_CLASSIFY_SCHEMA["properties"]["requirements"]["items"]
    assert set(item["properties"]) == {
        "id",
        "meaningfully_present",
        "fully_present",
        "reason",
    }
    assert item["required"] == [
        "id",
        "meaningfully_present",
        "fully_present",
        "reason",
    ]
    assert item["properties"]["meaningfully_present"]["type"] == "boolean"
    assert item["properties"]["fully_present"]["type"] == "boolean"
