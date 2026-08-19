"""Contract tests for the final one-call coverage prompt and schemas."""

import copy

from idp_eval.prompts.coverage import (
    COVERAGE_PROMPT_COMPACT_V1,
    COVERAGE_SCHEMA_COMPACT,
    COVERAGE_SCHEMA_VERBOSE,
    render_coverage_prompt,
)


def _system(verbose: bool = False) -> str:
    return " ".join(
        render_coverage_prompt("source", "output", verbose=verbose)[0][
            "content"
        ].split()
    )


def test_prompt_is_one_call_context_plus_output():
    messages = render_coverage_prompt("SOURCE", "GENERATED")
    assert [message["role"] for message in messages] == ["system", "user"]
    user = messages[1]["content"]
    assert "[CONTEXT]\nSOURCE" in user
    assert "[OUTPUT]\nGENERATED" in user
    assert "[INPUT]" not in user


def test_prompt_requires_all_materially_distinct_items_without_count_target():
    system = _system()
    assert "all materially distinct source items" in system
    assert "semantic consolidation" in system
    assert "no fixed or approximate item-count target" in system
    assert "Do not omit or merge distinct information" in system
    assert "approximately 10" not in system
    assert "target 10" not in system


def test_prompt_preserves_qualifiers_and_excludes_structural_meta_text():
    system = _system()
    assert "Preserve material qualifiers" in system
    assert "headings, section labels, introductory phrases" in system
    assert "structural instructions, meta-statements" in system
    assert "The solution must satisfy the following requirements" in system


def test_prompt_handles_distinct_objectives_without_redundant_umbrellas():
    system = _system()
    assert "objective or outcome only when it adds materially distinct meaning" in system
    assert "redundant umbrella item and child items" in system


def test_prompt_rejects_generic_overlap_for_partial_credit():
    system = _system()
    assert "Generic topical overlap alone is not meaningful presence" in system
    assert "concrete semantic component" in system
    assert "Vague language in the same topic area is insufficient" in system


def test_prompt_forbids_llm_numeric_scoring():
    system = _system()
    assert "Do not return an aggregate score" in system
    assert "Python derives all statuses and numeric scores" in system


def test_compact_schema_contains_only_required_judgment_fields():
    assert COVERAGE_SCHEMA_COMPACT["additionalProperties"] is False
    item = COVERAGE_SCHEMA_COMPACT["properties"]["items"]["items"]
    assert set(item["properties"]) == {
        "source_item",
        "meaningfully_present",
        "fully_present",
    }
    assert item["required"] == [
        "source_item",
        "meaningfully_present",
        "fully_present",
    ]
    assert item["additionalProperties"] is False


def test_verbose_schema_requires_reason_for_openai_strict_output():
    item = COVERAGE_SCHEMA_VERBOSE["properties"]["items"]["items"]
    assert set(item["properties"]) == {
        "source_item",
        "meaningfully_present",
        "fully_present",
        "reason",
    }
    assert item["required"] == [
        "source_item",
        "meaningfully_present",
        "fully_present",
        "reason",
    ]
    assert item["additionalProperties"] is False
    assert "empty string" in _system(verbose=True)
    assert "concise non-empty explanation" in _system(verbose=True)


def test_render_does_not_mutate_prompt_template():
    before = copy.deepcopy(COVERAGE_PROMPT_COMPACT_V1)
    render_coverage_prompt("a", "b")
    assert COVERAGE_PROMPT_COMPACT_V1 == before
