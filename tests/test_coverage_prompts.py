"""Contract tests for the final one-call coverage prompt and schemas."""

import copy

from idp_eval.prompts.coverage import (
    COVERAGE_PROMPT_COMPACT_V1,
    COVERAGE_SCHEMA_NONE,
    COVERAGE_SCHEMA_OVERALL,
    COVERAGE_SCHEMA_PER_ITEM,
    render_coverage_prompt,
)


def _system(reason_mode: str = "overall") -> str:
    return " ".join(
        render_coverage_prompt("source", "output", reason_mode=reason_mode)[0][
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


def test_prompt_requires_all_materially_distinct_items_when_unlimited():
    system = _system()
    assert "all materially distinct, reasonably atomic" in system
    assert "independently assessable" in system
    assert "approximately 10" not in system
    assert "target 10" not in system


def test_prompt_expresses_optional_limit_as_at_most_without_padding():
    system = " ".join(
        render_coverage_prompt("source", "output", max_items=5)[0][
            "content"
        ].split()
    )
    assert "select at most 5 of the most material and representative" in system
    assert "If fewer than 5 meaningful items exist" in system
    assert "Examine the complete CONTEXT before selecting" in system
    assert "Do not stop after finding the first 5 candidates" in system
    assert "independently of whether the OUTPUT covers them" in system
    assert "Only after selection, classify" in system
    assert "Do not invent, duplicate, or artificially split items" in system
    assert "Never merge multiple independent facts or requirements" in system
    assert "item limit controls how many units are selected" in system


def test_prompt_preserves_qualifiers_and_excludes_structural_meta_text():
    system = _system()
    assert "Preserve material qualifiers" in system
    assert "headings, section labels, introductory phrases" in system.lower()
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
    assert "Python derives all statuses, labels, and numeric scores" in system


def test_compact_schema_contains_only_required_judgment_fields():
    assert COVERAGE_SCHEMA_NONE["additionalProperties"] is False
    item = COVERAGE_SCHEMA_NONE["properties"]["items"]["items"]
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


def test_overall_and_per_item_schemas_are_strict_and_include_overall_reason():
    for schema in (COVERAGE_SCHEMA_OVERALL, COVERAGE_SCHEMA_PER_ITEM):
        assert schema["required"] == ["items", "overall_reason"]
        assert schema["properties"]["overall_reason"] == {"type": "string"}
        assert "maxItems" not in schema["properties"]["items"]
    item = COVERAGE_SCHEMA_OVERALL["properties"]["items"]["items"]
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
    assert "empty reason string" in _system("overall")
    assert "concise, non-empty diagnostic reason" in _system("overall")


def test_reason_mode_prompts_define_semantic_explanation_contract():
    overall = _system("overall")
    per_item = _system("per_item")
    none = _system("none")
    assert "at least one and at most three representative" in overall
    assert "Do not include a metric score, percentage, item counts" in overall
    assert (
        "Start directly with the substantive supported area or failure"
        in overall
    )
    assert "Do not begin with generic aggregate commentary" in overall
    assert "Most requirements are covered" in overall
    assert "items = []" in overall
    assert "Do not invent source items merely to avoid an empty array" in overall
    assert "The context contains no materially evaluable source items" in overall
    assert "non-empty reason for every source item" in per_item
    assert "at least one and at most three representative" in per_item
    assert "Start directly with the substantive supported area" in per_item
    assert "items is empty" in per_item
    assert "Do not return per-item reasons, overall_reason" in none


def test_schemas_never_ask_llm_for_score_label_or_max_items():
    for schema in (
        COVERAGE_SCHEMA_OVERALL,
        COVERAGE_SCHEMA_PER_ITEM,
        COVERAGE_SCHEMA_NONE,
    ):
        serialized = repr(schema)
        assert "score" not in serialized
        assert "label" not in serialized
        assert "maxItems" not in serialized


def test_render_does_not_mutate_prompt_template():
    before = copy.deepcopy(COVERAGE_PROMPT_COMPACT_V1)
    render_coverage_prompt("a", "b")
    assert COVERAGE_PROMPT_COMPACT_V1 == before
