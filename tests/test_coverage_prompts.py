"""Tests for task and source coverage prompts (no LLM needed)."""

import copy

from idp_eval.prompts.coverage_classify import (
    COVERAGE_CLASSIFY_PROMPT,
    COVERAGE_CLASSIFY_PROMPT_V1,
    COVERAGE_CLASSIFY_SCHEMA,
    COVERAGE_CLASSIFY_SCHEMA_COMPACT,
    COVERAGE_CLASSIFY_SCHEMA_VERBOSE,
    render_coverage_classify_prompt,
)
from idp_eval.prompts.coverage_extract import (
    COVERAGE_EXTRACT_PROMPT,
    COVERAGE_EXTRACT_PROMPT_V1,
    COVERAGE_EXTRACT_SCHEMA,
    render_coverage_extract_prompt,
)
from idp_eval.prompts.coverage import (
    COVERAGE_EXTRACT_SCHEMA as COVERAGE_DAG_EXTRACT_SCHEMA,
    COVERAGE_GEVAL_SCHEMA_COMPACT,
    COVERAGE_GEVAL_SCHEMA_VERBOSE,
    render_coverage_extract_prompt as render_coverage_dag_extract_prompt,
    render_coverage_geval_prompt,
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
    # Extraction must not ask for a numeric verdict.
    assert "score" in system  # only as part of "Do NOT return any status, score..."
    assert "Do NOT return any status, score, percentage" in system


def test_extract_system_instructs_semantic_consolidation_without_hard_limit():
    system = " ".join(COVERAGE_EXTRACT_PROMPT[0]["content"].split())
    # Consolidation guidance present.
    assert "Consolidate" in system
    assert "independently" in system  # independently-satisfiable stay separate
    # No hard maximum on the denominator.
    assert "no target or maximum number" in system
    # Extraction stays diagnostic-free (no rationale/evidence requested).
    assert "rationale" in system and "evidence" in system


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


def test_classify_default_schema_is_compact_no_reason():
    # Default (compact) schema: only id + the two booleans, no per-item reason.
    assert COVERAGE_CLASSIFY_SCHEMA is COVERAGE_CLASSIFY_SCHEMA_COMPACT
    item = COVERAGE_CLASSIFY_SCHEMA["properties"]["requirements"]["items"]
    assert set(item["properties"]) == {
        "id",
        "meaningfully_present",
        "fully_present",
    }
    assert item["required"] == ["id", "meaningfully_present", "fully_present"]
    assert item["properties"]["meaningfully_present"]["type"] == "boolean"
    assert item["properties"]["fully_present"]["type"] == "boolean"


def test_classify_verbose_schema_adds_optional_reason():
    item = COVERAGE_CLASSIFY_SCHEMA_VERBOSE["properties"]["requirements"]["items"]
    assert set(item["properties"]) == {
        "id",
        "meaningfully_present",
        "fully_present",
        "reason",
    }
    # reason is optional (only partial/missing items carry it).
    assert item["required"] == ["id", "meaningfully_present", "fully_present"]


def test_classify_compact_prompt_forbids_reason_verbose_requests_it():
    compact = render_coverage_classify_prompt(
        input_text="t", requirements_json="[]", output="o"
    )[0]["content"]
    verbose = render_coverage_classify_prompt(
        input_text="t", requirements_json="[]", output="o", verbose=True
    )[0]["content"]
    assert "Do NOT return any\nreason" in compact or "Do NOT return any reason" in (
        " ".join(compact.split())
    )
    assert "reason" in verbose and "one-sentence" in verbose


# --- whole-source coverage: DAG Stage-1 (~10 target) ------------------------


def test_dag_extract_is_context_only_no_output():
    messages = render_coverage_dag_extract_prompt(context="SOURCE_TEXT")
    assert [m["role"] for m in messages] == ["system", "user"]
    user = messages[1]["content"]
    assert "[CONTEXT]\nSOURCE_TEXT" in user
    assert "[OUTPUT]" not in user and "[INPUT]" not in user


def test_dag_extract_targets_about_ten_without_hard_cap():
    system = " ".join(render_coverage_dag_extract_prompt(context="x")[0]["content"].split())
    assert "approximately 10" in system
    assert "NOT a hard maximum" in system
    assert 'NOT "pick the top 10"' in system
    # semantic consolidation, independence, and no dropping of distinct items
    assert "Consolidate" in system
    assert "independently satisfiable" in system
    assert "Never drop a materially distinct" in system
    assert "return more" in system  # may exceed 10 when necessary
    # extraction is output-isolated and diagnostic-free
    assert "NOT given the generated answer" in system
    assert "must not grade" in system


def test_dag_extract_schema_is_source_item_strings_only():
    item = COVERAGE_DAG_EXTRACT_SCHEMA["properties"]["items"]["items"]
    assert set(item["properties"]) == {"source_item"}
    assert item["required"] == ["source_item"]


def test_dag_extract_render_does_not_mutate_template():
    from idp_eval.prompts.coverage import COVERAGE_EXTRACT_PROMPT

    before = copy.deepcopy(COVERAGE_EXTRACT_PROMPT)
    render_coverage_dag_extract_prompt(context="a")
    assert COVERAGE_EXTRACT_PROMPT == before


# --- whole-source coverage: G-Eval (one-call identify + classify) -----------


def test_geval_prompt_is_one_call_context_plus_output_same_rubric():
    messages = render_coverage_geval_prompt(context="SRC", output="OUT")
    user = messages[1]["content"]
    assert "[CONTEXT]\nSRC" in user and "[OUTPUT]\nOUT" in user
    system = " ".join(messages[0]["content"].split())
    # same source-item rubric as DAG Stage 1
    assert "approximately 10" in system
    assert "NOT a hard maximum" in system
    assert "independently satisfiable" in system
    # plus classification, and Python owns the score
    assert "meaningfully_present" in system and "fully_present" in system
    assert "Do not return an aggregate score" in system


def test_geval_compact_and_verbose_schemas():
    compact = COVERAGE_GEVAL_SCHEMA_COMPACT["properties"]["items"]["items"]
    verbose = COVERAGE_GEVAL_SCHEMA_VERBOSE["properties"]["items"]["items"]
    base = {"source_item", "meaningfully_present", "fully_present"}
    assert set(compact["properties"]) == base
    assert compact["required"] == ["source_item", "meaningfully_present", "fully_present"]
    assert set(verbose["properties"]) == base | {"reason"}
    assert "reason" in verbose["required"]
