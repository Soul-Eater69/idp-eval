"""Coverage batching, verbose invariance, compact output, and diagnostics.

All offline: a scripted judge parses the classify prompt to answer per id. No LLM,
Phoenix, or gateway calls.
"""

import json

import pytest

from idp_eval import EvaluationCase, SourceCoverageEvaluator, TaskCoverageEvaluator

SRC_CASE = EvaluationCase(context="Source document.", output="Generated output.")
TASK_CASE = EvaluationCase(input="Task.", context="Context.", output="Output.")


def _requirements_from_prompt(user_content: str) -> list[dict]:
    block = user_content.split("[REQUIREMENTS]\n", 1)[1].split("\n\n[OUTPUT]", 1)[0]
    return json.loads(block)


class ScriptedBatchJudge:
    """Extraction returns fixed items; classification answers per requested id.

    ``status_map`` maps stable id -> (meaningfully_present, fully_present). The
    same judge serves any number of classification batches because it recomputes
    its answer from the ids present in each prompt.
    """

    def __init__(self, items, status_map, *, reason_prefix=None):
        self.items = list(items)
        self.status_map = dict(status_map)
        self.reason_prefix = reason_prefix
        self.extract_calls = 0
        self.classify_calls = 0
        self.classify_schemas: list[dict] = []
        self.classify_prompts: list[str] = []

    def generate_object(self, prompt, schema):
        user = prompt[1]["content"]
        if "[REQUIREMENTS]" not in user:
            self.extract_calls += 1
            props = schema["properties"]
            if "source_items" in props:
                return {"source_items": [{"source_item": t} for t in self.items]}
            return {"requirements": [{"requirement": t} for t in self.items]}

        self.classify_calls += 1
        self.classify_schemas.append(schema)
        self.classify_prompts.append(prompt[0]["content"])
        answers = []
        for req in _requirements_from_prompt(user):
            mp, fp = self.status_map[req["id"]]
            entry = {"id": req["id"], "meaningfully_present": mp, "fully_present": fp}
            if self.reason_prefix and not (mp and fp):
                entry["reason"] = f"{self.reason_prefix}:{req['id']}"
            answers.append(entry)
        return {"requirements": answers}


def _items_and_status(n, prefix, pattern=None):
    """n distinct items and a status_map keyed by ``{prefix}{i}`` (1-based)."""
    items = [f"item number {i}" for i in range(1, n + 1)]
    status = {}
    for i in range(1, n + 1):
        if pattern is None:
            status[f"{prefix}{i}"] = (True, True)
        else:
            status[f"{prefix}{i}"] = pattern(i)
    return items, status


# --- batching (item 27) -----------------------------------------------------


@pytest.mark.parametrize(
    "n,batch_size,expected_batches",
    [(1, 15, 1), (15, 15, 1), (16, 15, 2), (31, 15, 3), (27, 12, 3)],
)
def test_batch_count(n, batch_size, expected_batches):
    items, status = _items_and_status(n, "s")
    judge = ScriptedBatchJudge(items, status)
    result = SourceCoverageEvaluator(
        judge, classification_batch_size=batch_size
    ).evaluate(SRC_CASE)
    assert judge.classify_calls == expected_batches
    assert result.details["batch_count"] == expected_batches
    assert result.details["total_items"] == n
    # Every id classified exactly once, original order restored.
    ids = [i["id"] for i in result.details["items"]]
    assert ids == [f"s{i}" for i in range(1, n + 1)]


def test_empty_extraction_makes_no_classify_call():
    judge = ScriptedBatchJudge([], {})
    result = SourceCoverageEvaluator(judge).evaluate(SRC_CASE)
    assert result.label == "not_applicable"
    assert judge.classify_calls == 0
    assert result.details == {
        "total_items": 0,
        "covered_count": 0,
        "partial_count": 0,
        "missing_count": 0,
        "items": [],
        "final_item_count": 0,
        "batch_count": 0,
        "judge_call_count": 1,
    }


# --- config defaults & validation (items 1, 16, 19 A-D) ---------------------


def test_default_classification_batch_size_is_12():
    from idp_eval.evaluators.coverage_base import (
        DEFAULT_CLASSIFICATION_BATCH_SIZE,
    )

    assert DEFAULT_CLASSIFICATION_BATCH_SIZE == 12
    # 13 items with the default -> 2 batches (ceil(13/12)).
    items, status = _items_and_status(13, "s")
    judge = ScriptedBatchJudge(items, status)
    result = SourceCoverageEvaluator(judge).evaluate(SRC_CASE)
    assert result.details["batch_size"] == 12
    assert result.details["batch_count"] == 2


def test_custom_batch_size_still_works():
    items, status = _items_and_status(10, "s")
    judge = ScriptedBatchJudge(items, status)
    result = SourceCoverageEvaluator(judge, classification_batch_size=4).evaluate(
        SRC_CASE
    )
    assert judge.classify_calls == 3  # ceil(10/4)
    assert result.details["batch_size"] == 4


@pytest.mark.parametrize("bad", [0, -1, 1.5, "12", True, None])
def test_invalid_batch_size_rejected(bad):
    with pytest.raises(ValueError, match="classification_batch_size must be a"):
        SourceCoverageEvaluator(object(), classification_batch_size=bad)


# --- judge_call_count (item 19 G) -------------------------------------------


@pytest.mark.parametrize(
    "n,batch_size,expected_calls",
    [(0, 12, 1), (5, 12, 2), (31, 12, 4)],  # 1 + batch_count (0,1,3)
)
def test_judge_call_count(n, batch_size, expected_calls):
    items, status = _items_and_status(n, "s")
    judge = ScriptedBatchJudge(items, status)
    result = SourceCoverageEvaluator(
        judge, classification_batch_size=batch_size
    ).evaluate(SRC_CASE)
    assert result.details["judge_call_count"] == expected_calls
    assert judge.extract_calls + judge.classify_calls == expected_calls


# --- compact explanation is deterministic (items 3, 19 E-F) -----------------


def test_compact_explanation_forms_are_deterministic():
    # complete
    items, status = _items_and_status(3, "s")  # all covered
    complete = SourceCoverageEvaluator(ScriptedBatchJudge(items, status)).evaluate(
        SRC_CASE
    )
    assert complete.explanation == "All 3 source items are fully represented."
    # none
    items, status = _items_and_status(4, "s", lambda i: (False, False))
    none = SourceCoverageEvaluator(ScriptedBatchJudge(items, status)).evaluate(
        SRC_CASE
    )
    assert none.explanation == "None of the 4 source items are represented."
    # mixed
    items, status = _items_and_status(
        3, "s", lambda i: [(True, True), (True, False), (False, False)][i - 1]
    )
    mixed = SourceCoverageEvaluator(ScriptedBatchJudge(items, status)).evaluate(
        SRC_CASE
    )
    assert mixed.explanation == (
        "1 of 3 source items are fully covered; 1 partial and 1 missing."
    )


def test_no_extra_judge_call_for_explanation():
    items, status = _items_and_status(3, "s")
    judge = ScriptedBatchJudge(items, status)
    SourceCoverageEvaluator(judge).evaluate(SRC_CASE)
    # 1 extract + 1 classify only; explanation is computed in Python.
    assert judge.extract_calls == 1 and judge.classify_calls == 1


def test_docs_and_code_make_no_60s_guarantee():
    import idp_eval.evaluators.coverage_base as base
    import idp_eval.prompts.coverage_classify as classify

    for text in (base.__doc__, classify.__doc__):
        # Normalize whitespace so line wrapping doesn't affect phrase checks.
        normalized = " ".join(text.lower().split())
        assert "well under" not in normalized
        assert "under 60" not in normalized
        # Any use of "guarantee" must be a disclaimer ("not a guarantee").
        if "guarantee" in normalized:
            assert "not a guarantee" in normalized


def test_batching_score_matches_single_call_equivalent():
    # 20 items with a mixed pattern; batched (size 7 -> 3 batches) vs one batch.
    pattern = lambda i: [(True, True), (True, False), (False, False)][i % 3]
    items, status = _items_and_status(20, "s", pattern)

    batched = SourceCoverageEvaluator(
        ScriptedBatchJudge(items, status), classification_batch_size=7
    ).evaluate(SRC_CASE)
    single = SourceCoverageEvaluator(
        ScriptedBatchJudge(items, status), classification_batch_size=1000
    ).evaluate(SRC_CASE)

    assert batched.details["batch_count"] == 3
    assert single.details["batch_count"] == 1
    # Batch boundaries have no scoring effect.
    assert batched.score == single.score
    assert [i["status"] for i in batched.details["items"]] == [
        i["status"] for i in single.details["items"]
    ]


def test_batch_missing_id_fails():
    items, status = _items_and_status(16, "s")

    class DropsLastId(ScriptedBatchJudge):
        def generate_object(self, prompt, schema):
            response = super().generate_object(prompt, schema)
            user = prompt[1]["content"]
            if "[REQUIREMENTS]" in user and any(
                r["id"] == "s16" for r in _requirements_from_prompt(user)
            ):
                response["requirements"] = [
                    r for r in response["requirements"] if r["id"] != "s16"
                ]
            return response

    with pytest.raises(ValueError, match="Missing classification"):
        SourceCoverageEvaluator(
            DropsLastId(items, status), classification_batch_size=15
        ).evaluate(SRC_CASE)


def test_batch_duplicate_id_across_batches_fails():
    items, status = _items_and_status(16, "s")

    class DuplicatesFirstId(ScriptedBatchJudge):
        def generate_object(self, prompt, schema):
            response = super().generate_object(prompt, schema)
            user = prompt[1]["content"]
            if "[REQUIREMENTS]" in user and any(
                r["id"] == "s16" for r in _requirements_from_prompt(user)
            ):
                response["requirements"].append(
                    {"id": "s1", "meaningfully_present": True, "fully_present": True}
                )
            return response

    with pytest.raises(ValueError, match="Duplicate requirement id"):
        SourceCoverageEvaluator(
            DuplicatesFirstId(items, status), classification_batch_size=15
        ).evaluate(SRC_CASE)


# --- verbose invariance (item 25) -------------------------------------------


@pytest.mark.parametrize(
    "evaluator_cls,case,prefix",
    [
        (SourceCoverageEvaluator, SRC_CASE, "s"),
        (TaskCoverageEvaluator, TASK_CASE, "r"),
    ],
)
def test_verbose_matches_compact_score_and_statuses(evaluator_cls, case, prefix):
    pattern = lambda i: [(True, True), (True, False), (False, False)][i % 3]
    items, status = _items_and_status(6, prefix, pattern)

    compact = evaluator_cls(
        ScriptedBatchJudge(items, status), verbose=False
    ).evaluate(case)
    verbose = evaluator_cls(
        ScriptedBatchJudge(items, status, reason_prefix="why"), verbose=True
    ).evaluate(case)

    assert compact.score == verbose.score
    assert compact.label == verbose.label

    def _fingerprint(result):
        # Everything except the diagnostic reason must be identical.
        return [
            (i["id"], i["meaningfully_present"], i["fully_present"],
             i["status"], i["score"])
            for i in result.details["items"]
        ]

    assert _fingerprint(compact) == _fingerprint(verbose)
    assert compact.details["total_items" if prefix == "s" else "total_requirements"] \
        == verbose.details["total_items" if prefix == "s" else "total_requirements"]
    # Only diagnostic reason content differs.
    assert all(i["reason"] == "" for i in compact.details["items"])
    verbose_reasons = {
        i["status"]: i["reason"] for i in verbose.details["items"]
    }
    assert verbose_reasons["partial"] and verbose_reasons["missing"]


# --- compact default vs verbose prompt/schema (item 26) ---------------------


def test_default_classify_uses_compact_schema_no_reason():
    items, status = _items_and_status(3, "s")
    judge = ScriptedBatchJudge(items, status)
    SourceCoverageEvaluator(judge).evaluate(SRC_CASE)
    item_schema = judge.classify_schemas[0]["properties"]["requirements"]["items"]
    assert "reason" not in item_schema["properties"]


def test_verbose_classify_uses_verbose_schema_and_prompt():
    items, status = _items_and_status(3, "s")
    judge = ScriptedBatchJudge(items, status, reason_prefix="why")
    result = SourceCoverageEvaluator(judge, verbose=True).evaluate(SRC_CASE)
    item_schema = judge.classify_schemas[0]["properties"]["requirements"]["items"]
    assert "reason" in item_schema["properties"]
    assert "one-sentence" in judge.classify_prompts[0]
    # Overall explanation still present and Python-generated (3 items all covered).
    assert result.explanation == "All 3 source items are fully represented."


# --- diagnostics / timeout-shape (item 29) ----------------------------------


def test_large_denominator_flagged_and_split():
    items, status = _items_and_status(27, "s")
    judge = ScriptedBatchJudge(items, status)
    result = SourceCoverageEvaluator(judge, classification_batch_size=15).evaluate(
        SRC_CASE
    )
    assert result.details["large_denominator"] is True
    assert result.details["final_item_count"] == 27
    assert result.details["batch_count"] == 2
    # No extra judge calls purely for diagnostics: 1 extract + 2 classify.
    assert judge.extract_calls == 1 and judge.classify_calls == 2


def test_latency_fields_present_and_non_negative():
    items, status = _items_and_status(3, "s")
    result = SourceCoverageEvaluator(ScriptedBatchJudge(items, status)).evaluate(
        SRC_CASE
    )
    for field in ("extract_ms", "classify_ms", "total_ms"):
        assert field in result.details
        assert isinstance(result.details[field], (int, float))
        assert result.details[field] >= 0


def test_diagnostics_do_not_change_scored_shape_keys():
    items, status = _items_and_status(2, "s", lambda i: (True, i == 1))
    result = SourceCoverageEvaluator(ScriptedBatchJudge(items, status)).evaluate(
        SRC_CASE
    )
    # Core scoring keys remain; diagnostics are additive.
    for key in ("total_items", "covered_count", "partial_count", "missing_count",
                "items"):
        assert key in result.details
