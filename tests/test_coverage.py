"""Two-stage coverage evaluator tests using a scripted fake judge (no real LLM)."""

import pytest

from idp_eval import CoverageEvaluator, EvaluationCase
from idp_eval.models import EvaluationResult


class ScriptedJudge:
    """Returns queued structured responses in order and records each call."""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def generate_object(self, prompt, schema: dict) -> dict:
        self.calls.append({"prompt": prompt, "schema": schema})
        if not self._responses:
            raise AssertionError("Unexpected extra judge call")
        return self._responses.pop(0)


def _extract(*requirements: str) -> dict:
    return {"requirements": [{"requirement": r} for r in requirements]}


def _classify(*entries) -> dict:
    """entries: (id, meaningfully_present, fully_present) tuples."""
    return {
        "requirements": [
            {
                "id": rid,
                "meaningfully_present": present,
                "fully_present": full,
                "reason": "r",
            }
            for (rid, present, full) in entries
        ]
    }


CASE = EvaluationCase(
    input="Generate a Jira Epic from the provided Theme.",
    context="Theme with onboarding goals.",
    output="UNIQUE_OUTPUT_TOKEN epic body.",
)


def _run(judge) -> EvaluationResult:
    return CoverageEvaluator(llm=judge).evaluate(CASE)


# --- aggregation & binary-to-status logic -----------------------------------


def test_full_coverage():
    judge = ScriptedJudge(
        _extract("a", "b"),
        _classify(("r1", True, True), ("r2", True, True)),
    )
    result = _run(judge)
    assert result.score == 1.0
    assert result.details["covered_count"] == 2


def test_complete_omission():
    judge = ScriptedJudge(
        _extract("a", "b"),
        _classify(("r1", False, False), ("r2", False, False)),
    )
    result = _run(judge)
    assert result.score == 0.0
    assert result.details["missing_count"] == 2


def test_mixed_deterministic_aggregation():
    judge = ScriptedJudge(
        _extract("a", "b", "c", "d"),
        _classify(
            ("r1", True, True),    # covered  -> 1.0
            ("r2", False, False),  # missing  -> 0.0
            ("r3", True, False),   # partial  -> 0.5
            ("r4", True, True),    # covered  -> 1.0
        ),
    )
    result = _run(judge)
    assert result.score == 0.625
    assert "62.5%" in result.explanation
    assert [i["status"] for i in result.details["items"]] == [
        "covered",
        "missing",
        "partial",
        "covered",
    ]
    assert [i["score"] for i in result.details["items"]] == [1.0, 0.0, 0.5, 1.0]


def test_binary_combinations_map_to_status():
    judge = ScriptedJudge(
        _extract("present-full", "present-partial", "absent"),
        _classify(
            ("r1", True, True),
            ("r2", True, False),
            ("r3", False, False),
        ),
    )
    items = _run(judge).details["items"]
    assert items[0]["status"] == "covered"
    assert items[1]["status"] == "partial"
    assert items[2]["status"] == "missing"


def test_invalid_binary_combination_fails():
    judge = ScriptedJudge(
        _extract("a"),
        _classify(("r1", False, True)),  # meaningfully_present=False + full=True
    )
    with pytest.raises(ValueError, match="Invalid coverage classification"):
        _run(judge)


def test_non_boolean_classification_fails():
    judge = ScriptedJudge(
        _extract("a"),
        {"requirements": [{"id": "r1", "meaningfully_present": "yes",
                           "fully_present": False, "reason": "r"}]},
    )
    with pytest.raises(ValueError, match="Non-boolean"):
        _run(judge)


# --- extraction: dedup, empty, malformed ------------------------------------


def test_normalized_exact_dedup_after_extraction():
    judge = ScriptedJudge(
        _extract("Reduce onboarding time", " reduce   onboarding time ",
                 "REDUCE ONBOARDING TIME"),
        _classify(("r1", True, True)),
    )
    result = _run(judge)
    assert result.details["total_requirements"] == 1
    assert result.details["items"][0]["requirement"] == "Reduce onboarding time"


def test_empty_extraction_is_not_applicable_and_skips_stage_two():
    judge = ScriptedJudge(_extract())  # only one response queued
    result = _run(judge)
    assert result.score is None
    assert result.label == "not_applicable"
    assert result.details == {
        "total_requirements": 0,
        "covered_count": 0,
        "partial_count": 0,
        "missing_count": 0,
        "items": [],
    }
    # Stage 2 was never called.
    assert len(judge.calls) == 1


def test_malformed_extraction_missing_key_fails():
    judge = ScriptedJudge({"requirements": [{"note": "no requirement field"}]})
    with pytest.raises(KeyError):
        _run(judge)


# --- two-call behavior & stage isolation ------------------------------------


def test_two_calls_for_non_empty_and_stage_isolation():
    judge = ScriptedJudge(
        _extract("a", "b"),
        _classify(("r1", True, True), ("r2", False, False)),
    )
    _run(judge)
    assert len(judge.calls) == 2

    extract_user = judge.calls[0]["prompt"][1]["content"]
    assert CASE.input in extract_user
    assert CASE.context in extract_user
    assert CASE.output not in extract_user  # Stage 1 must NOT see output

    classify_user = judge.calls[1]["prompt"][1]["content"]
    assert CASE.output in classify_user
    assert '"id": "r1"' in classify_user  # fixed requirement set passed in
    assert CASE.context not in classify_user  # full context not leaked


# --- requirement id integrity -----------------------------------------------


def test_missing_id_fails():
    judge = ScriptedJudge(_extract("a", "b"), _classify(("r1", True, True)))
    with pytest.raises(ValueError, match="Missing classification"):
        _run(judge)


def test_unknown_id_fails():
    judge = ScriptedJudge(
        _extract("a"),
        _classify(("r1", True, True), ("r2", True, True)),
    )
    with pytest.raises(ValueError, match="Unknown requirement id"):
        _run(judge)


def test_duplicate_id_fails():
    judge = ScriptedJudge(
        _extract("a"),
        _classify(("r1", True, True), ("r1", False, False)),
    )
    with pytest.raises(ValueError, match="Duplicate requirement id"):
        _run(judge)


def test_reordered_ids_are_reconstructed_in_original_order():
    judge = ScriptedJudge(
        _extract("first", "second"),
        _classify(("r2", True, False), ("r1", True, True)),  # returned reversed
    )
    items = _run(judge).details["items"]
    assert [i["id"] for i in items] == ["r1", "r2"]
    assert items[0]["requirement"] == "first"
    assert items[0]["status"] == "covered"
    assert items[1]["status"] == "partial"


# --- semantic expectations (via fake judgments) -----------------------------


def test_paraphrase_counts_as_covered():
    judge = ScriptedJudge(
        _extract("Automate identity verification."),
        _classify(("r1", True, True)),
    )
    assert _run(judge).score == 1.0


def test_qualifier_omission_is_partial():
    judge = ScriptedJudge(
        _extract("Reduce verification effort by 40%."),
        _classify(("r1", True, False)),
    )
    assert _run(judge).score == 0.5


def test_unsupported_addition_does_not_lower_coverage():
    judge = ScriptedJudge(
        _extract("Goal 1.", "Goal 2."),
        _classify(("r1", True, True), ("r2", True, True)),
    )
    assert _run(judge).score == 1.0
