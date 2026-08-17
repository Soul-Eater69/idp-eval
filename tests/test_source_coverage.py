"""SourceCoverageEvaluator tests using a scripted fake judge (no real LLM)."""

import pytest

from idp_eval import EvaluationCase, SourceCoverageEvaluator
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


def _extract(*items: str) -> dict:
    return {"source_items": [{"source_item": s} for s in items]}


def _classify(*entries) -> dict:
    return {
        "requirements": [
            {"id": rid, "meaningfully_present": p, "fully_present": f, "reason": "r"}
            for (rid, p, f) in entries
        ]
    }


CASE = EvaluationCase(
    input="TASK_SHOULD_BE_IGNORED",
    context="Retain the identity provider. SSO must remain supported.",
    output="UNIQUE_OUTPUT keeps the identity provider.",
)


def _run(judge) -> EvaluationResult:
    return SourceCoverageEvaluator(llm=judge).evaluate(CASE)


def test_metric_name_is_source_coverage():
    judge = ScriptedJudge(_extract("a"), _classify(("s1", True, True)))
    assert _run(judge).metric == "source_coverage"


def test_stage1_receives_context_only_no_input_no_output():
    judge = ScriptedJudge(_extract("a"), _classify(("s1", True, True)))
    _run(judge)
    extract_user = judge.calls[0]["prompt"][1]["content"]
    assert CASE.context in extract_user
    assert CASE.input not in extract_user      # task/input never sent
    assert CASE.output not in extract_user     # output never visible to extraction


def test_stage2_receives_items_and_output_not_the_task():
    judge = ScriptedJudge(
        _extract("Retain the identity provider."),
        _classify(("s1", True, True)),
    )
    _run(judge)
    classify_user = judge.calls[1]["prompt"][1]["content"]
    assert CASE.output in classify_user
    assert "Retain the identity provider." in classify_user
    assert CASE.input not in classify_user     # source coverage is task-agnostic


def test_covered_partial_missing_scoring():
    judge = ScriptedJudge(
        _extract("a", "b", "c"),
        _classify(("s1", True, True), ("s2", True, False), ("s3", False, False)),
    )
    result = _run(judge)
    assert result.score == (1.0 + 0.5 + 0.0) / 3
    statuses = [i["status"] for i in result.details["items"]]
    assert statuses == ["covered", "partial", "missing"]
    assert result.details["total_items"] == 3
    assert result.details["covered_count"] == 1
    # Items use the neutral "source_item" field, not "requirement".
    assert "source_item" in result.details["items"][0]
    assert "requirement" not in result.details["items"][0]


def test_empty_extraction_is_not_applicable_and_skips_stage_two():
    judge = ScriptedJudge(_extract())  # only one response queued
    result = _run(judge)
    assert result.score is None
    assert result.label == "not_applicable"
    assert result.details == {
        "total_items": 0,
        "covered_count": 0,
        "partial_count": 0,
        "missing_count": 0,
        "items": [],
    }
    assert len(judge.calls) == 1  # Stage 2 skipped


def test_normalized_exact_dedup():
    judge = ScriptedJudge(
        _extract("Retain the IdP.", " retain   the idp. ", "RETAIN THE IDP."),
        _classify(("s1", True, True)),
    )
    result = _run(judge)
    assert result.details["total_items"] == 1
    assert result.details["items"][0]["source_item"] == "Retain the IdP."


def test_stable_ids():
    judge = ScriptedJudge(
        _extract("first", "second"),
        _classify(("s2", True, False), ("s1", True, True)),  # returned reordered
    )
    items = _run(judge).details["items"]
    assert [i["id"] for i in items] == ["s1", "s2"]  # reconstructed in order
    assert items[0]["source_item"] == "first"


def test_malformed_classifier_missing_id_raises():
    judge = ScriptedJudge(_extract("a", "b"), _classify(("s1", True, True)))
    with pytest.raises(ValueError, match="Missing classification"):
        _run(judge)


def test_malformed_classifier_non_boolean_raises():
    judge = ScriptedJudge(
        _extract("a"),
        {"requirements": [{"id": "s1", "meaningfully_present": "yes",
                           "fully_present": False, "reason": "r"}]},
    )
    with pytest.raises(ValueError, match="Non-boolean"):
        _run(judge)
