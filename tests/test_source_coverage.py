"""One-call SourceCoverageEvaluator tests (no real LLM)."""

import pytest

from idp_eval import EvaluationCase, SourceCoverageEvaluator


class ScriptedJudge:
    """Returns one structured response and records every call."""

    def __init__(self, response):
        self.response = response
        self.calls: list[dict] = []

    def generate_object(self, prompt, schema: dict) -> dict:
        self.calls.append({"prompt": prompt, "schema": schema})
        return self.response


def _response(*entries) -> dict:
    items = []
    for entry in entries:
        source_item, present, full, *reason = entry
        item = {
            "source_item": source_item,
            "meaningfully_present": present,
            "fully_present": full,
        }
        if reason:
            item["reason"] = reason[0]
        items.append(item)
    return {"items": items}


CASE = EvaluationCase(
    input="TASK_SHOULD_BE_IGNORED",
    context="Retain the identity provider. SSO must remain supported.",
    output="UNIQUE_OUTPUT keeps the identity provider.",
)


def _run(judge, **kwargs):
    return SourceCoverageEvaluator(llm=judge, **kwargs).evaluate(CASE)


def test_metric_name_and_exactly_one_judge_call():
    judge = ScriptedJudge(_response(("Retain the IdP.", True, True)))
    result = _run(judge)
    assert result.metric == "source_coverage"
    assert result.details["judge_call_count"] == 1
    assert len(judge.calls) == 1


def test_one_call_receives_context_and_output_but_not_input():
    judge = ScriptedJudge(_response(("Retain the IdP.", True, True)))
    _run(judge)
    user = judge.calls[0]["prompt"][1]["content"]
    assert CASE.context in user
    assert CASE.output in user
    assert CASE.input not in user


def test_covered_partial_missing_scoring_and_labels():
    judge = ScriptedJudge(
        _response(
            ("a", True, True),
            ("b", True, False),
            ("c", False, False),
        )
    )
    result = _run(judge)
    assert result.score == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert result.label == "incomplete"
    assert [item["status"] for item in result.details["items"]] == [
        "covered",
        "partial",
        "missing",
    ]
    assert [item["score"] for item in result.details["items"]] == [1.0, 0.5, 0.0]
    assert result.details["covered_count"] == 1
    assert result.details["partial_count"] == 1
    assert result.details["missing_count"] == 1


@pytest.mark.parametrize(
    "response,score,label,explanation",
    [
        (
            _response(("a", True, True)),
            1.0,
            "complete",
            "All 1 source items are fully represented.",
        ),
        (
            _response(("a", False, False)),
            0.0,
            "missing",
            "None of the 1 source items are represented.",
        ),
    ],
)
def test_complete_and_missing_labels(response, score, label, explanation):
    result = _run(ScriptedJudge(response))
    assert result.score == score
    assert result.label == label
    assert result.explanation == explanation


def test_empty_items_are_not_applicable_after_one_call():
    judge = ScriptedJudge({"items": []})
    result = _run(judge)
    assert result.score is None
    assert result.label == "not_applicable"
    assert result.details["total_items"] == 0
    assert result.details["items"] == []
    assert result.details["judge_call_count"] == 1
    assert len(judge.calls) == 1


def test_normalized_exact_dedup_keeps_first_and_assigns_stable_ids():
    judge = ScriptedJudge(
        _response(
            ("Retain the IdP.", True, True),
            (" retain   the idp. ", True, True),
            ("RETAIN THE IDP.", True, True),
            ("Support SSO.", False, False),
        )
    )
    items = _run(judge).details["items"]
    assert [item["id"] for item in items] == ["s1", "s2"]
    assert [item["source_item"] for item in items] == [
        "Retain the IdP.",
        "Support SSO.",
    ]


def test_compact_and_verbose_equivalent_outputs_have_identical_scores():
    compact = ScriptedJudge(
        _response(("a", True, True), ("b", True, False), ("c", False, False))
    )
    verbose = ScriptedJudge(
        _response(
            ("a", True, True, ""),
            ("b", True, False, "qualifier missing"),
            ("c", False, False, "not represented"),
        )
    )
    compact_result = _run(compact, verbose=False)
    verbose_result = _run(verbose, verbose=True)
    assert compact_result.score == verbose_result.score
    assert compact_result.label == verbose_result.label
    assert [item["status"] for item in compact_result.details["items"]] == [
        item["status"] for item in verbose_result.details["items"]
    ]
    assert all(not item["reason"] for item in compact_result.details["items"])
    assert verbose_result.details["items"][1]["reason"] == "qualifier missing"
    assert len(compact.calls) == len(verbose.calls) == 1


def test_compact_schema_omits_reason_and_verbose_schema_requires_it():
    compact = ScriptedJudge(_response(("a", True, True)))
    verbose = ScriptedJudge(_response(("a", True, True, "")))
    _run(compact)
    _run(verbose, verbose=True)
    compact_item = compact.calls[0]["schema"]["properties"]["items"]["items"]
    verbose_item = verbose.calls[0]["schema"]["properties"]["items"]["items"]
    assert "reason" not in compact_item["properties"]
    assert "reason" in verbose_item["properties"]
    assert "reason" in verbose_item["required"]


@pytest.mark.parametrize(
    "response,match",
    [
        ({}, "missing `items`"),
        ({"items": None}, "must be a list"),
        ({"items": ["bad"]}, "expected an object"),
        (
            {
                "items": [
                    {
                        "source_item": "",
                        "meaningfully_present": True,
                        "fully_present": True,
                    }
                ]
            },
            "non-empty string",
        ),
        (
            {
                "items": [
                    {
                        "source_item": "a",
                        "meaningfully_present": "yes",
                        "fully_present": False,
                    }
                ]
            },
            "must be booleans",
        ),
    ],
)
def test_malformed_output_raises_clearly(response, match):
    with pytest.raises(ValueError, match=match):
        _run(ScriptedJudge(response))


def test_verbose_output_requires_reason_field():
    with pytest.raises(ValueError, match="verbose output requires `reason`"):
        _run(ScriptedJudge(_response(("a", True, True))), verbose=True)


def test_logically_inconsistent_binary_result_raises():
    with pytest.raises(ValueError, match="Invalid coverage classification"):
        _run(ScriptedJudge(_response(("a", False, True))))


def test_timing_details_are_one_call_specific():
    result = _run(ScriptedJudge(_response(("a", True, True))))
    assert result.details["judge_call_count"] == 1
    assert result.details["final_item_count"] == 1
    assert result.details["verbose"] is False
    assert result.details["evaluate_ms"] >= 0
    assert result.details["total_ms"] >= 0
    for obsolete in ("extract_ms", "classify_ms", "batch_count", "batch_size"):
        assert obsolete not in result.details
