"""Final one-call CoverageEvaluator behavior (offline; no real LLM)."""

import inspect
import pytest

from idp_eval import CoverageEvaluator, EvaluationCase, EvaluationFramework


CASE = EvaluationCase(
    input="IGNORED_TASK",
    context={"requirements": ["Retain SSO", "Preserve MFA"]},
    output={"summary": "SSO remains available"},
)


class Judge:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate_object(self, prompt, schema):
        self.calls.append({"prompt": prompt, "schema": schema})
        return self.response


def _item(text, present, full, reason=None):
    item = {
        "source_item": text,
        "meaningfully_present": present,
        "fully_present": full,
    }
    if reason is not None:
        item["reason"] = reason
    return item


def _response(items, *, reason_mode="overall", overall_reason="Semantic summary."):
    prepared = []
    for item in items:
        item = dict(item)
        if reason_mode == "overall":
            status_is_failure = not item["fully_present"]
            item.setdefault("reason", "Diagnostic failure." if status_is_failure else "")
        elif reason_mode == "per_item":
            item.setdefault("reason", "Concise item reason.")
        prepared.append(item)
    response = {"items": prepared}
    if reason_mode != "none":
        response["overall_reason"] = overall_reason
    return response


def _evaluate(
    items,
    *,
    verbose=False,
    max_items=None,
    reason_mode="overall",
    overall_reason="Semantic summary.",
):
    judge = Judge(
        _response(
            items, reason_mode=reason_mode, overall_reason=overall_reason
        )
    )
    result = CoverageEvaluator(
        judge,
        verbose=verbose,
        max_items=max_items,
        reason_mode=reason_mode,
    ).evaluate(CASE)
    return result, judge


def test_public_api_requires_context_and_output_only():
    assert CoverageEvaluator.required_fields == ("context", "output")
    CoverageEvaluator(object()).validate_case(EvaluationCase(context="c", output="o"))
    with pytest.raises(ValueError, match="requires non-empty `context`"):
        CoverageEvaluator(object()).evaluate(EvaluationCase(output="o"))
    with pytest.raises(ValueError, match="requires non-empty `output`"):
        CoverageEvaluator(object()).evaluate(EvaluationCase(context="c"))


def test_public_constructor_exposes_optional_item_limit():
    assert tuple(inspect.signature(CoverageEvaluator).parameters) == (
        "llm",
        "verbose",
        "max_items",
        "reason_mode",
    )


def test_reason_mode_default_and_validation_before_judge_work():
    assert CoverageEvaluator()._reason_mode == "overall"
    for mode in ("overall", "per_item", "none"):
        assert CoverageEvaluator(reason_mode=mode)._reason_mode == mode
    judge = Judge({"items": []})
    with pytest.raises(ValueError, match="reason_mode"):
        CoverageEvaluator(judge, reason_mode="invalid")
    assert judge.calls == []


def test_exactly_one_call_and_structured_context_output_rendering():
    result, judge = _evaluate([_item("Retain SSO", True, True)])
    assert result.metric == "coverage"
    assert result.explanation == "Semantic summary."
    assert len(judge.calls) == 1
    user = judge.calls[0]["prompt"][1]["content"]
    assert "Requirements:\n- Retain SSO\n- Preserve MFA" in user
    assert "Summary: SSO remains available" in user
    assert "IGNORED_TASK" not in user


@pytest.mark.parametrize("max_items", [None, 1, 5])
def test_item_limit_prompt_and_details_preserve_one_call(max_items):
    result, judge = _evaluate(
        [_item("Only real item", True, True)], max_items=max_items
    )
    system = judge.calls[0]["prompt"][0]["content"]
    if max_items is None:
        assert "all materially distinct, reasonably atomic" in system
        assert "maxItems" not in judge.calls[0]["schema"]["properties"]["items"]
    else:
        assert f"select at most {max_items}" in system
        assert "return only those that actually exist" in system
        assert "Do not invent, duplicate, or artificially split items" in system
        assert "maxItems" not in judge.calls[0]["schema"]["properties"]["items"]
    assert len(judge.calls) == 1
    assert result.details["max_items"] == max_items
    assert result.details["evaluated_items"] == 1


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "5"])
def test_invalid_item_limit_fails_before_judge_work(bad):
    judge = Judge({"items": []})
    with pytest.raises(ValueError, match="max_items"):
        CoverageEvaluator(judge, max_items=bad)
    assert judge.calls == []


def test_item_limit_rejects_over_limit_judge_response_without_truncating():
    with pytest.raises(ValueError, match="exceeds configured max_items=1"):
        _evaluate(
            [_item("A", True, True), _item("B", True, True)], max_items=1
        )


def test_unlimited_mode_scores_all_100_returned_items():
    items = [_item(f"Item {index}", True, True) for index in range(100)]
    result, judge = _evaluate(items)
    assert result.score == 1.0
    assert result.details["evaluated_items"] == 100
    assert len(judge.calls) == 1


@pytest.mark.parametrize(
    "present,full,score,label",
    [
        (True, True, 1.0, "covered"),
        (True, False, 0.5, "partial"),
        (False, False, 0.0, "missing"),
    ],
)
def test_binary_status_scoring_and_labels(present, full, score, label):
    result, _ = _evaluate([_item("A", present, full)])
    assert result.score == score
    assert result.label == label


def test_aggregate_mean_and_counts_are_deterministic_python():
    result, _ = _evaluate(
        [
            _item("A", True, True),
            _item("B", True, False),
            _item("C", False, False),
        ]
    )
    assert result.score == pytest.approx(0.5)
    assert result.details["covered_count"] == 1
    assert result.details["partial_count"] == 1
    assert result.details["missing_count"] == 1


def test_empty_items_is_not_applicable_after_one_call():
    result, judge = _evaluate(
        [], overall_reason="No important source items were identified."
    )
    assert len(judge.calls) == 1
    assert result.score is None
    assert result.label == "not_applicable"
    assert result.explanation == "No important source items were identified."
    assert result.details["final_item_count"] == 0


def test_empty_items_none_mode_is_not_applicable_without_explanation():
    result, judge = _evaluate([], reason_mode="none")
    assert len(judge.calls) == 1
    assert result.score is None
    assert result.label == "not_applicable"
    assert result.explanation is None
    assert result.details["final_item_count"] == 0


def test_compact_details_are_minimal_and_omit_items():
    result, _ = _evaluate([_item("A", True, True)])
    assert set(result.details) == {
        "final_item_count",
        "max_items",
        "evaluated_items",
        "covered_count",
        "partial_count",
        "missing_count",
        "judge_call_count",
        "total_ms",
        "verbose",
        "reason_mode",
    }
    assert result.details["judge_call_count"] == 1
    assert result.details["verbose"] is False


def test_verbose_details_include_full_audit_trail_and_reasons():
    result, _ = _evaluate(
        [
            _item("Covered", True, True, ""),
            _item("Partial", True, False, "Threshold missing."),
            _item("Missing", False, False, "Not represented."),
        ],
        verbose=True,
    )
    assert result.details["verbose"] is True
    assert [item["id"] for item in result.details["items"]] == ["S1", "S2", "S3"]
    assert [item["status"] for item in result.details["items"]] == [
        "covered",
        "partial",
        "missing",
    ]
    assert [item["item_score"] for item in result.details["items"]] == [
        1.0,
        0.5,
        0.0,
    ]
    assert result.details["items"][0]["reason"] == ""


def test_verbose_reason_contract_is_validated():
    with pytest.raises(ValueError, match="covered items must use"):
        _evaluate([_item("A", True, True, "not empty")])
    with pytest.raises(ValueError, match="must include a non-empty"):
        _evaluate([_item("A", True, False, "")])


def test_reason_modes_control_reason_contract_and_explanation():
    overall, overall_judge = _evaluate(
        [_item("Covered", True, True), _item("Missing", False, False, "Absent.")],
        verbose=True,
        overall_reason="The required MFA control is absent.",
    )
    assert overall.explanation == "The required MFA control is absent."
    assert len(overall_judge.calls) == 1

    per_item, per_item_judge = _evaluate(
        [_item("Covered", True, True, "Present in full.")],
        verbose=True,
        reason_mode="per_item",
        overall_reason="The authentication requirement is represented.",
    )
    assert per_item.details["items"][0]["reason"] == "Present in full."
    assert per_item.explanation == "The authentication requirement is represented."
    assert len(per_item_judge.calls) == 1

    none, none_judge = _evaluate(
        [_item("Covered", True, True)],
        verbose=True,
        reason_mode="none",
    )
    assert none.explanation is None
    assert "reason" not in none.details["items"][0]
    assert len(none_judge.calls) == 1


def test_overall_mode_does_not_require_reason_for_passing_item():
    judge = Judge(
        {
            "items": [_item("Covered", True, True)],
            "overall_reason": "The source requirement is represented.",
        }
    )
    result = CoverageEvaluator(judge, verbose=True).evaluate(CASE)
    assert result.details["items"][0]["reason"] == ""
    assert len(judge.calls) == 1


def test_per_item_requires_reason_for_every_item_and_overall_is_required():
    with pytest.raises(ValueError, match="every item"):
        CoverageEvaluator(
            Judge(
                {
                    "items": [_item("Covered", True, True, "")],
                    "overall_reason": "Summary.",
                }
            ),
            reason_mode="per_item",
        ).evaluate(CASE)
    with pytest.raises(ValueError, match="overall_reason"):
        CoverageEvaluator(
            Judge({"items": [_item("Covered", True, True, "")]})
        ).evaluate(CASE)


def test_normalized_exact_dedup_keeps_first_item():
    result, _ = _evaluate(
        [
            _item("Retain SSO", True, True, ""),
            _item(" retain   sso ", True, True, ""),
            _item("Preserve MFA", False, False, "Missing."),
        ],
        verbose=True,
    )
    assert result.details["final_item_count"] == 2
    assert [item["source_item"] for item in result.details["items"]] == [
        "Retain SSO",
        "Preserve MFA",
    ]


def test_normalized_exact_duplicate_with_conflicting_binary_judgment_fails():
    with pytest.raises(ValueError, match="duplicate normalized source item"):
        _evaluate(
            [
                _item("Retain SSO", True, True, ""),
                _item(" retain   sso ", True, False, "Qualifier absent."),
            ],
            verbose=True,
        )


def test_invalid_binary_combination_fails_clearly():
    with pytest.raises(ValueError, match="fully_present=True requires"):
        _evaluate([_item("A", False, True)])


@pytest.mark.parametrize(
    "response,match",
    [
        ({}, "expected exactly"),
        ({"items": None, "overall_reason": "Summary."}, "must be a list"),
        ({"items": ["bad"], "overall_reason": "Summary."}, "expected an object"),
        (
            _response([_item("", True, True)]),
            "non-empty string",
        ),
        (
            {
                "items": [
                    {
                        "source_item": "A",
                        "meaningfully_present": "yes",
                        "fully_present": False,
                        "reason": "Invalid boolean.",
                    }
                ],
                "overall_reason": "Summary.",
            },
            "must be booleans",
        ),
    ],
)
def test_malformed_response_fails_clearly(response, match):
    with pytest.raises(ValueError, match=match):
        CoverageEvaluator(Judge(response)).evaluate(CASE)


def test_framework_class_construction_and_evaluate_groups_use_one_call_per_output():
    judge = Judge(_response([_item("A", True, True)]))
    framework = EvaluationFramework(evaluators=[CoverageEvaluator], judge=judge)
    results = framework.evaluate_groups(
        [{"context": "source", "outputs": ["one", "two"], "group_id": "g"}]
    )
    assert [result["coverage"].score for result in results] == [1.0, 1.0]
    assert len(judge.calls) == 2


def test_verbose_and_compact_scores_match():
    compact, _ = _evaluate(
        [
            _item("A", True, True),
            _item("B", True, False),
            _item("C", False, False),
        ]
    )
    verbose, _ = _evaluate(
        [
            _item("A", True, True, ""),
            _item("B", True, False, "Qualifier missing."),
            _item("C", False, False, "Not present."),
        ],
        verbose=True,
    )
    assert compact.score == verbose.score
    assert compact.label == verbose.label


def test_backend_type_does_not_affect_coverage_result():
    response = {
        **_response(
            [
                _item("A", True, True),
                _item("B", True, False),
                _item("C", False, False),
            ]
        )
    }

    class GatewayCompatibleJudge(Judge):
        pass

    class AzureCompatibleJudge(Judge):
        pass

    gateway = CoverageEvaluator(GatewayCompatibleJudge(response)).evaluate(CASE)
    azure = CoverageEvaluator(AzureCompatibleJudge(response)).evaluate(CASE)
    assert gateway.score == azure.score == 0.5
    assert gateway.label == azure.label == "partial"
    assert {
        key: value
        for key, value in gateway.details.items()
        if key != "total_ms"
    } == {
        key: value
        for key, value in azure.details.items()
        if key != "total_ms"
    }
