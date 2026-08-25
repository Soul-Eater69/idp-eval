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


def _evaluate(items, *, verbose=False, max_items=None):
    judge = Judge({"items": items})
    result = CoverageEvaluator(
        judge, verbose=verbose, max_items=max_items
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
    )


def test_exactly_one_call_and_structured_context_output_rendering():
    result, judge = _evaluate([_item("Retain SSO", True, True)])
    assert result.metric == "coverage"
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
        assert "identify all materially distinct source items" in system
        assert "maxItems" not in judge.calls[0]["schema"]["properties"]["items"]
    else:
        assert f"select at most {max_items}" in system
        assert "return only those that actually exist" in system
        assert "Do not invent, duplicate, or artificially split items" in system
        assert judge.calls[0]["schema"]["properties"]["items"]["maxItems"] == max_items
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
    result, judge = _evaluate([])
    assert len(judge.calls) == 1
    assert result.score is None
    assert result.label == "not_applicable"
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
    with pytest.raises(ValueError, match="covered items must use an empty"):
        _evaluate([_item("A", True, True, "not empty")], verbose=True)
    with pytest.raises(ValueError, match="must include a non-empty"):
        _evaluate([_item("A", True, False, "")], verbose=True)


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


def test_invalid_binary_combination_fails_clearly():
    with pytest.raises(ValueError, match="fully_present=True requires"):
        _evaluate([_item("A", False, True)])


@pytest.mark.parametrize(
    "response,match",
    [
        ({}, "missing `items`"),
        ({"items": None}, "must be a list"),
        ({"items": ["bad"]}, "expected an object"),
        ({"items": [_item("", True, True)]}, "non-empty string"),
        (
            {
                "items": [
                    {
                        "source_item": "A",
                        "meaningfully_present": "yes",
                        "fully_present": False,
                    }
                ]
            },
            "must be booleans",
        ),
    ],
)
def test_malformed_response_fails_clearly(response, match):
    with pytest.raises(ValueError, match=match):
        CoverageEvaluator(Judge(response)).evaluate(CASE)


def test_framework_class_construction_and_evaluate_groups_use_one_call_per_output():
    judge = Judge({"items": [_item("A", True, True)]})
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
        "items": [
            _item("A", True, True),
            _item("B", True, False),
            _item("C", False, False),
        ]
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
