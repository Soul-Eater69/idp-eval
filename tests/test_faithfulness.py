"""Offline contract tests for one-call claim-level faithfulness."""

import asyncio
import inspect
import threading
import time

import pytest

from idp_eval import EvaluationCase, EvaluationFramework, FaithfulnessEvaluator
from idp_eval.prompts.faithfulness import (
    FAITHFULNESS_SCHEMA_NONE,
    FAITHFULNESS_SCHEMA_OVERALL,
    FAITHFULNESS_SCHEMA_PER_ITEM,
    render_faithfulness_prompt,
)


CASE = EvaluationCase(context="Refunds take five days.", output="Refunds are instant.")


class Judge:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def generate_object(self, prompt, schema):
        self.calls.append({"prompt": prompt, "schema": schema})
        return self.responses.pop(0)


def _claim(claim="A factual claim.", status="supported", reason=None):
    value = {"claim": claim, "status": status}
    if reason is not None:
        value["reason"] = reason
    return value


def _response(
    claims, *, reason_mode="overall", overall_reason="Semantic summary."
):
    prepared = []
    for claim in claims:
        claim = dict(claim)
        if reason_mode == "overall":
            claim.setdefault(
                "reason",
                "Unsupported by context."
                if claim["status"] == "unsupported"
                else "",
            )
        elif reason_mode == "per_item":
            claim.setdefault("reason", "Concise claim reason.")
        prepared.append(claim)
    response = {"claims": prepared}
    if reason_mode != "none":
        response["overall_reason"] = overall_reason
    return response


def test_public_constructor_uses_common_optional_item_limit():
    assert tuple(inspect.signature(FaithfulnessEvaluator).parameters) == (
        "llm",
        "verbose",
        "max_items",
        "reason_mode",
    )


def test_reason_mode_default_and_validation_before_judge_work():
    assert FaithfulnessEvaluator()._reason_mode == "overall"
    for mode in ("overall", "per_item", "none"):
        assert FaithfulnessEvaluator(reason_mode=mode)._reason_mode == mode
    judge = Judge({"claims": []})
    with pytest.raises(ValueError, match="reason_mode"):
        FaithfulnessEvaluator(judge, reason_mode="invalid")
    assert judge.calls == []


def test_one_sync_call_and_supported_scoring():
    judge = Judge(_response([_claim()]))
    result = FaithfulnessEvaluator(judge).evaluate(CASE)
    assert len(judge.calls) == 1
    assert result.score == 1.0 and result.label == "not_hallucinated"
    assert result.details["judge_call_count"] == 1
    assert result.details["claim_count"] == 1
    assert "claims" not in result.details
    assert result.explanation == "Semantic summary."


@pytest.mark.parametrize("max_items", [None, 1, 5])
def test_claim_limit_prompt_details_and_fewer_than_limit(max_items):
    judge = Judge(_response([_claim()]))
    result = FaithfulnessEvaluator(judge, max_items=max_items).evaluate(CASE)
    system = judge.calls[0]["prompt"][0]["content"]
    if max_items is None:
        assert "all materially distinct, reasonably atomic" in system
        assert "maxItems" not in judge.calls[0]["schema"]["properties"]["claims"]
    else:
        assert f"select at most {max_items}" in system
        assert "Examine the complete OUTPUT before selecting" in system
        assert f"Do not stop after finding the first {max_items}" in system
        assert "independently of whether CONTEXT will classify them" in system
        assert "CONTEXT is for support judgment, not claim selection" in system
        assert "return only those that actually exist" in system
        assert "Do not invent, duplicate, or artificially split claims" in system
        assert "maxItems" not in judge.calls[0]["schema"]["properties"]["claims"]
    assert len(judge.calls) == 1
    assert result.details["max_items"] == max_items
    assert result.details["evaluated_claims"] == 1


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "5"])
def test_invalid_claim_limit_fails_before_judge_work(bad):
    judge = Judge({"claims": []})
    with pytest.raises(ValueError, match="max_items"):
        FaithfulnessEvaluator(judge, max_items=bad)
    assert judge.calls == []


def test_claim_limit_rejects_over_limit_judge_response_without_truncating():
    judge = Judge(_response([_claim("A"), _claim("B")]))
    with pytest.raises(ValueError, match="exceeds configured max_items=1"):
        FaithfulnessEvaluator(judge, max_items=1).evaluate(CASE)


def test_unlimited_mode_scores_all_100_returned_claims():
    claims = [_claim(f"Claim {index}.") for index in range(100)]
    judge = Judge(_response(claims))
    result = FaithfulnessEvaluator(judge).evaluate(CASE)
    assert result.score == 1.0
    assert result.details["evaluated_claims"] == 100
    assert len(judge.calls) == 1


def test_verbose_mixed_score_ids_audit_and_explanation():
    judge = Judge(
        _response(
            [
                _claim("Cancellation is allowed.", "supported", ""),
                _claim(
                    "Refunds are instant.",
                    "unsupported",
                    "Context says five days.",
                ),
            ],
            overall_reason=(
                "The instant-refund claim conflicts with the five-day policy."
            ),
        )
    )
    result = FaithfulnessEvaluator(judge, verbose=True).evaluate(CASE)
    assert result.score == 0.5 and result.label == "hallucinated"
    assert result.explanation == (
        "The instant-refund claim conflicts with the five-day policy."
    )
    assert [item["id"] for item in result.details["claims"]] == ["F1", "F2"]
    assert [item["item_score"] for item in result.details["claims"]] == [1.0, 0.0]


def test_all_unsupported_scores_zero():
    result = FaithfulnessEvaluator(
        Judge(_response([_claim(status="unsupported")])),
    ).evaluate(CASE)
    assert result.score == 0.0 and result.label == "hallucinated"


@pytest.mark.parametrize("verbose", [False, True])
def test_no_claims_is_not_applicable_after_one_call(verbose):
    judge = Judge(
        _response(
            [],
            overall_reason="No checkable factual claims were identified.",
        )
    )
    result = FaithfulnessEvaluator(judge, verbose=verbose).evaluate(CASE)
    assert result.score is None and result.label == "not_applicable"
    assert result.explanation == (
        "No checkable factual claims were identified."
    )
    assert result.details["judge_call_count"] == 1
    assert result.details["claim_count"] == 0
    assert len(judge.calls) == 1


def test_no_claims_none_mode_is_not_applicable_without_explanation():
    judge = Judge(_response([], reason_mode="none"))
    result = FaithfulnessEvaluator(judge, reason_mode="none").evaluate(CASE)
    assert result.score is None and result.label == "not_applicable"
    assert result.explanation is None
    assert result.details["claim_count"] == 0
    assert len(judge.calls) == 1


def test_exact_normalized_dedup_keeps_first_and_stable_ids():
    judge = Judge(
        _response(
            [
                _claim(
                    " Refunds   take five days. ",
                    "unsupported",
                    "First diagnostic.",
                ),
                _claim(
                    "REFUNDS TAKE FIVE DAYS.",
                    "unsupported",
                    "Different duplicate diagnostic.",
                ),
                _claim("Cancellation takes 24 hours.", "supported", ""),
            ]
        )
    )
    result = FaithfulnessEvaluator(judge, verbose=True).evaluate(CASE)
    assert result.details["claim_count"] == 2
    assert [c["id"] for c in result.details["claims"]] == ["F1", "F2"]
    assert result.details["claims"][0]["status"] == "unsupported"
    assert result.details["claims"][0]["reason"] == "First diagnostic."


def test_normalized_duplicate_claim_with_conflicting_status_fails():
    judge = Judge(
        _response(
            [
                _claim("Refunds take five days.", "supported", ""),
                _claim(
                    " refunds   take five days. ",
                    "unsupported",
                    "Conflicting duplicate.",
                ),
            ]
        )
    )
    with pytest.raises(ValueError, match="duplicate normalized claim"):
        FaithfulnessEvaluator(judge).evaluate(CASE)


def test_reason_modes_control_contract_explanation_and_one_call():
    overall_judge = Judge(
        _response(
            [_claim("Refunds are instant.", "unsupported")],
            overall_reason="The instant-refund claim conflicts with context.",
        )
    )
    overall = FaithfulnessEvaluator(overall_judge, verbose=True).evaluate(CASE)
    assert overall.explanation == "The instant-refund claim conflicts with context."
    assert overall.details["claims"][0]["reason"]
    assert len(overall_judge.calls) == 1

    per_item_judge = Judge(
        _response(
            [_claim("Refunds take five days.", "supported", "Context states it.")],
            reason_mode="per_item",
            overall_reason="The refund statement is grounded in context.",
        )
    )
    per_item = FaithfulnessEvaluator(
        per_item_judge, verbose=True, reason_mode="per_item"
    ).evaluate(CASE)
    assert per_item.details["claims"][0]["reason"] == "Context states it."
    assert per_item.explanation == "The refund statement is grounded in context."
    assert len(per_item_judge.calls) == 1

    none_judge = Judge(_response([_claim()], reason_mode="none"))
    none = FaithfulnessEvaluator(
        none_judge, verbose=True, reason_mode="none"
    ).evaluate(CASE)
    assert none.explanation is None
    assert "reason" not in none.details["claims"][0]
    assert len(none_judge.calls) == 1


def test_overall_mode_does_not_require_reason_for_supported_claim():
    judge = Judge(
        {
            "claims": [_claim("Refunds take five days.", "supported")],
            "overall_reason": "The refund timing is grounded in context.",
        }
    )
    result = FaithfulnessEvaluator(judge, verbose=True).evaluate(CASE)
    assert result.details["claims"][0]["reason"] == ""
    assert len(judge.calls) == 1


def test_reason_requirements_are_mode_specific():
    with pytest.raises(ValueError, match="unsupported claims"):
        FaithfulnessEvaluator(
            Judge(
                {
                    "claims": [_claim(status="unsupported", reason="")],
                    "overall_reason": "Unsupported refund timing.",
                }
            )
        ).evaluate(CASE)
    with pytest.raises(ValueError, match="every claim"):
        FaithfulnessEvaluator(
            Judge(
                {
                    "claims": [_claim(reason="")],
                    "overall_reason": "Refund timing is supported.",
                }
            ),
            reason_mode="per_item",
        ).evaluate(CASE)
    with pytest.raises(ValueError, match="overall_reason"):
        FaithfulnessEvaluator(Judge({"claims": [_claim(reason="")]})).evaluate(
            CASE
        )


@pytest.mark.parametrize("missing", ["context", "output"])
def test_required_fields_fail_before_judge(missing):
    judge = Judge({"claims": []})
    values = {"context": "source", "output": "answer"}
    values[missing] = None
    with pytest.raises(ValueError, match=f"requires non-empty `{missing}`"):
        FaithfulnessEvaluator(judge).evaluate(EvaluationCase(**values))
    assert judge.calls == []


def test_only_rendered_context_and_output_reach_prompt():
    judge = Judge(_response([_claim()]))
    case = EvaluationCase(
        input="IGNORE-INPUT", instructions="IGNORE-INSTRUCTIONS",
        context={"policy": ["Refund in five days"]},
        output=[{"summary": "Refund in five days"}],
        metadata={"secret": "IGNORE-METADATA"},
        retrieved_documents=["IGNORE-RETRIEVAL"],
    )
    FaithfulnessEvaluator(judge).evaluate(case)
    user = judge.calls[0]["prompt"][1]["content"]
    assert "Policy:\n- Refund in five days" in user
    assert "Summary: Refund in five days" in user
    assert all(
        marker not in user
        for marker in (
            "IGNORE-INPUT",
            "IGNORE-INSTRUCTIONS",
            "IGNORE-METADATA",
            "IGNORE-RETRIEVAL",
        )
    )


@pytest.mark.parametrize(
    "response,reason_mode,match",
    [
        ([], "overall", "expected an object"),
        ({}, "overall", "expected exactly"),
        ({"claims": [], "extra": 1}, "none", "expected exactly"),
        ({"claims": "bad"}, "none", "must be a list"),
        ({"claims": ["bad"]}, "none", "expected an object"),
        ({"claims": [{"claim": "x"}]}, "none", "unexpected or missing"),
        ({"claims": [_claim(reason="extra")]}, "none", "unexpected or missing"),
        ({"claims": [_claim(" ")]}, "none", "non-empty string"),
        ({"claims": [_claim(status="partial")]}, "none", "Unknown faithfulness"),
        (
            {
                "claims": [_claim(reason=3)],
                "overall_reason": "Summary.",
            },
            "overall",
            "must be a string",
        ),
        (
            {
                "claims": [_claim(status="unsupported", reason=" ")],
                "overall_reason": "Summary.",
            },
            "overall",
            "non-empty",
        ),
    ],
)
def test_strict_response_validation(response, reason_mode, match):
    with pytest.raises(ValueError, match=match):
        FaithfulnessEvaluator(
            Judge(response), reason_mode=reason_mode
        ).evaluate(CASE)


def test_reason_mode_schemas_are_strict_and_have_no_max_items():
    for schema, keys in (
        (FAITHFULNESS_SCHEMA_NONE, {"claim", "status"}),
        (FAITHFULNESS_SCHEMA_OVERALL, {"claim", "status", "reason"}),
        (FAITHFULNESS_SCHEMA_PER_ITEM, {"claim", "status", "reason"}),
    ):
        assert schema["additionalProperties"] is False
        assert schema["required"] == (
            ["claims"]
            if schema is FAITHFULNESS_SCHEMA_NONE
            else ["claims", "overall_reason"]
        )
        assert "maxItems" not in schema["properties"]["claims"]
        item = schema["properties"]["claims"]["items"]
        assert item["additionalProperties"] is False
        assert set(item["required"]) == keys
        assert set(item["properties"]) == keys


def test_prompt_contract_excludes_scores_and_distinguishes_coverage():
    system = render_faithfulness_prompt(
        context="c", output="o"
    )[0]["content"]
    normalized_system = " ".join(system.split())
    assert "OUTPUT -> CONTEXT" in system
    assert '"supported" or "unsupported"' in system
    assert "not a faithfulness failure; omissions belong to Coverage" in system
    assert "Do not return IDs, item scores, aggregate scores" in system
    assert "all materially distinct, reasonably atomic" in system
    assert "The item limit controls how many units are selected" in normalized_system


def test_reason_prompts_are_semantic_and_mode_specific():
    overall = render_faithfulness_prompt(
        context="c", output="o", reason_mode="overall"
    )[0]["content"]
    per_item = render_faithfulness_prompt(
        context="c", output="o", reason_mode="per_item"
    )[0]["content"]
    none = render_faithfulness_prompt(
        context="c", output="o", reason_mode="none"
    )[0]["content"]
    normalized = " ".join(overall.split())
    normalized_per_item = " ".join(per_item.split())
    assert "at least one and at most three representative" in normalized
    assert (
        "Do not include a metric score, percentage, claim counts" in normalized
    )
    assert (
        "Start directly with the substantive supported area or failure"
        in normalized
    )
    assert "Do not begin with generic aggregate commentary" in normalized
    assert "Most claims are supported" in normalized
    assert "claims = []" in normalized
    assert "Do not invent claims merely to avoid an empty array" in normalized
    assert (
        "The output contains no materially checkable factual claims"
        in normalized
    )
    assert "non-empty reason for every claim" in normalized_per_item
    assert (
        "at least one and at most three representative"
        in normalized_per_item
    )
    assert (
        "Start directly with the substantive supported area"
        in normalized_per_item
    )
    assert "claims is empty" in normalized_per_item
    assert "Do not return per-item reasons" in none


def test_faithfulness_schemas_never_include_scores_labels_or_max_items():
    for schema in (
        FAITHFULNESS_SCHEMA_OVERALL,
        FAITHFULNESS_SCHEMA_PER_ITEM,
        FAITHFULNESS_SCHEMA_NONE,
    ):
        serialized = repr(schema)
        assert "score" not in serialized
        assert "label" not in serialized
        assert "maxItems" not in serialized


class AsyncJudge:
    def __init__(self):
        self.async_calls = 0
        self.sync_calls = 0

    async def async_generate_object(self, prompt, schema):
        self.async_calls += 1
        return _response([_claim()])

    def generate_object(self, prompt, schema):
        self.sync_calls += 1
        raise AssertionError("native async path should be used")


def test_native_async_path_calls_once():
    judge = AsyncJudge()
    result = asyncio.run(
        FaithfulnessEvaluator(judge).a_evaluate(
            CASE, judge_limiter=asyncio.Semaphore(1)
        )
    )
    assert result.score == 1.0
    assert judge.async_calls == 1 and judge.sync_calls == 0


def test_framework_async_thread_fallback_calls_once():
    judge = Judge(_response([_claim()]))
    result = asyncio.run(
        EvaluationFramework([FaithfulnessEvaluator(judge)]).a_evaluate(
            CASE, max_concurrency=1
        )
    )["faithfulness"]
    assert result.score == 1.0 and len(judge.calls) == 1


class BlockingJudge:
    def __init__(self, responses):
        self.responses = list(responses)
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def generate_object(self, prompt, schema):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.02)
        with self.lock:
            self.active -= 1
        return self.responses.pop(0)


def test_async_many_respects_shared_semaphore_and_order():
    judge = BlockingJudge([_response([_claim(str(i))]) for i in range(4)])
    framework = EvaluationFramework([FaithfulnessEvaluator(judge)])
    results = asyncio.run(
        framework.a_evaluate_many([CASE] * 4, max_concurrency=2)
    )
    assert len(results) == 4 and judge.max_active <= 2


def test_evaluate_many_and_groups_keep_each_output_one_case():
    judge = Judge(*[_response([_claim(str(i))]) for i in range(4)])
    framework = EvaluationFramework([FaithfulnessEvaluator(judge)])
    assert len(framework.evaluate_many([CASE, CASE])) == 2
    grouped = framework.evaluate_groups([
        {"context": "source", "outputs": [[{"fact": "a"}], {"fact": "b"}]}
    ])
    assert len(grouped) == 2 and len(judge.calls) == 4


@pytest.mark.parametrize(
    "description,status",
    [
        ("Omitting other context facts does not invalidate this claim", "supported"),
        ("A contradiction is unsupported", "unsupported"),
        ("Unsupported specificity is unsupported", "unsupported"),
        ("A changed numeric qualifier is unsupported", "unsupported"),
    ],
)
def test_semantic_outcomes_are_reflected_without_extra_scoring(description, status):
    result = FaithfulnessEvaluator(
            Judge(_response([_claim(description, status)]))
    ).evaluate(CASE)
    assert result.score == (1.0 if status == "supported" else 0.0)
