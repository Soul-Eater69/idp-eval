"""Offline contract tests for one-call claim-level faithfulness."""

import asyncio
import threading
import time

import pytest

from idp_eval import EvaluationCase, EvaluationFramework, FaithfulnessEvaluator
from idp_eval.prompts.faithfulness import (
    FAITHFULNESS_SCHEMA_COMPACT,
    FAITHFULNESS_SCHEMA_VERBOSE,
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


def test_one_sync_call_and_supported_scoring():
    judge = Judge({"claims": [_claim()]})
    result = FaithfulnessEvaluator(judge).evaluate(CASE)
    assert len(judge.calls) == 1
    assert result.score == 1.0 and result.label == "faithful"
    assert result.details["judge_call_count"] == 1
    assert result.details["claim_count"] == 1
    assert "claims" not in result.details


def test_verbose_mixed_score_ids_audit_and_explanation():
    judge = Judge(
        {"claims": [
            _claim("Cancellation is allowed.", "supported", ""),
            _claim("Refunds are instant.", "unsupported", "Context says five days."),
        ]}
    )
    result = FaithfulnessEvaluator(judge, verbose=True).evaluate(CASE)
    assert result.score == 0.5 and result.label == "unfaithful"
    assert result.explanation == (
        "1 of 2 factual claims were supported; 1 was unsupported."
    )
    assert [item["id"] for item in result.details["claims"]] == ["F1", "F2"]
    assert [item["item_score"] for item in result.details["claims"]] == [1.0, 0.0]


def test_all_unsupported_scores_zero():
    result = FaithfulnessEvaluator(
        Judge({"claims": [_claim(status="unsupported")]}),
    ).evaluate(CASE)
    assert result.score == 0.0 and result.label == "unfaithful"


@pytest.mark.parametrize("verbose", [False, True])
def test_no_claims_is_not_applicable_after_one_call(verbose):
    judge = Judge({"claims": []})
    result = FaithfulnessEvaluator(judge, verbose=verbose).evaluate(CASE)
    assert result.score is None and result.label == "not_applicable"
    assert result.explanation == (
        "No checkable factual claims were identified in the output."
    )
    assert result.details["judge_call_count"] == 1
    assert result.details["claim_count"] == 0
    assert len(judge.calls) == 1


def test_exact_normalized_dedup_keeps_first_and_stable_ids():
    judge = Judge({"claims": [
        _claim(" Refunds   take five days. ", "supported", ""),
        _claim("REFUNDS TAKE FIVE DAYS.", "unsupported", "duplicate"),
        _claim("Cancellation takes 24 hours.", "supported", ""),
    ]})
    result = FaithfulnessEvaluator(judge, verbose=True).evaluate(CASE)
    assert result.details["claim_count"] == 2
    assert [c["id"] for c in result.details["claims"]] == ["F1", "F2"]
    assert result.details["claims"][0]["status"] == "supported"


@pytest.mark.parametrize("missing", ["context", "output"])
def test_required_fields_fail_before_judge(missing):
    judge = Judge({"claims": []})
    values = {"context": "source", "output": "answer"}
    values[missing] = None
    with pytest.raises(ValueError, match=f"requires non-empty `{missing}`"):
        FaithfulnessEvaluator(judge).evaluate(EvaluationCase(**values))
    assert judge.calls == []


def test_only_rendered_context_and_output_reach_prompt():
    judge = Judge({"claims": [_claim()]})
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
    "response,verbose,match",
    [
        ([], False, "expected an object"),
        ({}, False, "expected only"),
        ({"claims": [], "extra": 1}, False, "expected only"),
        ({"claims": "bad"}, False, "must be a list"),
        ({"claims": ["bad"]}, False, "expected an object"),
        ({"claims": [{"claim": "x"}]}, False, "expected exactly"),
        ({"claims": [_claim(reason="extra")]}, False, "expected exactly"),
        ({"claims": [_claim(" ")]}, False, "non-empty string"),
        ({"claims": [_claim(status="partial")]}, False, "Unknown faithfulness"),
        ({"claims": [_claim(reason=3)]}, True, "must be a string"),
        ({"claims": [_claim(status="unsupported", reason=" ")]}, True, "non-empty"),
    ],
)
def test_strict_response_validation(response, verbose, match):
    with pytest.raises(ValueError, match=match):
        FaithfulnessEvaluator(Judge(response), verbose=verbose).evaluate(CASE)


def test_compact_and_verbose_schemas_are_strict():
    for schema, keys in (
        (FAITHFULNESS_SCHEMA_COMPACT, {"claim", "status"}),
        (FAITHFULNESS_SCHEMA_VERBOSE, {"claim", "status", "reason"}),
    ):
        assert schema["additionalProperties"] is False
        assert schema["required"] == ["claims"]
        item = schema["properties"]["claims"]["items"]
        assert item["additionalProperties"] is False
        assert set(item["required"]) == keys
        assert set(item["properties"]) == keys


def test_prompt_contract_excludes_scores_and_distinguishes_coverage():
    system = render_faithfulness_prompt(
        context="c", output="o", verbose=False
    )[0]["content"]
    assert "OUTPUT -> CONTEXT" in system
    assert '"supported" or "unsupported"' in system
    assert "not a faithfulness failure; omissions belong to Coverage" in system
    assert "Do not return IDs, item scores, aggregate scores" in system


class AsyncJudge:
    def __init__(self):
        self.async_calls = 0
        self.sync_calls = 0

    async def async_generate_object(self, prompt, schema):
        self.async_calls += 1
        return {"claims": [_claim()]}

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
    judge = Judge({"claims": [_claim()]})
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
    judge = BlockingJudge([{"claims": [_claim(str(i))]} for i in range(4)])
    framework = EvaluationFramework([FaithfulnessEvaluator(judge)])
    results = asyncio.run(
        framework.a_evaluate_many([CASE] * 4, max_concurrency=2)
    )
    assert len(results) == 4 and judge.max_active <= 2


def test_evaluate_many_and_groups_keep_each_output_one_case():
    judge = Judge(*[{"claims": [_claim(str(i))]} for i in range(4)])
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
        Judge({"claims": [_claim(description, status)]})
    ).evaluate(CASE)
    assert result.score == (1.0 if status == "supported" else 0.0)
