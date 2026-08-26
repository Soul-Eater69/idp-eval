"""Offline contract tests for few-shot business/content leakage."""

from __future__ import annotations

import asyncio
import inspect
from contextlib import contextmanager

import pytest

from idp_eval import (
    EvaluationCase,
    EvaluationFramework,
    FewShotContentLeakageEvaluator,
)
from idp_eval.prompts.few_shot_content_leakage import (
    FEW_SHOT_CONTENT_LEAKAGE_SCHEMA_NONE,
    FEW_SHOT_CONTENT_LEAKAGE_SCHEMA_OVERALL,
    FEW_SHOT_CONTENT_LEAKAGE_SCHEMA_PER_ITEM,
    render_few_shot_content_leakage_prompt,
)
from idp_eval.scoring import (
    calculate_few_shot_content_leakage,
    classify_few_shot_source,
    few_shot_content_leakage_label,
    few_shot_item_leakage_score,
)

openpyxl = pytest.importorskip("openpyxl")


CASE = EvaluationCase(
    context="Users can reset passwords through email.",
    retrieved_documents=["Administrators must authenticate using MFA."],
    output="Users can reset passwords by email. Administrators must use MFA.",
)


class Judge:
    model = "offline-leakage-judge"

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def generate_object(self, prompt, schema):
        self.calls.append({"prompt": prompt, "schema": schema})
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _claim(
    claim="Users can reset passwords through email.",
    theme_supported=True,
    example_supported=False,
    reason=None,
):
    value = {
        "claim": claim,
        "theme_supported": theme_supported,
        "example_supported": example_supported,
    }
    if reason is not None:
        value["reason"] = reason
    return value


def _response(
    claims, *, reason_mode="overall", overall_reason="Semantic summary."
):
    prepared = []
    for claim in claims:
        value = dict(claim)
        if reason_mode == "overall":
            value.setdefault(
                "reason",
                ""
                if value["theme_supported"]
                else "The current context does not support this claim.",
            )
        elif reason_mode == "per_item":
            value.setdefault("reason", "The supplied sources determine support.")
        prepared.append(value)
    response = {"claims": prepared}
    if reason_mode != "none":
        response["overall_reason"] = overall_reason
    return response


def _sheet(path, name):
    return list(openpyxl.load_workbook(path)[name].iter_rows(values_only=True))


def test_public_constructor_matches_core_evaluator_conventions():
    assert tuple(inspect.signature(FewShotContentLeakageEvaluator).parameters) == (
        "llm",
        "verbose",
        "max_items",
        "reason_mode",
    )


@pytest.mark.parametrize(
    "theme_supported,example_supported,classification,score,label",
    [
        (True, False, "theme_only", 0.0, "no_leakage"),
        (True, True, "theme_and_examples", 0.0, "no_leakage"),
        (False, True, "example_only", 1.0, "leakage_detected"),
        (False, False, "unsupported", 0.0, "no_leakage"),
    ],
)
def test_all_source_combinations_are_python_derived(
    theme_supported, example_supported, classification, score, label
):
    reason = "" if theme_supported else "Diagnostic source distinction."
    judge = Judge(
        _response(
            [
                _claim(
                    theme_supported=theme_supported,
                    example_supported=example_supported,
                    reason=reason,
                )
            ]
        )
    )
    result = FewShotContentLeakageEvaluator(judge, verbose=True).evaluate(CASE)
    item = result.details["claims"][0]

    assert item["classification"] == classification
    assert item["item_leakage_score"] == score
    assert result.score == score and result.label == label
    assert len(judge.calls) == 1


def test_mixed_claims_use_exact_example_only_fraction_and_counts():
    claims = [
        _claim("Current only.", True, False),
        _claim("Both sources.", True, True),
        _claim("Example only.", False, True),
        _claim("Neither source.", False, False),
    ]
    result = FewShotContentLeakageEvaluator(
        Judge(_response(claims)), verbose=True
    ).evaluate(CASE)

    assert result.score == 0.25
    assert result.label == "leakage_detected"
    assert result.details["theme_only_count"] == 1
    assert result.details["theme_and_examples_count"] == 1
    assert result.details["example_only_count"] == 1
    assert result.details["unsupported_count"] == 1
    assert [item["id"] for item in result.details["claims"]] == [
        "FS1",
        "FS2",
        "FS3",
        "FS4",
    ]


@pytest.mark.parametrize("verbose", [False, True])
def test_empty_claims_are_not_applicable_after_one_call(verbose):
    judge = Judge(
        _response(
            [],
            overall_reason="The output contains no materially checkable content claims.",
        )
    )
    result = FewShotContentLeakageEvaluator(judge, verbose=verbose).evaluate(CASE)

    assert result.score is None and result.label == "not_applicable"
    assert result.details["claim_count"] == 0
    assert result.details["judge_call_count"] == 1
    assert ("claims" in result.details) is verbose
    assert len(judge.calls) == 1


def test_verbose_controls_claim_exposure_only():
    response = _response([_claim()])
    compact = FewShotContentLeakageEvaluator(Judge(response)).evaluate(CASE)
    verbose = FewShotContentLeakageEvaluator(
        Judge(response), verbose=True
    ).evaluate(CASE)

    assert "claims" not in compact.details
    assert verbose.details["claims"][0]["classification"] == "theme_only"
    assert compact.explanation == verbose.explanation == "Semantic summary."


def test_overall_reason_mode_requires_failure_reasons_and_empty_passing_reasons():
    passing_reason = _response([_claim(reason="Current context says so.")])
    with pytest.raises(ValueError, match="must use an empty"):
        FewShotContentLeakageEvaluator(Judge(passing_reason)).evaluate(CASE)

    failing_reason = _response(
        [_claim(theme_supported=False, example_supported=True, reason="")]
    )
    with pytest.raises(ValueError, match="non-empty"):
        FewShotContentLeakageEvaluator(Judge(failing_reason)).evaluate(CASE)


def test_per_item_and_none_reason_modes_are_distinct():
    per_item = FewShotContentLeakageEvaluator(
        Judge(
            _response(
                [_claim(reason="Current context directly supports the claim.")],
                reason_mode="per_item",
            )
        ),
        verbose=True,
        reason_mode="per_item",
    ).evaluate(CASE)
    assert per_item.details["claims"][0]["reason"]
    assert per_item.explanation == "Semantic summary."

    none = FewShotContentLeakageEvaluator(
        Judge(_response([_claim()], reason_mode="none")),
        verbose=True,
        reason_mode="none",
    ).evaluate(CASE)
    assert "reason" not in none.details["claims"][0]
    assert none.explanation is None


def test_per_item_mode_requires_every_reason():
    response = _response([_claim(reason="")], reason_mode="per_item")
    with pytest.raises(ValueError, match="every claim"):
        FewShotContentLeakageEvaluator(
            Judge(response), reason_mode="per_item"
        ).evaluate(CASE)


@pytest.mark.parametrize("max_items", [None, 1, 5])
def test_item_limit_is_prompt_and_python_enforced_without_schema_max_items(
    max_items,
):
    judge = Judge(_response([_claim()]))
    result = FewShotContentLeakageEvaluator(
        judge, max_items=max_items
    ).evaluate(CASE)
    system = judge.calls[0]["prompt"][0]["content"]

    if max_items is None:
        assert "all materially distinct" in system
    else:
        assert f"select at most {max_items}" in system
        assert "Examine the complete OUTPUT before selecting" in system
    assert "maxItems" not in judge.calls[0]["schema"]["properties"]["claims"]
    assert result.details["max_items"] == max_items


def test_unlimited_response_is_exhaustive_and_bounded_response_is_validated():
    many = [_claim(f"Claim {index}.") for index in range(100)]
    result = FewShotContentLeakageEvaluator(Judge(_response(many))).evaluate(CASE)
    assert result.details["evaluated_claims"] == 100

    with pytest.raises(ValueError, match="exceeds configured max_items=1"):
        FewShotContentLeakageEvaluator(
            Judge(_response([_claim("A"), _claim("B")])), max_items=1
        ).evaluate(CASE)


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "5"])
def test_invalid_max_items_is_rejected_before_judge_work(bad):
    judge = Judge(_response([]))
    with pytest.raises(ValueError, match="max_items"):
        FewShotContentLeakageEvaluator(judge, max_items=bad)
    assert judge.calls == []


def test_invalid_reason_mode_is_rejected_before_judge_work():
    judge = Judge(_response([]))
    with pytest.raises(ValueError, match="reason_mode"):
        FewShotContentLeakageEvaluator(judge, reason_mode="invalid")
    assert judge.calls == []


@pytest.mark.parametrize(
    "response,reason_mode,match",
    [
        ([], "overall", "expected an object"),
        ({}, "overall", "expected exactly"),
        ({"claims": [], "extra": 1}, "none", "expected exactly"),
        ({"claims": "bad"}, "none", "must be a list"),
        ({"claims": ["bad"]}, "none", "expected an object"),
        ({"claims": [{"claim": "x"}]}, "none", "unexpected or missing"),
        (
            {"claims": [_claim(reason="extra")]},
            "none",
            "unexpected or missing",
        ),
        ({"claims": [_claim(" ")]}, "none", "non-empty string"),
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
                "claims": [_claim()],
                "overall_reason": " ",
            },
            "overall",
            "overall_reason",
        ),
    ],
)
def test_strict_response_validation(response, reason_mode, match):
    with pytest.raises(ValueError, match=match):
        FewShotContentLeakageEvaluator(
            Judge(response), reason_mode=reason_mode
        ).evaluate(CASE)


@pytest.mark.parametrize(
    "field,value",
    [("theme_supported", 1), ("example_supported", "true")],
)
def test_support_fields_must_be_actual_booleans(field, value):
    claim = _claim()
    claim[field] = value
    with pytest.raises(ValueError, match=f"`{field}` must be a boolean"):
        FewShotContentLeakageEvaluator(
            Judge(_response([claim], reason_mode="none")), reason_mode="none"
        ).evaluate(CASE)


def test_duplicate_same_support_is_deduped_with_first_stable_occurrence():
    result = FewShotContentLeakageEvaluator(
        Judge(
            _response(
                [
                    _claim(" Administrator   MFA is required. ", False, True),
                    _claim("ADMINISTRATOR MFA IS REQUIRED.", False, True),
                ]
            )
        ),
        verbose=True,
    ).evaluate(CASE)

    assert result.details["claim_count"] == 1
    assert result.details["claims"][0]["id"] == "FS1"
    assert result.details["claims"][0]["claim"] == (
        "Administrator MFA is required."
    )


def test_duplicate_conflicting_support_booleans_raise():
    response = _response(
        [
            _claim("Administrator MFA is required.", False, True),
            _claim(" administrator  mfa is required. ", True, True),
        ]
    )
    with pytest.raises(ValueError, match="conflicting support booleans"):
        FewShotContentLeakageEvaluator(Judge(response)).evaluate(CASE)


@pytest.mark.parametrize("missing", ["context", "retrieved_documents", "output"])
def test_required_fields_fail_before_judge_work(missing):
    values = {
        "context": "current source",
        "retrieved_documents": ["historical example"],
        "output": "generation",
    }
    values[missing] = None
    judge = Judge(_response([]))
    with pytest.raises(ValueError, match=f"requires non-empty `{missing}`"):
        FewShotContentLeakageEvaluator(judge).evaluate(EvaluationCase(**values))
    assert judge.calls == []


def test_prompt_renders_only_three_semantic_fields_with_explicit_source_roles():
    case = EvaluationCase(
        input="IGNORE-INPUT",
        instructions="IGNORE-INSTRUCTIONS",
        metadata={"secret": "IGNORE-METADATA"},
        context={"current_policy": "Email reset is required."},
        retrieved_documents=[{"historical_policy": "MFA is required."}],
        output={"requirement": "MFA is required."},
    )
    judge = Judge(_response([_claim()]))
    FewShotContentLeakageEvaluator(judge).evaluate(case)
    system = judge.calls[0]["prompt"][0]["content"]
    normalized_system = " ".join(system.split())
    user = judge.calls[0]["prompt"][1]["content"]

    assert "CURRENT CONTEXT is the only authoritative evidence" in system
    assert "HISTORICAL FEW-SHOT EXAMPLES are non-authoritative" in system
    assert (
        "likely content leakage or content overlap, not strict causal proof"
        in normalized_system
    )
    assert "[CURRENT CONTEXT — AUTHORITATIVE EVIDENCE FOR THIS OUTPUT]" in user
    assert "[HISTORICAL FEW-SHOT EXAMPLES — NON-AUTHORITATIVE]" in user
    assert "Current Policy: Email reset is required." in user
    assert "Historical Policy: MFA is required." in user
    assert "Requirement: MFA is required." in user
    assert all(
        marker not in user
        for marker in ("IGNORE-INPUT", "IGNORE-INSTRUCTIONS", "IGNORE-METADATA")
    )


def test_schemas_are_strict_and_exclude_llm_owned_derived_values():
    for schema, expected in (
        (
            FEW_SHOT_CONTENT_LEAKAGE_SCHEMA_NONE,
            {"claim", "theme_supported", "example_supported"},
        ),
        (
            FEW_SHOT_CONTENT_LEAKAGE_SCHEMA_OVERALL,
            {"claim", "theme_supported", "example_supported", "reason"},
        ),
        (
            FEW_SHOT_CONTENT_LEAKAGE_SCHEMA_PER_ITEM,
            {"claim", "theme_supported", "example_supported", "reason"},
        ),
    ):
        assert schema["additionalProperties"] is False
        assert "maxItems" not in schema["properties"]["claims"]
        item = schema["properties"]["claims"]["items"]
        assert item["additionalProperties"] is False
        assert set(item["required"]) == expected
        assert set(item["properties"]) == expected
        serialized = repr(schema)
        for forbidden in ("classification", "score", "label", "confidence"):
            assert forbidden not in serialized


def test_scoring_helpers_are_deterministic_and_reject_bad_inputs():
    assert classify_few_shot_source(True, False) == "theme_only"
    assert classify_few_shot_source(True, True) == "theme_and_examples"
    assert classify_few_shot_source(False, True) == "example_only"
    assert classify_few_shot_source(False, False) == "unsupported"
    assert few_shot_item_leakage_score("example_only") == 1.0
    assert few_shot_item_leakage_score("unsupported") == 0.0
    assert calculate_few_shot_content_leakage(
        [{"classification": "example_only"}, {"classification": "theme_only"}]
    ) == 0.5
    assert few_shot_content_leakage_label(0.0) == "no_leakage"
    assert few_shot_content_leakage_label(0.1) == "leakage_detected"
    with pytest.raises(ValueError, match="must be booleans"):
        classify_few_shot_source(1, False)
    with pytest.raises(ValueError, match="Unknown few-shot"):
        few_shot_item_leakage_score("other")
    with pytest.raises(ValueError, match="At least one claim"):
        calculate_few_shot_content_leakage([])


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


def test_native_async_evaluator_path_uses_one_call():
    judge = AsyncJudge()
    result = asyncio.run(
        FewShotContentLeakageEvaluator(judge).a_evaluate(
            CASE, judge_limiter=asyncio.Semaphore(1)
        )
    )
    assert result.score == 0.0
    assert judge.async_calls == 1 and judge.sync_calls == 0


def test_evaluator_runs_through_framework_with_shared_judge():
    judge = Judge(
        _response(
            [_claim(theme_supported=False, example_supported=False)]
        )
    )
    result = EvaluationFramework(
        [FewShotContentLeakageEvaluator], judge=judge
    ).evaluate(CASE)["few_shot_content_leakage"]
    assert result.metric == "few_shot_content_leakage"
    assert len(judge.calls) == 1


def test_operational_error_handling_is_inherited_from_framework():
    judge = Judge(TimeoutError("provider unavailable"))
    result = EvaluationFramework(
        [FewShotContentLeakageEvaluator(judge)]
    ).evaluate(CASE, on_error="continue")["few_shot_content_leakage"]
    assert result.label == "error"
    assert result.details["retryable"] is True
    assert len(judge.calls) == 1


def test_resume_signature_captures_configuration_and_safe_judge_identity():
    base = FewShotContentLeakageEvaluator(Judge()).resume_signature()
    assert base["contract_version"] == 1
    assert base["judge"]["model"] == "offline-leakage-judge"
    signatures = {
        repr(FewShotContentLeakageEvaluator(Judge()).resume_signature()),
        repr(
            FewShotContentLeakageEvaluator(
                Judge(), verbose=True
            ).resume_signature()
        ),
        repr(
            FewShotContentLeakageEvaluator(
                Judge(), max_items=2
            ).resume_signature()
        ),
        repr(
            FewShotContentLeakageEvaluator(
                Judge(), reason_mode="none"
            ).resume_signature()
        ),
    }
    assert len(signatures) == 4
    assert "endpoint" not in repr(base).lower()
    assert "api_key" not in repr(base).lower()


def test_trace_uses_stable_single_stage_name(monkeypatch):
    spans = []

    @contextmanager
    def judge_span(name, attributes):
        spans.append((name, attributes))
        yield

    monkeypatch.setattr(
        "idp_eval.evaluators.few_shot_content_leakage.tracing.judge_span",
        judge_span,
    )
    FewShotContentLeakageEvaluator(Judge(_response([_claim()]))).evaluate(CASE)
    assert spans == [
        (
            "few_shot_content_leakage.evaluate",
            {
                "idp_eval.metric": "few_shot_content_leakage",
                "idp_eval.stage": "evaluate",
            },
        )
    ]


def test_verbose_excel_output_has_claim_audit_sheet(tmp_path):
    path = tmp_path / "leakage.xlsx"
    evaluator = FewShotContentLeakageEvaluator(
        Judge(_response([_claim("Administrator MFA.", False, True)])),
        verbose=True,
    )
    EvaluationFramework(
        [evaluator], output="excel", excel_path=str(path)
    ).evaluate(EvaluationCase(**{**CASE.__dict__, "case_id": "case-1"}))

    header, row = _sheet(path, "few_shot_content_leakage_items")
    values = dict(zip(header, row))
    assert values["claim_id"] == "FS1"
    assert values["classification"] == "example_only"
    assert values["item_leakage_score"] == 1.0
