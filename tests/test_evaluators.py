"""Evaluator and framework tests using fakes (no real LLM calls)."""

import asyncio

import pytest

from idp_eval import (
    CoverageEvaluator,
    EvaluationCase,
    EvaluationFramework,
    FaithfulnessEvaluator,
    InstructionAdherenceEvaluator,
)
from idp_eval.models import EvaluationResult, Evaluator


class FakeJudge:
    """A judge stub that returns a canned structured response.

    Mirrors the ``generate_object(prompt, schema) -> dict`` contract that the
    instruction-adherence evaluator depends on.
    """

    def __init__(self, response: dict):
        self.response = response
        self.calls: list[dict] = []

    def generate_object(self, prompt: str, schema: dict) -> dict:
        self.calls.append({"prompt": prompt, "schema": schema})
        return self.response


class ScriptedJudge:
    """Returns queued structured responses in order."""

    def __init__(self, *responses: dict):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def generate_object(self, prompt, schema: dict) -> dict:
        self.calls.append({"prompt": prompt, "schema": schema})
        return self._responses.pop(0)


CASE = EvaluationCase(
    input="Summarize the invoice features from the provided source.",
    instructions="Keep it concise.",
    context="Users can view invoices. Invoices show the total amount due.",
    output="Users can view invoices. Invoices are stored in AWS S3.",
)


def _coverage_judge():
    """One-call coverage judge: 1 covered + 1 missing -> score 0.5."""
    return ScriptedJudge(
        {
            "items": [
                {"source_item": "Users can view invoices.",
                 "meaningfully_present": True, "fully_present": True},
                {"source_item": "Invoices show total amount due.",
                 "meaningfully_present": False, "fully_present": False},
            ]
        }
    )


def _faithfulness_judge():
    return ScriptedJudge(
        {"claims": [{"claim": "Users can view invoices.", "status": "supported"}]}
    )


def test_coverage_evaluator():
    judge = _coverage_judge()
    result = CoverageEvaluator(llm=judge).evaluate(CASE)

    assert result.metric == "coverage"
    assert result.score == 0.5
    assert result.details["final_item_count"] == 2
    assert result.details["missing_count"] == 1
    assert len(judge.calls) == 1
    user = judge.calls[0]["prompt"][1]["content"]
    assert CASE.context in user and CASE.output in user
    assert CASE.input not in user


def test_coverage_partial_and_missing():
    judge = ScriptedJudge(
        {"items": [
            {"source_item": "a", "meaningfully_present": True,
             "fully_present": True, "reason": ""},
            {"source_item": "b", "meaningfully_present": True,
             "fully_present": False, "reason": "Qualifier missing."},
            {"source_item": "c", "meaningfully_present": False,
             "fully_present": False, "reason": "Missing."},
        ]}
    )
    result = CoverageEvaluator(llm=judge, verbose=True).evaluate(CASE)

    assert result.score == (1.0 + 0.5 + 0.0) / 3
    assert result.details["missing_count"] == 1
    assert result.details["items"][1]["item_score"] == 0.5


def test_faithfulness_uses_only_context_and_output_semantically():
    case = EvaluationCase(
        input="INPUT MUST BE NEUTRALIZED",
        instructions="INSTRUCTIONS MUST BE IGNORED",
        context={"policy": ["Use approved regions", "Encrypt records"]},
        output={"summary": "Use approved regions and encrypt records."},
        metadata={"marker": "METADATA MUST BE IGNORED"},
        retrieved_documents=["RETRIEVAL MUST BE IGNORED"],
    )
    judge = _faithfulness_judge()
    evaluator = FaithfulnessEvaluator(llm=judge)
    result = evaluator.evaluate(case)

    assert result.metric == "faithfulness"
    assert result.score == 1.0
    assert result.label == "faithful"
    assert result.explanation == (
        "1 of 1 factual claims were supported; 0 were unsupported."
    )
    assert result.details["judge_call_count"] == 1
    assert len(judge.calls) == 1
    user = judge.calls[0]["prompt"][1]["content"]
    assert "Policy:\n- Use approved regions\n- Encrypt records" in user
    assert "Summary: Use approved regions and encrypt records." in user
    for ignored in (
        "INPUT MUST BE NEUTRALIZED",
        "INSTRUCTIONS MUST BE IGNORED",
        "METADATA MUST BE IGNORED",
        "RETRIEVAL MUST BE IGNORED",
    ):
        assert ignored not in user


@pytest.mark.parametrize("missing", ["context", "output"])
def test_faithfulness_missing_required_field_fails_before_judge_call(missing):
    judge = _faithfulness_judge()
    evaluator = FaithfulnessEvaluator(llm=judge)
    fields = {"context": "source", "output": "answer"}
    fields[missing] = None
    with pytest.raises(ValueError, match=f"requires non-empty `{missing}`"):
        evaluator.evaluate(EvaluationCase(**fields))
    assert judge.calls == []


def test_faithfulness_async_uses_framework_fallback_once():
    judge = _faithfulness_judge()
    evaluator = FaithfulnessEvaluator(llm=judge)
    result = asyncio.run(
        EvaluationFramework(evaluators=[evaluator]).a_evaluate(
            EvaluationCase(context="source", output="answer"),
            max_concurrency=1,
        )
    )["faithfulness"]
    assert result.score == 1.0
    assert len(judge.calls) == 1


def test_framework_runs_all_metrics_by_default():
    framework = EvaluationFramework(
        evaluators=[
            FaithfulnessEvaluator(llm=_faithfulness_judge()),
            CoverageEvaluator(llm=_coverage_judge()),
        ]
    )
    results = framework.evaluate(CASE)

    assert set(results) == {"faithfulness", "coverage"}


def test_framework_runs_all_three_metrics():
    instruction_judge = ScriptedJudge(
        {
            "instructions": [
                {"instruction": "Keep it concise.", "status": "followed"}
            ]
        },
    )
    framework = EvaluationFramework(
        evaluators=[
            FaithfulnessEvaluator(llm=_faithfulness_judge()),
            CoverageEvaluator(llm=_coverage_judge()),
            InstructionAdherenceEvaluator(llm=instruction_judge),
        ]
    )
    results = framework.evaluate(CASE)

    assert set(results) == {"faithfulness", "coverage", "instruction_adherence"}


def test_framework_runs_selected_metrics():
    framework = EvaluationFramework(
        evaluators=[
            FaithfulnessEvaluator(llm=_faithfulness_judge()),
            CoverageEvaluator(llm=_coverage_judge()),
        ]
    )
    results = framework.evaluate(CASE, metrics=["coverage"])

    assert set(results) == {"coverage"}
    assert results["coverage"].score == 0.5


def test_framework_rejects_duplicate_names():
    a = CoverageEvaluator(llm=FakeJudge({"items": []}))
    b = CoverageEvaluator(llm=FakeJudge({"items": []}))
    with pytest.raises(ValueError):
        EvaluationFramework(evaluators=[a, b])


def test_framework_rejects_unknown_metric():
    framework = EvaluationFramework(
        evaluators=[CoverageEvaluator(llm=FakeJudge({"items": []}))]
    )
    with pytest.raises(KeyError):
        framework.evaluate(CASE, metrics=["nope"])


def test_custom_evaluator_needs_no_framework_change():
    """A new metric only implements the Evaluator interface."""

    class ConstantEvaluator(Evaluator):
        name = "constant"

        def evaluate(self, case: EvaluationCase) -> EvaluationResult:
            return EvaluationResult(
                metric=self.name, score=1.0, label="high", explanation="ok"
            )

    framework = EvaluationFramework(evaluators=[ConstantEvaluator()])
    results = framework.evaluate(CASE)
    assert results["constant"].score == 1.0


# --- class-based construction (preferred API) -------------------------------


def test_framework_from_classes_shares_judge():
    judge = _coverage_judge()
    framework = EvaluationFramework(
        evaluators=[CoverageEvaluator, InstructionAdherenceEvaluator],
        judge=judge,
    )
    assert set(framework.metrics) == {"coverage", "instruction_adherence"}
    # The same judge instance was injected into each constructed evaluator.
    assert framework._evaluators["coverage"]._llm is judge
    assert framework._evaluators["instruction_adherence"]._llm is judge


def test_framework_classes_select_two_only():
    framework = EvaluationFramework(
        evaluators=[FaithfulnessEvaluator, CoverageEvaluator],
        judge=object(),
    )
    assert set(framework.metrics) == {"faithfulness", "coverage"}


def test_framework_all_three_classes():
    framework = EvaluationFramework(
        evaluators=[
            FaithfulnessEvaluator,
            CoverageEvaluator,
            InstructionAdherenceEvaluator,
        ],
        judge=object(),
    )
    assert set(framework.metrics) == {
        "faithfulness",
        "coverage",
        "instruction_adherence",
    }


def test_framework_duplicate_classes_fail():
    with pytest.raises(ValueError):
        EvaluationFramework(
            evaluators=[CoverageEvaluator, CoverageEvaluator], judge=object()
        )


def test_framework_class_without_judge_raises():
    with pytest.raises(ValueError):
        EvaluationFramework(evaluators=[CoverageEvaluator])


def test_framework_rejects_invalid_entry():
    with pytest.raises(TypeError):
        EvaluationFramework(evaluators=[123], judge=object())


def test_framework_custom_evaluator_class_with_judge():
    class MyEvaluator(Evaluator):
        name = "my_metric"

        def __init__(self, llm):
            self._llm = llm

        def evaluate(self, case: EvaluationCase) -> EvaluationResult:
            return EvaluationResult(
                metric=self.name, score=0.5, label="medium", explanation="ok"
            )

    judge = object()
    framework = EvaluationFramework(evaluators=[MyEvaluator], judge=judge)
    assert framework._evaluators["my_metric"]._llm is judge
    assert framework.evaluate(CASE)["my_metric"].score == 0.5
