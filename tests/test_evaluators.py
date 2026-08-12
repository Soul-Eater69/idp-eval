"""Evaluator tests using a fake judge (no real LLM calls)."""

import pytest

from idp_eval import (
    EvaluationCase,
    EvaluationFramework,
    HallucinationEvaluator,
    InputCoverageEvaluator,
)
from idp_eval.models import EvaluationResult, Evaluator


class FakeJudge:
    """A judge stub that returns a canned structured response.

    Mirrors the ``generate_object(prompt, schema) -> dict`` contract that the
    custom evaluators depend on.
    """

    def __init__(self, response: dict):
        self.response = response
        self.calls: list[dict] = []

    def generate_object(self, prompt: str, schema: dict) -> dict:
        self.calls.append({"prompt": prompt, "schema": schema})
        return self.response


CASE = EvaluationCase(
    input="Generate content from the provided source.",
    context="Users can view invoices. Invoices show the total amount due.",
    output="Users can view invoices. Invoices are stored in AWS S3.",
)


def test_hallucination_evaluator():
    judge = FakeJudge(
        {
            "claims": [
                {"claim": "Users can view invoices.", "status": "supported"},
                {"claim": "Invoices are stored in AWS S3.", "status": "unsupported"},
            ]
        }
    )
    result = HallucinationEvaluator(llm=judge).evaluate(CASE)

    assert result.metric == "hallucination"
    assert result.score == 0.5
    assert result.details["unsupported_claims"] == ["Invoices are stored in AWS S3."]
    # Judge was actually invoked with a formatted prompt.
    assert CASE.output in judge.calls[0]["prompt"]


def test_input_coverage_evaluator():
    judge = FakeJudge(
        {
            "items": [
                {"source_item": "Users can view invoices.", "status": "covered"},
                {"source_item": "Invoices show total amount due.", "status": "missing"},
            ]
        }
    )
    result = InputCoverageEvaluator(llm=judge).evaluate(CASE)

    assert result.metric == "input_coverage"
    assert result.score == 0.5
    assert result.details["missing_items"] == ["Invoices show total amount due."]


def test_framework_runs_selected_metrics():
    hallucination = HallucinationEvaluator(
        llm=FakeJudge({"claims": [{"claim": "x", "status": "supported"}]})
    )
    coverage = InputCoverageEvaluator(
        llm=FakeJudge({"items": [{"source_item": "x", "status": "covered"}]})
    )
    framework = EvaluationFramework(evaluators=[hallucination, coverage])

    results = framework.evaluate(CASE, metrics=["input_coverage"])

    assert set(results) == {"input_coverage"}
    assert results["input_coverage"].score == 1.0


def test_framework_runs_all_metrics_by_default():
    hallucination = HallucinationEvaluator(
        llm=FakeJudge({"claims": [{"claim": "x", "status": "supported"}]})
    )
    coverage = InputCoverageEvaluator(
        llm=FakeJudge({"items": [{"source_item": "x", "status": "covered"}]})
    )
    framework = EvaluationFramework(evaluators=[hallucination, coverage])

    results = framework.evaluate(CASE)

    assert set(results) == {"hallucination", "input_coverage"}


def test_framework_rejects_duplicate_names():
    a = InputCoverageEvaluator(llm=FakeJudge({"items": []}))
    b = InputCoverageEvaluator(llm=FakeJudge({"items": []}))
    with pytest.raises(ValueError):
        EvaluationFramework(evaluators=[a, b])


def test_framework_rejects_unknown_metric():
    framework = EvaluationFramework(
        evaluators=[InputCoverageEvaluator(llm=FakeJudge({"items": []}))]
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
