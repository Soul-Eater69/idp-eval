"""Evaluator and framework tests using fakes (no real LLM calls)."""

import sys
import types

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
    custom coverage evaluator depends on.
    """

    def __init__(self, response: dict):
        self.response = response
        self.calls: list[dict] = []

    def generate_object(self, prompt: str, schema: dict) -> dict:
        self.calls.append({"prompt": prompt, "schema": schema})
        return self.response


class _FakePhoenixResult:
    """Mimics one entry of a Phoenix evaluator result."""

    def __init__(self, score, label, explanation):
        self.score = score
        self.label = label
        self.explanation = explanation


@pytest.fixture(autouse=True)
def fake_phoenix(monkeypatch):
    """Installs a fake ``phoenix.evals.metrics`` so FaithfulnessEvaluator works.

    Lets us exercise ``FaithfulnessEvaluator`` without installing Phoenix or making
    real LLM calls.
    """

    class FakeFaithfulnessEvaluator:
        def __init__(self, llm):
            self.llm = llm

        def evaluate(self, record):
            self.record = record
            return [_FakePhoenixResult(1.0, "faithful", "Grounded in context.")]

    phoenix_mod = types.ModuleType("phoenix")
    evals_mod = types.ModuleType("phoenix.evals")
    metrics_mod = types.ModuleType("phoenix.evals.metrics")
    metrics_mod.FaithfulnessEvaluator = FakeFaithfulnessEvaluator

    monkeypatch.setitem(sys.modules, "phoenix", phoenix_mod)
    monkeypatch.setitem(sys.modules, "phoenix.evals", evals_mod)
    monkeypatch.setitem(sys.modules, "phoenix.evals.metrics", metrics_mod)


CASE = EvaluationCase(
    input="Summarize the invoice features from the provided source.",
    instructions="Keep it concise.",
    context="Users can view invoices. Invoices show the total amount due.",
    output="Users can view invoices. Invoices are stored in AWS S3.",
)


def _coverage_judge():
    return FakeJudge(
        {
            "requirements": [
                {
                    "requirement": "Users can view invoices.",
                    "status": "covered",
                    "reason": "Output states invoices are viewable.",
                },
                {
                    "requirement": "Invoices show total amount due.",
                    "status": "missing",
                    "reason": "Output omits the total amount due.",
                },
            ]
        }
    )


def test_coverage_evaluator():
    judge = _coverage_judge()
    result = CoverageEvaluator(llm=judge).evaluate(CASE)

    assert result.metric == "coverage"
    assert result.score == 0.5
    assert result.details["total_requirements"] == 2
    assert result.details["missing_count"] == 1
    # Judge was invoked with a rendered message-list prompt.
    prompt = judge.calls[0]["prompt"]
    assert isinstance(prompt, list)
    roles = [message["role"] for message in prompt]
    assert roles == ["system", "user"]
    user_content = prompt[1]["content"]
    assert CASE.output in user_content
    assert CASE.input in user_content


def test_coverage_partial_and_missing():
    judge = FakeJudge(
        {
            "requirements": [
                {"requirement": "a", "status": "covered", "reason": "r"},
                {"requirement": "b", "status": "partial", "reason": "r"},
                {"requirement": "c", "status": "missing", "reason": "r"},
            ]
        }
    )
    result = CoverageEvaluator(llm=judge).evaluate(CASE)

    assert result.score == (1.0 + 0.5 + 0.0) / 3
    assert result.details["missing_count"] == 1
    assert result.details["items"][1]["score"] == 0.5


def test_faithfulness_evaluator():
    result = FaithfulnessEvaluator(llm=object()).evaluate(CASE)

    assert result.metric == "faithfulness"
    assert result.score == 1.0
    assert result.label == "faithful"
    assert result.explanation == "Grounded in context."


def test_framework_runs_all_metrics_by_default():
    framework = EvaluationFramework(
        evaluators=[
            FaithfulnessEvaluator(llm=object()),
            CoverageEvaluator(llm=_coverage_judge()),
        ]
    )
    results = framework.evaluate(CASE)

    assert set(results) == {"faithfulness", "coverage"}


def test_framework_runs_all_three_metrics():
    instruction_judge = FakeJudge(
        {"instructions": [{"instruction": "x", "status": "followed"}]}
    )
    framework = EvaluationFramework(
        evaluators=[
            FaithfulnessEvaluator(llm=object()),
            CoverageEvaluator(llm=_coverage_judge()),
            InstructionAdherenceEvaluator(llm=instruction_judge),
        ]
    )
    results = framework.evaluate(CASE)

    assert set(results) == {"faithfulness", "coverage", "instruction_adherence"}


def test_framework_runs_selected_metrics():
    framework = EvaluationFramework(
        evaluators=[
            FaithfulnessEvaluator(llm=object()),
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
