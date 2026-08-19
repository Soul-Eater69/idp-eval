"""Metric-aware required-field validation and subset selection (offline)."""

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


@pytest.fixture(autouse=True)
def fake_phoenix(monkeypatch):
    class FakeFaithfulnessEvaluator:
        def __init__(self, llm):
            self.llm = llm

        def evaluate(self, record):
            class _Result:
                score, label, explanation = 1.0, "faithful", "Grounded."

            return [_Result()]

    phoenix_mod = types.ModuleType("phoenix")
    evals_mod = types.ModuleType("phoenix.evals")
    metrics_mod = types.ModuleType("phoenix.evals.metrics")
    metrics_mod.FaithfulnessEvaluator = FakeFaithfulnessEvaluator
    monkeypatch.setitem(sys.modules, "phoenix", phoenix_mod)
    monkeypatch.setitem(sys.modules, "phoenix.evals", evals_mod)
    monkeypatch.setitem(sys.modules, "phoenix.evals.metrics", metrics_mod)


class CountingJudge:
    def __init__(self):
        self.calls = 0

    def generate_object(self, prompt, schema):
        self.calls += 1
        return {
            "items": [
                {
                    "source_item": "A",
                    "meaningfully_present": True,
                    "fully_present": True,
                }
            ]
        }


def test_required_fields_are_declared():
    assert CoverageEvaluator.required_fields == ("context", "output")
    assert FaithfulnessEvaluator.required_fields == ("context", "output")
    assert InstructionAdherenceEvaluator.required_fields == (
        "instructions",
        "output",
    )


def test_coverage_requires_context_and_output_only():
    evaluator = CoverageEvaluator(object())
    evaluator.validate_case(EvaluationCase(context="c", output="o"))
    with pytest.raises(ValueError, match="requires non-empty `context`"):
        evaluator.validate_case(EvaluationCase(output="o"))
    with pytest.raises(ValueError, match="requires non-empty `output`"):
        evaluator.validate_case(EvaluationCase(context="c"))


def test_empty_structures_count_as_missing_but_zero_and_false_are_present():
    evaluator = CoverageEvaluator(object())
    for empty in ({}, [], "", "  ", None):
        with pytest.raises(ValueError):
            evaluator.validate_case(EvaluationCase(context=empty, output="o"))
    evaluator.validate_case(EvaluationCase(context=0, output=False))


def test_extra_fields_are_allowed():
    CoverageEvaluator(object()).validate_case(
        EvaluationCase(input="task", context="c", output="o", instructions="rules")
    )


@pytest.mark.parametrize("missing", ["context", "output"])
def test_direct_evaluate_validates_before_judge_call(missing):
    fields = {"context": "c", "output": "o"}
    fields[missing] = None
    judge = CountingJudge()
    with pytest.raises(ValueError, match=f"requires non-empty `{missing}`"):
        CoverageEvaluator(judge).evaluate(EvaluationCase(**fields))
    assert judge.calls == 0


def test_faithfulness_and_instruction_adherence_validation_unchanged():
    with pytest.raises(ValueError, match="requires non-empty `context`"):
        FaithfulnessEvaluator(object()).evaluate(EvaluationCase(output="o"))
    with pytest.raises(ValueError, match="requires non-empty `instructions`"):
        InstructionAdherenceEvaluator(object()).evaluate(EvaluationCase(output="o"))


def _framework():
    return EvaluationFramework(
        evaluators=[
            CoverageEvaluator(CountingJudge()),
            FaithfulnessEvaluator(object()),
            InstructionAdherenceEvaluator(object()),
        ]
    )


def test_default_evaluate_validates_every_selected_metric():
    with pytest.raises(ValueError, match="requires non-empty `instructions`"):
        _framework().evaluate(EvaluationCase(context="c", output="o"))


def test_metric_subset_skips_unselected_validation():
    results = _framework().evaluate(
        EvaluationCase(context="c", output="o"),
        metrics=["coverage", "faithfulness"],
    )
    assert set(results) == {"coverage", "faithfulness"}


def test_unknown_metric_raises_clear_error():
    with pytest.raises(KeyError, match="Unknown metric"):
        _framework().evaluate(
            EvaluationCase(context="c", output="o"),
            metrics=["not_configured"],
        )
