"""Metric-aware required-field validation and metric-subset selection (no LLM)."""

import sys
import types

import pytest

from idp_eval import (
    EvaluationCase,
    EvaluationFramework,
    FaithfulnessEvaluator,
    InstructionAdherenceEvaluator,
    CoverageEvaluator,
    TaskCoverageEvaluator,
)


@pytest.fixture(autouse=True)
def fake_phoenix(monkeypatch):
    """Installs a fake ``phoenix.evals.metrics`` so FaithfulnessEvaluator builds."""

    class FakeFaithfulnessEvaluator:
        def __init__(self, llm):
            self.llm = llm

        def evaluate(self, record):
            class _R:
                score, label, explanation = 1.0, "faithful", "Grounded."

            return [_R()]

    phoenix_mod = types.ModuleType("phoenix")
    evals_mod = types.ModuleType("phoenix.evals")
    metrics_mod = types.ModuleType("phoenix.evals.metrics")
    metrics_mod.FaithfulnessEvaluator = FakeFaithfulnessEvaluator
    monkeypatch.setitem(sys.modules, "phoenix", phoenix_mod)
    monkeypatch.setitem(sys.modules, "phoenix.evals", evals_mod)
    monkeypatch.setitem(sys.modules, "phoenix.evals.metrics", metrics_mod)


class ScriptedJudge:
    def __init__(self, *responses):
        self._responses = list(responses)

    def generate_object(self, prompt, schema):
        return self._responses.pop(0)


class CountingJudge:
    """Fails loudly if called; used to prove validation runs before judge work."""

    def __init__(self):
        self.calls = 0

    def generate_object(self, prompt, schema):
        self.calls += 1
        raise AssertionError("judge must not be called on validation failure")


def _source_judge():
    return ScriptedJudge(
        {
            "items": [
                {
                    "source_item": "a",
                    "meaningfully_present": True,
                    "fully_present": True,
                }
            ]
        }
    )


# --- required fields declared per evaluator ---------------------------------


def test_required_fields_are_declared():
    assert CoverageEvaluator.required_fields == ("context", "output")
    assert TaskCoverageEvaluator.required_fields == ("input", "context", "output")
    assert FaithfulnessEvaluator.required_fields == ("context", "output")
    assert InstructionAdherenceEvaluator.required_fields == (
        "instructions", "output",
    )


# --- source coverage --------------------------------------------------------


def test_source_coverage_requires_context_and_output_only():
    ev = CoverageEvaluator(llm=object())
    ev.validate_case(EvaluationCase(context="c", output="o"))  # no input needed


def test_source_coverage_missing_context_fails():
    ev = CoverageEvaluator(llm=object())
    with pytest.raises(ValueError, match="requires non-empty `context`"):
        ev.validate_case(EvaluationCase(output="o"))


def test_source_coverage_missing_output_fails():
    ev = CoverageEvaluator(llm=object())
    with pytest.raises(ValueError, match="requires non-empty `output`"):
        ev.validate_case(EvaluationCase(context="c"))


# --- task coverage ----------------------------------------------------------


def test_task_coverage_requires_input_context_output():
    ev = TaskCoverageEvaluator(llm=object())
    ev.validate_case(EvaluationCase(input="i", context="c", output="o"))


@pytest.mark.parametrize("drop", ["input", "context", "output"])
def test_task_coverage_missing_any_field_fails(drop):
    fields = {"input": "i", "context": "c", "output": "o"}
    fields[drop] = "   "  # whitespace counts as missing
    ev = TaskCoverageEvaluator(llm=object())
    with pytest.raises(ValueError, match=f"requires non-empty `{drop}`"):
        ev.validate_case(EvaluationCase(**fields))


# --- faithfulness -----------------------------------------------------------


def test_faithfulness_requires_context_and_output():
    ev = FaithfulnessEvaluator(llm=object())
    ev.validate_case(EvaluationCase(context="c", output="o"))
    with pytest.raises(ValueError, match="requires non-empty `context`"):
        ev.validate_case(EvaluationCase(output="o"))
    with pytest.raises(ValueError, match="requires non-empty `output`"):
        ev.validate_case(EvaluationCase(context="c"))


# --- instruction adherence --------------------------------------------------


def test_instruction_adherence_requires_instructions_and_output():
    ev = InstructionAdherenceEvaluator(llm=object())
    ev.validate_case(EvaluationCase(instructions="do x", output="o"))
    with pytest.raises(ValueError, match="requires non-empty `instructions`"):
        ev.validate_case(EvaluationCase(output="o"))
    with pytest.raises(ValueError, match="requires non-empty `output`"):
        ev.validate_case(EvaluationCase(instructions="do x"))


# --- empty-value semantics --------------------------------------------------


def test_empty_structures_count_as_missing():
    ev = CoverageEvaluator(llm=object())
    for empty in [{}, [], "", "  ", None]:
        with pytest.raises(ValueError):
            ev.validate_case(EvaluationCase(context=empty, output="o"))


def test_scalar_zero_and_false_are_present():
    ev = CoverageEvaluator(llm=object())
    ev.validate_case(EvaluationCase(context=0, output=False))  # legitimate values


# --- extra fields allowed ---------------------------------------------------


def test_extra_fields_do_not_fail_validation():
    ev = CoverageEvaluator(llm=object())
    # input + instructions present but unused by source coverage -> still valid.
    ev.validate_case(
        EvaluationCase(input="task", context="c", output="o", instructions="rules")
    )


# --- error message format ---------------------------------------------------


def test_error_message_lists_present_and_missing():
    ev = TaskCoverageEvaluator(llm=object())
    with pytest.raises(ValueError) as exc:
        ev.validate_case(EvaluationCase(context="c", output="o"))  # no input
    message = str(exc.value)
    assert "TaskCoverageEvaluator requires non-empty `input`." in message
    assert "input: missing" in message
    assert "context: present" in message
    assert "output: present" in message


# --- direct evaluate() validates before any judge call ----------------------


def test_direct_source_coverage_missing_context_errors_no_judge_call():
    judge = CountingJudge()
    with pytest.raises(ValueError, match="requires non-empty `context`"):
        CoverageEvaluator(judge, mode="g_eval").evaluate(EvaluationCase(output="o"))
    assert judge.calls == 0


def test_direct_source_coverage_missing_output_errors_no_judge_call():
    judge = CountingJudge()
    with pytest.raises(ValueError, match="requires non-empty `output`"):
        CoverageEvaluator(judge, mode="g_eval").evaluate(EvaluationCase(context="c"))
    assert judge.calls == 0


@pytest.mark.parametrize("drop", ["input", "context", "output"])
def test_direct_task_coverage_missing_field_errors_no_judge_call(drop):
    fields = {"input": "i", "context": "c", "output": "o"}
    fields[drop] = None
    judge = CountingJudge()
    with pytest.raises(ValueError, match=f"requires non-empty `{drop}`"):
        TaskCoverageEvaluator(judge).evaluate(EvaluationCase(**fields))
    assert judge.calls == 0


def test_direct_faithfulness_missing_field_errors():
    with pytest.raises(ValueError, match="requires non-empty `context`"):
        FaithfulnessEvaluator(object()).evaluate(EvaluationCase(output="o"))
    with pytest.raises(ValueError, match="requires non-empty `output`"):
        FaithfulnessEvaluator(object()).evaluate(EvaluationCase(context="c"))


def test_direct_instruction_adherence_missing_field_errors_no_judge_call():
    judge = CountingJudge()
    with pytest.raises(ValueError, match="requires non-empty `instructions`"):
        InstructionAdherenceEvaluator(judge).evaluate(EvaluationCase(output="o"))
    with pytest.raises(ValueError, match="requires non-empty `output`"):
        InstructionAdherenceEvaluator(judge).evaluate(
            EvaluationCase(instructions="do x")
        )
    assert judge.calls == 0


def test_direct_and_framework_reject_the_same_case_identically():
    case = EvaluationCase(context="c", output="o")  # missing input
    with pytest.raises(ValueError, match="requires non-empty `input`"):
        TaskCoverageEvaluator(CountingJudge()).evaluate(case)
    framework = EvaluationFramework(evaluators=[TaskCoverageEvaluator(CountingJudge())])
    with pytest.raises(ValueError, match="requires non-empty `input`"):
        framework.evaluate(case)


# --- metric subset validation respects selection ----------------------------


def _framework():
    return EvaluationFramework(
        evaluators=[
            CoverageEvaluator(_source_judge(), mode="g_eval", verbose=True),
            TaskCoverageEvaluator(object()),
            FaithfulnessEvaluator(object()),
        ]
    )


def test_default_evaluate_fails_when_a_selected_metric_is_unsatisfied():
    # No input -> TaskCoverageEvaluator (selected by default) is unsatisfied.
    case = EvaluationCase(context="c", output="o")
    with pytest.raises(ValueError, match="requires non-empty `input`"):
        _framework().evaluate(case)


def test_metric_subset_skips_validation_for_unselected():
    case = EvaluationCase(context="c", output="o")  # no input
    results = _framework().evaluate(
        case, metrics=["coverage", "faithfulness"]
    )
    assert set(results) == {"coverage", "faithfulness"}


def test_unknown_metric_raises_clear_error():
    case = EvaluationCase(context="c", output="o")
    with pytest.raises(KeyError, match="Unknown metric"):
        _framework().evaluate(case, metrics=["not_configured_metric"])
