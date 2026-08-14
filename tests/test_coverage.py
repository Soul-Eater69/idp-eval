"""Coverage evaluator behavior tests using a fake judge (no real LLM)."""

import pytest

from idp_eval import CoverageEvaluator, EvaluationCase
from idp_eval.models import EvaluationResult


class FakeJudge:
    """Judge stub returning a canned structured coverage response."""

    def __init__(self, response: dict):
        self.response = response
        self.calls: list[dict] = []

    def generate_object(self, prompt, schema: dict) -> dict:
        self.calls.append({"prompt": prompt, "schema": schema})
        return self.response


def _requirements(*pairs) -> FakeJudge:
    """Builds a judge whose response has (requirement, status) entries."""
    return FakeJudge(
        {
            "requirements": [
                {"requirement": req, "status": status, "reason": "r"}
                for req, status in pairs
            ]
        }
    )


CASE = EvaluationCase(
    input="Generate a Jira Epic from the provided Theme.",
    context="Theme with onboarding goals.",
    output="Epic body.",
)


def _evaluate(judge) -> EvaluationResult:
    return CoverageEvaluator(llm=judge).evaluate(CASE)


def test_full_coverage():
    result = _evaluate(_requirements(("a", "covered"), ("b", "covered")))
    assert result.score == 1.0
    assert result.details["covered_count"] == 2
    assert result.details["missing_count"] == 0


def test_complete_omission():
    result = _evaluate(_requirements(("a", "missing"), ("b", "missing")))
    assert result.score == 0.0
    assert result.details["missing_count"] == 2


def test_mixed_coverage_deterministic_aggregation():
    result = _evaluate(
        _requirements(
            ("a", "covered"),
            ("b", "covered"),
            ("c", "partial"),
            ("d", "missing"),
        )
    )
    assert result.score == 0.625
    assert result.details["total_requirements"] == 4
    assert "62.5%" in result.explanation
    # Every item carries its Python-computed numeric score.
    scores = [item["score"] for item in result.details["items"]]
    assert scores == [1.0, 1.0, 0.5, 0.0]


def test_compound_requirement_decomposed_by_judge():
    # The judge represents a compound sentence as two atomic requirements.
    result = _evaluate(
        _requirements(
            ("Reduce onboarding time by 25%.", "covered"),
            ("Automate manual verification.", "covered"),
        )
    )
    assert result.score == 1.0
    assert result.details["total_requirements"] == 2


def test_qualifier_partial():
    result = _evaluate(
        _requirements(("Reduce manual verification effort by 40%.", "partial"))
    )
    assert result.score == 0.5
    assert result.details["partial_count"] == 1


def test_paraphrase_counts_as_covered():
    result = _evaluate(
        _requirements(("Automate manual identity verification.", "covered"))
    )
    assert result.score == 1.0


def test_irrelevant_context_not_counted():
    # The judge only returns task-relevant requirements; irrelevant office /
    # employee details never appear, so they never affect the denominator.
    judge = _requirements(("Business risk A.", "covered"), ("Business risk B.", "covered"))
    result = _evaluate(judge)
    requirements = [item["requirement"] for item in result.details["items"]]
    assert requirements == ["Business risk A.", "Business risk B."]
    assert result.score == 1.0


def test_unsupported_addition_does_not_lower_coverage():
    # Output covers all required info (plus an unsupported claim faithfulness
    # would catch). Coverage stays full: it does not do hallucination detection.
    result = _evaluate(_requirements(("Goal 1.", "covered"), ("Goal 2.", "covered")))
    assert result.score == 1.0


def test_no_requirements_is_full_and_explained():
    result = _evaluate(FakeJudge({"requirements": []}))
    assert result.score == 1.0
    assert result.details["total_requirements"] == 0
    assert (
        result.explanation
        == "No task-relevant requirements were identified in the supplied context."
    )


def test_unknown_status_fails_clearly():
    judge = FakeJudge(
        {"requirements": [{"requirement": "a", "status": "kinda", "reason": "r"}]}
    )
    with pytest.raises(ValueError, match="Unknown coverage status"):
        _evaluate(judge)


def test_missing_status_key_fails_clearly():
    judge = FakeJudge({"requirements": [{"requirement": "a", "reason": "r"}]})
    with pytest.raises(KeyError):
        _evaluate(judge)


def test_percentage_formatting_whole_number():
    result = _evaluate(_requirements(("a", "covered"), ("b", "missing")))
    # 0.5 -> "50%", not "50.0%".
    assert "50%" in result.explanation
