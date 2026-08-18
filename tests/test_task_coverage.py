"""TaskCoverageEvaluator tests (metric ``task_coverage``)."""

import pytest

from idp_eval import EvaluationCase, TaskCoverageEvaluator


class ScriptedJudge:
    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def generate_object(self, prompt, schema: dict) -> dict:
        self.calls.append({"prompt": prompt, "schema": schema})
        if not self._responses:
            raise AssertionError("Unexpected extra judge call")
        return self._responses.pop(0)


def _extract(*reqs: str) -> dict:
    return {"requirements": [{"requirement": r} for r in reqs]}


def _classify(*entries) -> dict:
    return {
        "requirements": [
            {"id": rid, "meaningfully_present": p, "fully_present": f, "reason": "r"}
            for (rid, p, f) in entries
        ]
    }


CASE = EvaluationCase(
    input="UNIQUE_TASK generate onboarding epic",
    context="UNIQUE_CONTEXT onboarding goals",
    output="UNIQUE_OUTPUT epic body",
)


def _task_judge():
    return ScriptedJudge(
        _extract("Users can view invoices.", "Invoices show total amount due."),
        _classify(("r1", True, True), ("r2", False, False)),
    )


# --- metric identity & field scoping ----------------------------------------


def test_task_metric_name():
    judge = ScriptedJudge(_extract("a"), _classify(("r1", True, True)))
    assert TaskCoverageEvaluator(llm=judge).evaluate(CASE).metric == "task_coverage"


def test_task_stage1_receives_input_and_context_not_output():
    judge = ScriptedJudge(_extract("a"), _classify(("r1", True, True)))
    TaskCoverageEvaluator(llm=judge).evaluate(CASE)
    extract_user = judge.calls[0]["prompt"][1]["content"]
    assert CASE.input in extract_user and CASE.context in extract_user
    assert CASE.output not in extract_user   # output never visible to extraction


def test_task_details_use_requirement_and_total_requirements():
    judge = ScriptedJudge(
        _extract("a", "b"),
        _classify(("r1", True, True), ("r2", False, False)),
    )
    result = TaskCoverageEvaluator(llm=judge).evaluate(CASE)
    assert result.score == 0.5
    assert result.details["total_requirements"] == 2
    assert "requirement" in result.details["items"][0]
    assert "62.5%" not in (result.explanation or "")
    assert "task-relevant requirements" in result.explanation


def test_task_coverage_labels():
    complete = ScriptedJudge(_extract("a"), _classify(("r1", True, True)))
    assert TaskCoverageEvaluator(llm=complete).evaluate(CASE).label == "complete"

    incomplete = ScriptedJudge(
        _extract("a", "b"), _classify(("r1", True, True), ("r2", False, False))
    )
    assert TaskCoverageEvaluator(llm=incomplete).evaluate(CASE).label == "incomplete"

    missing = ScriptedJudge(_extract("a"), _classify(("r1", False, False)))
    assert TaskCoverageEvaluator(llm=missing).evaluate(CASE).label == "missing"


def test_task_empty_extraction_is_not_applicable():
    judge = ScriptedJudge(_extract())
    result = TaskCoverageEvaluator(llm=judge).evaluate(CASE)
    assert result.score is None
    assert result.label == "not_applicable"
    assert result.details == {
        "total_requirements": 0,
        "covered_count": 0,
        "partial_count": 0,
        "missing_count": 0,
        "items": [],
        "final_item_count": 0,
        "batch_count": 0,
        "judge_call_count": 1,
    }
    assert len(judge.calls) == 1  # Stage 2 skipped


# --- shared internals (used by the benchmark tooling) -----------------------


def test_benchmark_internal_methods_available():
    # scripts/coverage_benchmark.py targets TaskCoverageEvaluator and calls these.
    judge = _task_judge()
    ev = TaskCoverageEvaluator(llm=judge)
    reqs = ev._extract_requirements(CASE)
    assert reqs[0]["id"] == "r1" and "requirement" in reqs[0]
    judgments, batch_count = ev._run_classify(CASE, reqs)
    assert batch_count == 1
    items = ev._build_items(reqs, judgments)
    assert items[0]["status"] == "covered"


def test_source_and_task_share_scoring_module():
    # Both derive covered/partial/missing via the same shared scoring helpers.
    from idp_eval import scoring
    from idp_eval.evaluators import coverage_base

    assert coverage_base.calculate_coverage is scoring.calculate_coverage
    assert (
        coverage_base.coverage_status_from_binary
        is scoring.coverage_status_from_binary
    )
