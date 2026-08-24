"""Offline progress observability tests for bulk evaluation."""

import asyncio

import pytest

from idp_eval import EvaluationCase, EvaluationFramework, EvaluationResult, Evaluator


class ConstantEvaluator(Evaluator):
    """Returns a configured result without judge or network work."""

    def __init__(self, name: str, score: float | None, label: str | None):
        self._name = name
        self._score = score
        self._label = label
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    def evaluate(self, case: EvaluationCase) -> EvaluationResult:
        self.calls += 1
        return EvaluationResult(
            metric=self.name,
            score=self._score,
            label=self._label,
            explanation="offline result",
        )


class ExplodingEvaluator(Evaluator):
    """Raises one stable exception object to verify exact propagation."""

    name = "explode"

    def __init__(self, error: Exception):
        self.error = error

    def evaluate(self, case: EvaluationCase) -> EvaluationResult:
        raise self.error


class AsyncOrderEvaluator(Evaluator):
    """Completes cases out of order while returning identifiable scores."""

    name = "order"

    def evaluate(self, case: EvaluationCase) -> EvaluationResult:
        raise AssertionError("the native async evaluator path should be used")

    async def a_evaluate(
        self, case: EvaluationCase, *, judge_limiter=None
    ) -> EvaluationResult:
        await asyncio.sleep(case.output["delay"])
        return EvaluationResult(
            metric=self.name,
            score=case.output["score"],
            label=case.output["label"],
            explanation="offline async result",
        )


def _captured_text(capsys) -> str:
    captured = capsys.readouterr()
    return captured.out + captured.err


def test_evaluate_many_is_quiet_by_default(capsys):
    framework = EvaluationFramework([ConstantEvaluator("quality", 1.0, "pass")])

    results = framework.evaluate_many(
        [EvaluationCase(output="a"), EvaluationCase(output="b")]
    )

    assert len(results) == 2
    assert _captured_text(capsys) == ""


def test_sequential_progress_counts_cases_and_displays_all_metrics(capsys):
    framework = EvaluationFramework(
        [
            ConstantEvaluator("quality", 2 / 3, "partial"),
            ConstantEvaluator("optional", None, "not_applicable"),
        ]
    )
    cases = [
        EvaluationCase(output="a", case_id="case-a"),
        EvaluationCase(output="b", case_id="case-b"),
    ]

    framework.evaluate_many(cases, show_progress=True)

    output = _captured_text(capsys)
    assert "✓ [1/2] case=case-a completed in" in output
    assert "✓ [2/2] case=case-b completed in" in output
    assert "quality=0.667 (partial)" in output
    assert "optional=None (not_applicable)" in output
    assert "Evaluation complete: 2/2 cases in" in output


def test_progress_failure_is_reported_and_original_exception_propagates(capsys):
    error = RuntimeError("gateway timeout")
    framework = EvaluationFramework([ExplodingEvaluator(error)])

    with pytest.raises(RuntimeError) as caught:
        framework.evaluate_many(
            [EvaluationCase(output="safe", case_id="failed-case")],
            show_progress=True,
        )

    assert caught.value is error
    output = _captured_text(capsys)
    assert "✗ [1/1] case=failed-case failed after" in output
    assert "RuntimeError: gateway timeout" in output
    assert "Evaluation complete" not in output


def test_async_progress_follows_completion_while_results_preserve_input_order(capsys):
    framework = EvaluationFramework([AsyncOrderEvaluator()])
    cases = [
        EvaluationCase(
            output={"delay": 0.03, "score": 0.0, "label": "slow"},
            case_id="slow-first",
        ),
        EvaluationCase(
            output={"delay": 0.0, "score": 1.0, "label": "fast"},
            case_id="fast-second",
        ),
    ]

    results = asyncio.run(
        framework.a_evaluate_many(
            cases, max_concurrency=2, show_progress=True
        )
    )

    assert [result["order"].score for result in results] == [0.0, 1.0]
    output = _captured_text(capsys)
    assert output.index("case=fast-second") < output.index("case=slow-first")
    assert "✓ [1/2] case=fast-second" in output
    assert "✓ [2/2] case=slow-first" in output
    assert "Evaluation complete: 2/2 cases in" in output


def test_async_evaluate_many_is_quiet_by_default(capsys):
    framework = EvaluationFramework([AsyncOrderEvaluator()])

    results = asyncio.run(
        framework.a_evaluate_many(
            [
                EvaluationCase(
                    output={"delay": 0.0, "score": 1.0, "label": "pass"}
                )
            ]
        )
    )

    assert results[0]["order"].score == 1.0
    assert _captured_text(capsys) == ""


def test_scoped_case_is_one_top_level_progress_item(capsys):
    evaluator = ConstantEvaluator("quality", 1.0, "pass")
    framework = EvaluationFramework([evaluator])
    case = EvaluationCase(
        output=["a", "b"],
        case_id="scoped",
        evaluation_scope="both",
    )

    results = framework.evaluate_many([case], show_progress=True)

    assert evaluator.calls == 3
    assert len(results[0]["individual"]) == 2
    output = _captured_text(capsys)
    assert output.count("✓ [1/1] case=scoped") == 1
    assert "Evaluation complete: 1/1 cases in" in output


def test_async_scoped_case_is_one_top_level_progress_item(capsys):
    evaluator = ConstantEvaluator("quality", 1.0, "pass")
    framework = EvaluationFramework([evaluator])
    case = EvaluationCase(
        output=["a", "b"],
        case_id="async-scoped",
        evaluation_scope="both",
    )

    results = asyncio.run(
        framework.a_evaluate_many(
            [case], max_concurrency=2, show_progress=True
        )
    )

    assert evaluator.calls == 3
    assert len(results[0]["individual"]) == 2
    output = _captured_text(capsys)
    assert output.count("✓ [1/1] case=async-scoped") == 1
    assert "Evaluation complete: 1/1 cases in" in output


def test_progress_uses_input_index_when_case_id_is_absent(capsys):
    framework = EvaluationFramework([ConstantEvaluator("quality", 1.0, "pass")])

    framework.evaluate_many(
        [EvaluationCase(output="a")], show_progress=True
    )

    assert "case=index 0" in _captured_text(capsys)
