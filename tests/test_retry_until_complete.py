"""Offline tests for checkpoint-backed bulk retry rounds."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from idp_eval import EvaluationCase, EvaluationFramework, EvaluationResult
from idp_eval.models import Evaluator
from idp_eval.output import PersistenceError

openpyxl = pytest.importorskip("openpyxl")


class OutcomeEvaluator(Evaluator):
    """Returns or raises queued outcomes while exposing exact call history."""

    annotator_kind = "CODE"

    def __init__(self, name, outcomes):
        self._name = name
        self._outcomes = {
            case_id: list(values) for case_id, values in outcomes.items()
        }
        self.calls: list[str | None] = []

    @property
    def name(self):
        return self._name

    def evaluate(self, case):
        self.calls.append(case.case_id)
        outcome = self._outcomes[case.case_id].pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return EvaluationResult(
            metric=self.name,
            score=float(outcome),
            label="pass",
            explanation="completed",
        )


class AsyncOutcomeEvaluator(OutcomeEvaluator):
    """Async counterpart that proves the non-blocking retry path."""

    def evaluate(self, case):  # pragma: no cover - async path is required
        raise AssertionError("sync evaluator path must not run")

    async def a_evaluate(self, case, *, judge_limiter):
        async with judge_limiter:
            self.calls.append(case.case_id)
            outcome = self._outcomes[case.case_id].pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return EvaluationResult(
                metric=self.name,
                score=float(outcome),
                label="pass",
                explanation="completed",
            )


def _framework(tmp_path, evaluators, *, filename="retry.xlsx"):
    return EvaluationFramework(
        evaluators,
        output="excel",
        excel_path=str(tmp_path / filename),
        resume=True,
    )


def _rows(path, sheet="evaluations"):
    values = list(
        openpyxl.load_workbook(path)[sheet].iter_rows(values_only=True)
    )
    header, *data = values
    return [dict(zip(header, row)) for row in data]


def test_retry_default_is_off_and_does_not_sleep(monkeypatch):
    evaluator = OutcomeEvaluator("quality", {"a": [TimeoutError("later")]})
    sleeps = []
    monkeypatch.setattr("idp_eval.framework.time.sleep", sleeps.append)

    results = EvaluationFramework([evaluator]).evaluate_many(
        [EvaluationCase(case_id="a")], on_error="continue"
    )

    assert results[0]["quality"].label == "error"
    assert evaluator.calls == ["a"]
    assert sleeps == []


def test_sync_retry_succeeds_on_second_round_and_upserts_error(
    monkeypatch, tmp_path
):
    path = tmp_path / "retry.xlsx"
    evaluator = OutcomeEvaluator(
        "quality", {"a": [TimeoutError("later"), 1.0]}
    )
    sleeps = []
    monkeypatch.setattr("idp_eval.framework.time.sleep", sleeps.append)

    results = _framework(tmp_path, [evaluator]).evaluate_many(
        [EvaluationCase(case_id="a")],
        on_error="continue",
        retry_until_complete=True,
        retry_interval_seconds=7,
    )

    assert results[0]["quality"].score == 1.0
    assert evaluator.calls == ["a", "a"]
    assert sleeps == [7.0]
    rows = _rows(path)
    assert len(rows) == 1
    assert rows[0]["status"] == "success"
    assert rows[0]["label"] == "pass"


def test_retry_enabled_does_not_sleep_or_add_round_after_initial_success(
    monkeypatch, tmp_path
):
    evaluator = OutcomeEvaluator("quality", {"a": [1.0]})
    sleeps = []
    monkeypatch.setattr("idp_eval.framework.time.sleep", sleeps.append)

    results = _framework(tmp_path, [evaluator]).evaluate_many(
        [EvaluationCase(case_id="a")],
        on_error="continue",
        retry_until_complete=True,
        retry_interval_seconds=3,
    )

    assert results[0]["quality"].label == "pass"
    assert evaluator.calls == ["a"]
    assert sleeps == []


def test_sync_retry_skips_completed_metric_and_successful_case_trace(
    monkeypatch, tmp_path
):
    complete = OutcomeEvaluator("complete", {"a": [0.8], "b": [0.9]})
    retry = OutcomeEvaluator(
        "retry", {"a": [1.0], "b": [TimeoutError("later"), 0.7]}
    )
    root_cases = []

    @contextmanager
    def case_span(case_id, *_args, **_kwargs):
        root_cases.append(case_id)
        yield SimpleNamespace(trace_id=None, span_id=None)

    monkeypatch.setattr(
        "idp_eval.framework.tracing.case_evaluation_span", case_span
    )
    monkeypatch.setattr("idp_eval.framework.time.sleep", lambda _seconds: None)

    results = _framework(tmp_path, [complete, retry]).evaluate_many(
        [EvaluationCase(case_id="a"), EvaluationCase(case_id="b")],
        on_error="continue",
        retry_until_complete=True,
        retry_interval_seconds=1,
    )

    assert all(result["retry"].label == "pass" for result in results)
    assert complete.calls == ["a", "b"]
    assert retry.calls == ["a", "b", "b"]
    assert root_cases == ["a", "b", "b"]


def test_sync_retry_supports_multiple_rounds(monkeypatch, tmp_path):
    evaluator = OutcomeEvaluator(
        "quality",
        {"a": [TimeoutError("one"), ConnectionError("two"), 1.0]},
    )
    sleeps = []
    monkeypatch.setattr("idp_eval.framework.time.sleep", sleeps.append)

    result = _framework(tmp_path, [evaluator]).evaluate_many(
        [EvaluationCase(case_id="a")],
        on_error="continue",
        retry_until_complete=True,
        retry_interval_seconds=2.5,
    )

    assert result[0]["quality"].label == "pass"
    assert evaluator.calls == ["a", "a", "a"]
    assert sleeps == [2.5, 2.5]


@pytest.mark.parametrize(
    "error", [ValueError("malformed judge response"), RuntimeError("bug")]
)
def test_non_operational_failures_raise_without_retry(
    error, monkeypatch, tmp_path
):
    evaluator = OutcomeEvaluator("quality", {"a": [error]})
    sleeps = []
    monkeypatch.setattr("idp_eval.framework.time.sleep", sleeps.append)

    with pytest.raises(type(error), match=str(error)):
        _framework(tmp_path, [evaluator]).evaluate_many(
            [EvaluationCase(case_id="a")],
            on_error="continue",
            retry_until_complete=True,
            retry_interval_seconds=1,
        )

    assert evaluator.calls == ["a"]
    assert sleeps == []


def test_persistence_failure_raises_without_retry(monkeypatch, tmp_path):
    class FailingWriter:
        def write(self, records):
            raise OSError("disk unavailable")

    evaluator = OutcomeEvaluator("quality", {"a": [1.0]})
    framework = _framework(tmp_path, [evaluator])
    framework._writers.append(FailingWriter())
    sleeps = []
    monkeypatch.setattr("idp_eval.framework.time.sleep", sleeps.append)

    with pytest.raises(PersistenceError, match="disk unavailable") as exc_info:
        framework.evaluate_many(
            [EvaluationCase(case_id="a")],
            on_error="continue",
            retry_until_complete=True,
            retry_interval_seconds=1,
        )

    assert exc_info.value.results["quality"].score == 1.0
    assert evaluator.calls == ["a"]
    assert sleeps == []


@pytest.mark.parametrize(
    "interval", [0, -1, True, "1", float("nan"), float("inf")]
)
def test_invalid_retry_interval_is_rejected_before_work(interval, tmp_path):
    evaluator = OutcomeEvaluator("quality", {"a": [1.0]})

    with pytest.raises(ValueError, match="positive real number"):
        _framework(tmp_path, [evaluator]).evaluate_many(
            [EvaluationCase(case_id="a")],
            on_error="continue",
            retry_interval_seconds=interval,
        )

    assert evaluator.calls == []


def test_invalid_retry_flag_and_mode_are_rejected_before_work(tmp_path):
    evaluator = OutcomeEvaluator("quality", {"a": [1.0]})
    framework = _framework(tmp_path, [evaluator])

    with pytest.raises(ValueError, match="must be a bool"):
        framework.evaluate_many(
            [EvaluationCase(case_id="a")],
            retry_until_complete="yes",
        )
    with pytest.raises(ValueError, match="requires on_error='continue'"):
        framework.evaluate_many(
            [EvaluationCase(case_id="a")], retry_until_complete=True
        )

    assert evaluator.calls == []


@pytest.mark.parametrize(
    "framework_kwargs",
    [
        {},
        {"output": "excel", "excel_path": "unused.xlsx"},
    ],
)
def test_retry_requires_resumable_excel_checkpoint_before_work(
    framework_kwargs, tmp_path
):
    evaluator = OutcomeEvaluator("quality", {"a": [1.0]})
    if "excel_path" in framework_kwargs:
        framework_kwargs["excel_path"] = str(tmp_path / "not-resumable.xlsx")
    framework = EvaluationFramework([evaluator], **framework_kwargs)

    with pytest.raises(ValueError, match="requires resume=True"):
        framework.evaluate_many(
            [EvaluationCase(case_id="a")],
            on_error="continue",
            retry_until_complete=True,
        )

    assert evaluator.calls == []


def test_async_retry_uses_async_sleep_and_returns_success(monkeypatch, tmp_path):
    evaluator = AsyncOutcomeEvaluator(
        "quality", {"a": [TimeoutError("later"), 1.0]}
    )
    async_sleeps = []

    async def fake_sleep(seconds):
        async_sleeps.append(seconds)

    monkeypatch.setattr("idp_eval.framework.asyncio.sleep", fake_sleep)
    monkeypatch.setattr(
        "idp_eval.framework.time.sleep",
        lambda _seconds: pytest.fail("blocking sleep used by async retry"),
    )

    async def scenario():
        return await _framework(tmp_path, [evaluator]).a_evaluate_many(
            [EvaluationCase(case_id="a")],
            on_error="continue",
            retry_until_complete=True,
            retry_interval_seconds=4,
        )

    results = asyncio.run(scenario())
    assert results[0]["quality"].label == "pass"
    assert evaluator.calls == ["a", "a"]
    assert async_sleeps == [4.0]


def test_async_partial_retry_skips_completed_metric(monkeypatch, tmp_path):
    complete = AsyncOutcomeEvaluator("complete", {"a": [0.8]})
    retry = AsyncOutcomeEvaluator(
        "retry", {"a": [TimeoutError("later"), 0.7]}
    )

    async def fake_sleep(_seconds):
        return None

    monkeypatch.setattr("idp_eval.framework.asyncio.sleep", fake_sleep)

    async def scenario():
        return await _framework(tmp_path, [complete, retry]).a_evaluate_many(
            [EvaluationCase(case_id="a")],
            on_error="continue",
            retry_until_complete=True,
            retry_interval_seconds=1,
        )

    results = asyncio.run(scenario())
    assert results[0]["retry"].label == "pass"
    assert complete.calls == ["a"]
    assert retry.calls == ["a", "a"]


def test_grouped_sync_routes_through_retry(monkeypatch, tmp_path):
    evaluator = OutcomeEvaluator(
        "quality", {"g:0": [TimeoutError("later"), 1.0]}
    )
    monkeypatch.setattr("idp_eval.framework.time.sleep", lambda _seconds: None)

    results = _framework(tmp_path, [evaluator]).evaluate_groups(
        [{"group_id": "g", "outputs": ["answer"]}],
        on_error="continue",
        retry_until_complete=True,
        retry_interval_seconds=1,
    )

    assert results[0]["quality"].label == "pass"
    assert evaluator.calls == ["g:0", "g:0"]


def test_grouped_async_routes_through_retry(monkeypatch, tmp_path):
    evaluator = AsyncOutcomeEvaluator(
        "quality", {"g:0": [TimeoutError("later"), 1.0]}
    )

    async def fake_sleep(_seconds):
        return None

    monkeypatch.setattr("idp_eval.framework.asyncio.sleep", fake_sleep)

    async def scenario():
        return await _framework(tmp_path, [evaluator]).a_evaluate_groups(
            [{"group_id": "g", "outputs": ["answer"]}],
            on_error="continue",
            retry_until_complete=True,
            retry_interval_seconds=1,
        )

    results = asyncio.run(scenario())
    assert results[0]["quality"].label == "pass"
    assert evaluator.calls == ["g:0", "g:0"]


def test_retry_progress_is_round_level_and_quiet_when_disabled(
    monkeypatch, tmp_path, capsys
):
    quiet = OutcomeEvaluator(
        "quality", {"quiet": [TimeoutError("later"), 1.0]}
    )
    monkeypatch.setattr("idp_eval.framework.time.sleep", lambda _seconds: None)
    _framework(tmp_path, [quiet], filename="quiet.xlsx").evaluate_many(
        [EvaluationCase(case_id="quiet")],
        on_error="continue",
        retry_until_complete=True,
        retry_interval_seconds=1,
    )
    assert capsys.readouterr().out == ""

    visible = OutcomeEvaluator(
        "quality", {"visible": [TimeoutError("later"), 1.0]}
    )
    _framework(tmp_path, [visible], filename="visible.xlsx").evaluate_many(
        [EvaluationCase(case_id="visible")],
        on_error="continue",
        retry_until_complete=True,
        retry_interval_seconds=1,
        show_progress=True,
    )
    output = capsys.readouterr()
    text = output.out + output.err
    assert "Evaluation round 1 complete" in text
    assert "Retrying 1 failed metrics in 1 seconds" in text
    assert "Evaluation round 2 complete" in text
    assert "Evaluation retry complete" in text
