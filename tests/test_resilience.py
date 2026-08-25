"""Operational-error isolation and resumable Excel evaluation (offline)."""

from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from types import SimpleNamespace

import httpx
import pytest

from idp_eval import (
    CoverageEvaluator,
    EvaluationCase,
    EvaluationFramework,
    EvaluationResult,
    FaithfulnessEvaluator,
    HitRateAtKEvaluator,
    RelevanceAtKEvaluator,
)
from idp_eval.models import Evaluator
from idp_eval.operational_errors import classify_operational_error
from idp_eval.output import PersistenceError
from idp_eval.resume import case_fingerprint, evaluation_fingerprint

openpyxl = pytest.importorskip("openpyxl")


class ScriptedEvaluator(Evaluator):
    """Small configurable evaluator used to count calls and raise outcomes."""

    def __init__(self, name, outcomes, *, config="v1"):
        self._name = name
        self._outcomes = {
            case_id: list(values) for case_id, values in outcomes.items()
        }
        self.calls: list[str | None] = []
        self.config = config

    @property
    def name(self):
        return self._name

    def resume_signature(self):
        return {"contract_version": 1, "config": self.config}

    def evaluate(self, case):
        self.calls.append(case.case_id)
        values = self._outcomes.get(case.case_id)
        value = values.pop(0) if values else 1.0
        if isinstance(value, BaseException):
            raise value
        return EvaluationResult(
            self.name,
            float(value),
            "pass",
            "completed",
            {"case": case.case_id, "value": value},
        )


class AsyncScriptedEvaluator(Evaluator):
    name = "async_metric"

    def __init__(self, *, gate=None, fast_published=None):
        self.gate = gate
        self.fast_published = fast_published
        self.slow_finished = False

    def evaluate(self, case):  # pragma: no cover - async path is expected
        raise AssertionError("sync path should not run")

    async def a_evaluate(self, case, *, judge_limiter):
        async with judge_limiter:
            if case.case_id == "slow":
                await self.gate.wait()
                self.slow_finished = True
            return EvaluationResult(
                self.name, 1.0, "pass", f"completed {case.case_id}"
            )


class RecordingWriter:
    def __init__(self, event=None):
        self.case_ids = []
        self.event = event
        self.active = 0
        self.max_active = 0

    def write(self, records):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.case_ids.extend(record.case_id for record in records)
        if self.event is not None and any(
            record.case_id == "fast" for record in records
        ):
            self.event.set()
        self.active -= 1


def _rows(path, sheet="evaluations"):
    values = list(
        openpyxl.load_workbook(path)[sheet].iter_rows(values_only=True)
    )
    header, *data = values
    return [dict(zip(header, row)) for row in data]


class QueuedJudge:
    """Captures prompts and returns/raises deterministic queued outcomes."""

    model = "offline-audit-model"

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def generate_object(self, prompt, schema):
        self.calls.append({"prompt": prompt, "schema": schema})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _covered_response(source_item):
    return {
        "items": [
            {
                "source_item": source_item,
                "meaningfully_present": True,
                "fully_present": True,
                "reason": "",
            }
        ],
        "overall_reason": "The source item is represented.",
    }


def _supported_response(claim):
    return {
        "claims": [
            {"claim": claim, "status": "supported", "reason": ""}
        ],
        "overall_reason": "The claim is grounded in context.",
    }


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_http_provider_statuses_are_operational(status):
    request = httpx.Request("POST", "https://provider.invalid")
    response = httpx.Response(status, request=request)
    error = httpx.HTTPStatusError(
        "provider failed", request=request, response=response
    )
    info = classify_operational_error(error)
    assert info is not None
    assert info.retryable is True
    assert info.status_code == status


def test_rate_limit_headers_are_preserved_without_exposing_body():
    request = httpx.Request("POST", "https://provider.invalid")
    response = httpx.Response(
        429,
        headers={"retry-after": "30", "x-request-id": "request-123"},
        request=request,
    )
    info = classify_operational_error(
        httpx.HTTPStatusError("throttled", request=request, response=response)
    )
    assert info is not None
    assert info.retry_after_seconds == 30.0
    assert info.request_id == "request-123"


def test_timeout_connection_and_wrapped_errors_are_operational():
    request = httpx.Request("POST", "https://provider.invalid")
    for error in (
        TimeoutError("timed out"),
        httpx.ReadTimeout("timed out", request=request),
        httpx.ConnectError("connection failed", request=request),
    ):
        wrapped = RuntimeError("judge wrapper")
        wrapped.__cause__ = error
        assert classify_operational_error(wrapped) is not None


def test_openai_operational_exception_types_are_classified():
    openai = pytest.importorskip("openai")
    request = httpx.Request("POST", "https://provider.invalid")
    rate_response = httpx.Response(429, request=request)
    server_response = httpx.Response(500, request=request)
    errors = (
        openai.RateLimitError("rate limited", response=rate_response, body=None),
        openai.APIConnectionError(request=request),
        openai.APITimeoutError(request),
        openai.InternalServerError(
            "server failed", response=server_response, body=None
        ),
    )
    for error in errors:
        info = classify_operational_error(error)
        assert info is not None
        assert info.provider == "openai"
        assert info.retryable is True


def test_phoenix_rate_limit_fields_are_preserved():
    rate_limiters = pytest.importorskip("phoenix.evals.rate_limiters")
    error = rate_limiters.RateLimitError(
        current_rate_tokens_per_sec=2.0,
        initial_rate_tokens_per_sec=8.0,
        enforcement_window_seconds=30.0,
    )
    info = classify_operational_error(error)
    assert info is not None
    assert info.provider == "phoenix"
    assert info.current_rate_tokens_per_sec == 2.0
    assert info.initial_rate_tokens_per_sec == 8.0
    assert info.enforcement_window_seconds == 30.0


@pytest.mark.parametrize("error", [ValueError("bad"), TypeError("bug")])
def test_programming_and_validation_errors_are_not_operational(error):
    assert classify_operational_error(error) is None


def test_arbitrary_bug_with_status_code_is_not_misclassified():
    class EvaluatorBug(RuntimeError):
        status_code = 500

    assert classify_operational_error(EvaluatorBug("bug")) is None


def test_continue_isolates_one_metric_and_default_still_raises():
    case = EvaluationCase(case_id="a", output="answer")
    successful = ScriptedEvaluator("successful", {"a": [0.8, 0.8]})
    failing = ScriptedEvaluator(
        "failing", {"a": [TimeoutError("provider timeout"), TimeoutError()]}
    )
    framework = EvaluationFramework([successful, failing])

    results = framework.evaluate(case, on_error="continue")
    assert results["successful"].score == 0.8
    assert results["failing"].score is None
    assert results["failing"].label == "error"
    assert results["failing"].details["status"] == "error"

    with pytest.raises(TimeoutError):
        framework.evaluate(case)


def test_continue_runs_unrelated_metric_after_operational_failure():
    calls = []

    class OrderedEvaluator(Evaluator):
        def __init__(self, name, failure=None):
            self._name = name
            self.failure = failure

        @property
        def name(self):
            return self._name

        def evaluate(self, case):
            calls.append(self.name)
            if self.failure is not None:
                raise self.failure
            return EvaluationResult(self.name, 1.0, "pass", "ok")

    framework = EvaluationFramework(
        [
            OrderedEvaluator("metric_a"),
            OrderedEvaluator("metric_b", TimeoutError("throttled")),
            OrderedEvaluator("metric_c"),
        ],
    )
    results = framework.evaluate(
        EvaluationCase(case_id="one"), on_error="continue"
    )
    assert calls == ["metric_a", "metric_b", "metric_c"]
    assert results["metric_a"].score == 1.0
    assert results["metric_b"].label == "error"
    assert results["metric_c"].score == 1.0

    calls.clear()
    with pytest.raises(TimeoutError):
        framework.evaluate(EvaluationCase(case_id="one"))
    assert calls == ["metric_a", "metric_b"]


def test_non_operational_and_persistence_failures_are_never_converted():
    broken = ScriptedEvaluator("broken", {"a": [RuntimeError("code bug")]})
    with pytest.raises(RuntimeError, match="code bug"):
        EvaluationFramework([broken]).evaluate(
            EvaluationCase(case_id="a"), on_error="continue"
        )

    class FailingWriter:
        def write(self, records):
            raise OSError("disk full")

    framework = EvaluationFramework([ScriptedEvaluator("ok", {})])
    framework._writers = [FailingWriter()]
    with pytest.raises(PersistenceError) as exc:
        framework.evaluate(EvaluationCase(case_id="a"), on_error="continue")
    assert exc.value.results["ok"].score == 1.0


def test_shared_retrieval_operational_failure_marks_dependents_once():
    class Judge:
        def __init__(self):
            self.calls = 0

        def generate_object(self, prompt, schema):
            self.calls += 1
            raise TimeoutError("retrieval provider timeout")

    judge = Judge()
    unrelated = ScriptedEvaluator("unrelated", {})
    framework = EvaluationFramework(
        [
            RelevanceAtKEvaluator(2, judge),
            HitRateAtKEvaluator(2, judge),
            unrelated,
        ]
    )
    results = framework.evaluate(
        EvaluationCase(
            case_id="r",
            input="query",
            retrieved_documents=["one", "two"],
        ),
        on_error="continue",
    )
    assert judge.calls == 1
    assert results["relevance_at_2"].label == "error"
    assert results["hit_rate_at_2"].label == "error"
    assert results["unrelated"].score == 1.0

    raising_judge = Judge()
    raising_unrelated = ScriptedEvaluator("unrelated", {})
    raising_framework = EvaluationFramework(
        [
            RelevanceAtKEvaluator(2, raising_judge),
            HitRateAtKEvaluator(2, raising_judge),
            raising_unrelated,
        ]
    )
    with pytest.raises(TimeoutError):
        raising_framework.evaluate(
            EvaluationCase(
                case_id="r",
                input="query",
                retrieved_documents=["one", "two"],
            )
        )
    assert raising_judge.calls == 1
    assert raising_unrelated.calls == []


def test_many_persists_success_error_and_later_success(tmp_path):
    path = tmp_path / "many.xlsx"
    evaluator = ScriptedEvaluator(
        "quality", {"a": [1.0], "b": [TimeoutError("throttled")], "c": [0.5]}
    )
    results = EvaluationFramework(
        [evaluator], output="excel", excel_path=str(path)
    ).evaluate_many(
        [EvaluationCase(case_id=value) for value in ("a", "b", "c")],
        on_error="continue",
    )
    assert [result["quality"].label for result in results] == [
        "pass",
        "error",
        "pass",
    ]
    rows = _rows(path)
    assert [row["key_id"] for row in rows] == ["a", "b", "c"]
    assert [row["status"] for row in rows] == ["success", "error", "success"]


def test_not_applicable_is_persisted_as_success(tmp_path):
    class NotApplicable(Evaluator):
        name = "optional"

        def evaluate(self, case):
            return EvaluationResult(
                self.name, None, "not_applicable", "nothing to judge"
            )

    path = tmp_path / "na.xlsx"
    EvaluationFramework(
        [NotApplicable()], output="excel", excel_path=str(path)
    ).evaluate(EvaluationCase(case_id="na"))
    assert _rows(path)[0]["status"] == "success"


def test_async_continue_preserves_input_order():
    class AsyncFailures(Evaluator):
        name = "async"

        def evaluate(self, case):
            raise AssertionError("sync path")

        async def a_evaluate(self, case, *, judge_limiter):
            async with judge_limiter:
                await asyncio.sleep(0.01 if case.case_id == "first" else 0)
                if case.case_id == "second":
                    raise TimeoutError("throttled")
                return EvaluationResult(self.name, 1.0, "pass", "ok")

    results = asyncio.run(
        EvaluationFramework([AsyncFailures()]).a_evaluate_many(
            [
                EvaluationCase(case_id="first"),
                EvaluationCase(case_id="second"),
                EvaluationCase(case_id="third"),
            ],
            max_concurrency=2,
            on_error="continue",
        )
    )
    assert [result["async"].label for result in results] == [
        "pass",
        "error",
        "pass",
    ]


def test_async_single_case_partial_error_and_default_raise():
    class AsyncMetric(Evaluator):
        def __init__(self, name, fail=False):
            self._name = name
            self.fail = fail

        @property
        def name(self):
            return self._name

        def evaluate(self, case):
            raise AssertionError("sync path")

        async def a_evaluate(self, case, *, judge_limiter):
            async with judge_limiter:
                if self.fail:
                    raise TimeoutError("provider timeout")
                return EvaluationResult(self.name, 1.0, "pass", "ok")

    framework = EvaluationFramework(
        [AsyncMetric("good"), AsyncMetric("bad", fail=True)]
    )
    results = asyncio.run(
        framework.a_evaluate(
            EvaluationCase(case_id="one"), on_error="continue"
        )
    )
    assert results["good"].score == 1.0
    assert results["bad"].label == "error"
    with pytest.raises(TimeoutError):
        asyncio.run(framework.a_evaluate(EvaluationCase(case_id="one")))


def test_async_persistence_error_raises_without_rerunning_evaluator():
    class AsyncSuccess(Evaluator):
        name = "quality"

        def __init__(self):
            self.calls = 0

        def evaluate(self, case):
            raise AssertionError("sync path")

        async def a_evaluate(self, case, *, judge_limiter):
            async with judge_limiter:
                self.calls += 1
                return EvaluationResult(self.name, 1.0, "pass", "ok")

    class FailingWriter:
        def write(self, records):
            raise OSError("disk full")

    evaluator = AsyncSuccess()
    framework = EvaluationFramework([evaluator])
    framework._writers = [FailingWriter()]
    with pytest.raises(PersistenceError) as exc:
        asyncio.run(
            framework.a_evaluate_many(
                [EvaluationCase(case_id="a")], on_error="continue"
            )
        )
    assert evaluator.calls == 1
    assert exc.value.results["quality"].score == 1.0


def test_async_completed_case_persists_before_slow_case_finishes():
    async def scenario():
        gate = asyncio.Event()
        fast_published = asyncio.Event()
        evaluator = AsyncScriptedEvaluator(gate=gate)
        writer = RecordingWriter(fast_published)
        framework = EvaluationFramework([evaluator])
        framework._writers = [writer]
        task = asyncio.create_task(
            framework.a_evaluate_many(
                [
                    EvaluationCase(case_id="slow"),
                    EvaluationCase(case_id="fast"),
                ],
                max_concurrency=2,
            )
        )
        await asyncio.wait_for(fast_published.wait(), timeout=1)
        assert evaluator.slow_finished is False
        assert writer.case_ids == ["fast"]
        gate.set()
        results = await task
        return results, writer

    results, writer = asyncio.run(scenario())
    assert [result["async_metric"].score for result in results] == [1.0, 1.0]
    assert writer.case_ids == ["fast", "slow"]
    assert writer.max_active == 1


def test_async_excel_checkpoints_completed_cases_before_slow_failure(tmp_path):
    class AsyncThreeCaseEvaluator(Evaluator):
        name = "quality"

        def __init__(self, gate):
            self.gate = gate
            self.case_2_finished = False

        def evaluate(self, case):
            raise AssertionError("sync path")

        async def a_evaluate(self, case, *, judge_limiter):
            async with judge_limiter:
                if case.case_id == "case_2":
                    await self.gate.wait()
                    self.case_2_finished = True
                    raise TimeoutError("provider timeout")
                if case.case_id == "case_1":
                    await asyncio.sleep(0.01)
                return EvaluationResult(
                    self.name, 1.0, "pass", f"completed {case.case_id}"
                )

    class CompletionWriter:
        def __init__(self, completed):
            self.completed = completed
            self.seen = []
            self.active = 0
            self.max_active = 0

        def write(self, records):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.seen.extend(record.case_id for record in records)
            if {"case_1", "case_3"}.issubset(self.seen):
                self.completed.set()
            self.active -= 1

    async def scenario(path):
        gate = asyncio.Event()
        completed = asyncio.Event()
        evaluator = AsyncThreeCaseEvaluator(gate)
        marker = CompletionWriter(completed)
        framework = EvaluationFramework(
            [evaluator],
            output="excel",
            excel_path=str(path),
            resume=True,
        )
        checkpoint = framework._checkpoint
        framework._writers = [checkpoint, marker]
        task = asyncio.create_task(
            framework.a_evaluate_many(
                [
                    EvaluationCase(case_id="case_1"),
                    EvaluationCase(case_id="case_2"),
                    EvaluationCase(case_id="case_3"),
                ],
                max_concurrency=3,
                on_error="continue",
            )
        )
        await asyncio.wait_for(completed.wait(), timeout=1)
        checkpoint_rows = _rows(path)
        assert {row["key_id"] for row in checkpoint_rows} == {
            "case_1",
            "case_3",
        }
        assert evaluator.case_2_finished is False
        gate.set()
        results = await task
        return results, marker

    path = tmp_path / "async-incremental.xlsx"
    results, marker = asyncio.run(scenario(path))
    assert [result["quality"].label for result in results] == [
        "pass",
        "error",
        "pass",
    ]
    assert marker.max_active == 1
    assert len(_rows(path)) == 3
    assert {row["key_id"]: row["status"] for row in _rows(path)} == {
        "case_1": "success",
        "case_2": "error",
        "case_3": "success",
    }


def test_resume_reuses_success_reruns_error_and_upserts(tmp_path):
    path = tmp_path / "resume.xlsx"
    cases = [EvaluationCase(case_id="a"), EvaluationCase(case_id="b")]
    first_coverage = ScriptedEvaluator("coverage", {"a": [0.8], "b": [0.9]})
    first_faithfulness = ScriptedEvaluator(
        "faithfulness", {"a": [TimeoutError("429")], "b": [1.0]}
    )
    EvaluationFramework(
        [first_coverage, first_faithfulness],
        output="excel",
        excel_path=str(path),
    ).evaluate_many(
        cases,
        run_name="run",
        dataset_name="data",
        on_error="continue",
    )

    second_coverage = ScriptedEvaluator("coverage", {})
    second_faithfulness = ScriptedEvaluator("faithfulness", {"a": [0.7]})
    results = EvaluationFramework(
        [second_coverage, second_faithfulness],
        output="excel",
        excel_path=str(path),
        resume=True,
    ).evaluate_many(
        cases,
        run_name="run",
        dataset_name="data",
        on_error="continue",
    )

    assert second_coverage.calls == []
    assert second_faithfulness.calls == ["a"]
    assert results[0]["coverage"].score == 0.8
    assert results[0]["coverage"].details == {"case": "a", "value": 0.8}
    assert results[0]["faithfulness"].score == 0.7
    assert results[1]["faithfulness"].score == 1.0
    rows = _rows(path)
    checkpoint_rows = _rows(path, "_idp_eval_checkpoint")
    assert len(rows) == 4
    assert all(row["status"] == "success" for row in rows)
    assert len(
        {
            (
                row["run_name"],
                row["dataset_name"],
                row["case_id"],
                row["evaluation_fingerprint"],
            )
            for row in checkpoint_rows
        }
    ) == 4


def test_process_style_partial_metric_resume_exactly_reruns_failed_metric(
    tmp_path,
):
    """Three cases prove per-metric resume survives framework destruction."""
    path = tmp_path / "partial-metric-resume.xlsx"
    cases = [
        EvaluationCase(
            case_id=f"case_{index}",
            context=f"source {index}",
            output=f"answer {index}",
        )
        for index in range(1, 4)
    ]
    first_coverage = QueuedJudge(
        *(_covered_response(f"source {index}") for index in range(1, 4))
    )
    first_faithfulness = QueuedJudge(
        _supported_response("answer 1"),
        TimeoutError("provider rate limit exhausted"),
        _supported_response("answer 3"),
    )
    first_framework = EvaluationFramework(
        [
            CoverageEvaluator(first_coverage, verbose=True),
            FaithfulnessEvaluator(first_faithfulness, verbose=True),
        ],
        output="excel",
        excel_path=str(path),
        resume=True,
    )
    first_results = first_framework.evaluate_many(
        cases,
        run_name="audit-run",
        dataset_name="audit-data",
        on_error="continue",
    )
    assert [result["coverage"].label for result in first_results] == [
        "covered",
        "covered",
        "covered",
    ]
    assert [result["faithfulness"].label for result in first_results] == [
        "not_hallucinated",
        "error",
        "not_hallucinated",
    ]
    first_rows = _rows(path)
    assert len(first_rows) == 6
    assert {
        (row["key_id"], row["metric"]): row["status"]
        for row in first_rows
    } == {
        ("case_1", "coverage"): "success",
        ("case_1", "faithfulness"): "success",
        ("case_2", "coverage"): "success",
        ("case_2", "faithfulness"): "error",
        ("case_3", "coverage"): "success",
        ("case_3", "faithfulness"): "success",
    }
    del first_framework

    resumed_coverage = QueuedJudge()
    resumed_faithfulness = QueuedJudge(_supported_response("answer 2"))
    resumed_framework = EvaluationFramework(
        [
            CoverageEvaluator(resumed_coverage, verbose=True),
            FaithfulnessEvaluator(resumed_faithfulness, verbose=True),
        ],
        output="excel",
        excel_path=str(path),
        resume=True,
    )
    resumed_results = resumed_framework.evaluate_many(
        cases,
        run_name="audit-run",
        dataset_name="audit-data",
        on_error="continue",
    )

    assert resumed_coverage.calls == []
    assert len(resumed_faithfulness.calls) == 1
    assert "answer 2" in str(resumed_faithfulness.calls[0]["prompt"])
    assert all(
        set(result) == {"coverage", "faithfulness"}
        for result in resumed_results
    )
    assert all(
        result[metric].label != "error"
        for result in resumed_results
        for metric in ("coverage", "faithfulness")
    )
    final_rows = _rows(path)
    assert len(final_rows) == 6
    assert all(row["status"] == "success" for row in final_rows)
    assert len(_rows(path, "coverage_items")) == 3
    assert len(_rows(path, "faithfulness_items")) == 3


def test_resume_reconstructs_successful_not_applicable(tmp_path):
    class NotApplicable(Evaluator):
        name = "optional"

        def __init__(self):
            self.calls = 0

        def evaluate(self, case):
            self.calls += 1
            return EvaluationResult(
                self.name,
                None,
                "not_applicable",
                "nothing to judge",
                {"judge_call_count": 0},
            )

    path = tmp_path / "resume-na.xlsx"
    first = NotApplicable()
    EvaluationFramework(
        [first], output="excel", excel_path=str(path)
    ).evaluate(EvaluationCase(case_id="na"))
    second = NotApplicable()
    result = EvaluationFramework(
        [second], output="excel", excel_path=str(path), resume=True
    ).evaluate(EvaluationCase(case_id="na"))["optional"]
    assert first.calls == 1
    assert second.calls == 0
    assert result.score is None
    assert result.label == "not_applicable"
    assert result.details == {"judge_call_count": 0}


def test_async_resume_reuses_success_and_reruns_error(tmp_path):
    class AsyncResumeMetric(Evaluator):
        def __init__(self, name, *, fail=False):
            self._name = name
            self.fail = fail
            self.calls = 0

        @property
        def name(self):
            return self._name

        def evaluate(self, case):
            raise AssertionError("sync path")

        async def a_evaluate(self, case, *, judge_limiter):
            async with judge_limiter:
                self.calls += 1
                if self.fail:
                    raise TimeoutError("throttled")
                return EvaluationResult(self.name, 1.0, "pass", "ok")

    path = tmp_path / "async-resume.xlsx"
    first_ok = AsyncResumeMetric("ok")
    first_failed = AsyncResumeMetric("retry", fail=True)
    asyncio.run(
        EvaluationFramework(
            [first_ok, first_failed],
            output="excel",
            excel_path=str(path),
        ).a_evaluate(
            EvaluationCase(case_id="a"), on_error="continue"
        )
    )

    resumed_ok = AsyncResumeMetric("ok")
    resumed_retry = AsyncResumeMetric("retry")
    result = asyncio.run(
        EvaluationFramework(
            [resumed_ok, resumed_retry],
            output="excel",
            excel_path=str(path),
            resume=True,
        ).a_evaluate(
            EvaluationCase(case_id="a"), on_error="continue"
        )
    )
    assert resumed_ok.calls == 0
    assert resumed_retry.calls == 1
    assert result["ok"].score == 1.0
    assert result["retry"].score == 1.0


def test_resume_verbose_detail_rows_are_replaced_not_duplicated(tmp_path):
    from idp_eval import CoverageEvaluator

    class Judge:
        def __init__(self, response):
            self.response = response
            self.calls = 0

        def generate_object(self, prompt, schema):
            self.calls += 1
            if isinstance(self.response, BaseException):
                raise self.response
            return self.response

    path = tmp_path / "details.xlsx"
    case = EvaluationCase(case_id="coverage", context="source", output="answer")
    failed = Judge(TimeoutError("throttled"))
    EvaluationFramework(
        [CoverageEvaluator(failed, verbose=True)],
        output="excel",
        excel_path=str(path),
    ).evaluate(case, on_error="continue")

    response = {
        "items": [
            {
                "source_item": "First",
                "meaningfully_present": True,
                "fully_present": True,
                "reason": "",
            },
            {
                "source_item": "Second",
                "meaningfully_present": False,
                "fully_present": False,
                "reason": "absent",
            },
        ],
        "overall_reason": "The second source item is absent.",
    }
    successful = Judge(response)
    EvaluationFramework(
        [CoverageEvaluator(successful, verbose=True)],
        output="excel",
        excel_path=str(path),
        resume=True,
    ).evaluate(case, on_error="continue")
    assert len(_rows(path)) == 1
    assert len(_rows(path, "coverage_items")) == 2

    skipped = Judge(AssertionError("must not run"))
    EvaluationFramework(
        [CoverageEvaluator(skipped, verbose=True)],
        output="excel",
        excel_path=str(path),
        resume=True,
    ).evaluate(case)
    assert skipped.calls == 0
    assert len(_rows(path, "coverage_items")) == 2


@pytest.mark.parametrize(
    "change", ["context", "output", "config", "run", "dataset"]
)
def test_resume_reruns_when_evaluation_identity_changes(tmp_path, change):
    path = tmp_path / f"{change}.xlsx"
    first_case = EvaluationCase(case_id="same", context="old")
    EvaluationFramework(
        [ScriptedEvaluator("quality", {})],
        output="excel",
        excel_path=str(path),
    ).evaluate(
        first_case, run_name="run", dataset_name="data"
    )

    if change == "context":
        case = EvaluationCase(case_id="same", context="new")
    elif change == "output":
        case = EvaluationCase(case_id="same", context="old", output="new")
    else:
        case = first_case
    evaluator = ScriptedEvaluator(
        "quality", {}, config="v2" if change == "config" else "v1"
    )
    EvaluationFramework(
        [evaluator],
        output="excel",
        excel_path=str(path),
        resume=True,
    ).evaluate(
        case,
        run_name="other" if change == "run" else "run",
        dataset_name="other" if change == "dataset" else "data",
    )
    assert evaluator.calls == ["same"]


def test_builtin_resume_fingerprint_changes_with_judge_model(tmp_path):
    from idp_eval import CoverageEvaluator

    class Judge:
        def __init__(self, model):
            self.model = model
            self.calls = 0

        def generate_object(self, prompt, schema):
            self.calls += 1
            return {
                "items": [
                    {
                        "source_item": "fact",
                        "meaningfully_present": True,
                        "fully_present": True,
                        "reason": "",
                    }
                ],
                "overall_reason": "The fact is represented.",
            }

    path = tmp_path / "model.xlsx"
    case = EvaluationCase(case_id="same", context="fact", output="fact")
    first = Judge("model-a")
    EvaluationFramework(
        [CoverageEvaluator(first)], output="excel", excel_path=str(path)
    ).evaluate(case)
    changed = Judge("model-b")
    EvaluationFramework(
        [CoverageEvaluator(changed)],
        output="excel",
        excel_path=str(path),
        resume=True,
    ).evaluate(case)
    assert changed.calls == 1


def test_run_and_dataset_names_isolate_checkpoint_experiments(tmp_path):
    path = tmp_path / "experiments.xlsx"
    case = EvaluationCase(case_id="same", context="same", output="same")
    run_names = (
        "generated_epic_vs_theme_text",
        "generated_epic_vs_context_at_5",
        "generated_epic_vs_theme_plus_context_at_5",
    )
    first = ScriptedEvaluator("quality", {})
    framework = EvaluationFramework(
        [first], output="excel", excel_path=str(path), resume=True
    )
    for run_name in run_names:
        framework.evaluate(case, run_name=run_name, dataset_name="dataset-a")
    framework.evaluate(
        case, run_name=run_names[0], dataset_name="dataset-b"
    )
    assert first.calls == ["same", "same", "same", "same"]

    resumed = ScriptedEvaluator("quality", {})
    new_framework = EvaluationFramework(
        [resumed], output="excel", excel_path=str(path), resume=True
    )
    for run_name in run_names:
        new_framework.evaluate(
            case, run_name=run_name, dataset_name="dataset-a"
        )
    new_framework.evaluate(
        case, run_name=run_names[0], dataset_name="dataset-b"
    )
    assert resumed.calls == []
    assert len(_rows(path)) == 4


def test_resume_rejects_legacy_workbook_and_non_excel_mode(tmp_path):
    legacy = tmp_path / "legacy.xlsx"
    workbook = openpyxl.Workbook()
    workbook.active.title = "evaluations"
    workbook.active.append(["case_id", "metric", "score"])
    workbook.save(legacy)

    with pytest.raises(ValueError, match="predates|resumable"):
        EvaluationFramework(
            [], output="excel", excel_path=str(legacy), resume=True
        )
    with pytest.raises(ValueError, match="requires output='excel' or output='both'"):
        EvaluationFramework([], output="phoenix", resume=True)


@pytest.mark.parametrize(
    "column,value,match",
    [
        ("status", "pending", "status must be"),
        ("evaluation_fingerprint", None, "missing a required fingerprint"),
        ("case_fingerprint", None, "missing a required fingerprint"),
    ],
)
def test_resume_rejects_corrupt_checkpoint_identity_before_judge(
    tmp_path, column, value, match
):
    path = tmp_path / f"corrupt-{column}.xlsx"
    EvaluationFramework(
        [ScriptedEvaluator("quality", {})],
        output="excel",
        excel_path=str(path),
    ).evaluate(EvaluationCase(case_id="a"))
    workbook = openpyxl.load_workbook(path)
    sheet = workbook["_idp_eval_checkpoint"]
    headers = {cell.value: index for index, cell in enumerate(sheet[1], start=1)}
    sheet.cell(row=2, column=headers[column]).value = value
    workbook.save(path)

    judge = ScriptedEvaluator("quality", {})
    with pytest.raises(ValueError, match=match):
        EvaluationFramework(
            [judge], output="excel", excel_path=str(path), resume=True
        )
    assert judge.calls == []


def test_resume_rejects_corrupt_result_json_before_judge(tmp_path):
    path = tmp_path / "corrupt-details.xlsx"
    EvaluationFramework(
        [ScriptedEvaluator("quality", {})],
        output="excel",
        excel_path=str(path),
    ).evaluate(EvaluationCase(case_id="a"))
    workbook = openpyxl.load_workbook(path)
    sheet = workbook["_idp_eval_checkpoint"]
    headers = {cell.value: index for index, cell in enumerate(sheet[1], start=1)}
    sheet.cell(row=2, column=headers["raw_details_json"]).value = "{invalid"
    workbook.save(path)

    evaluator = ScriptedEvaluator("quality", {})
    framework = EvaluationFramework(
        [evaluator], output="excel", excel_path=str(path), resume=True
    )
    with pytest.raises(ValueError, match="invalid raw_details_json"):
        framework.evaluate(EvaluationCase(case_id="a"))
    assert evaluator.calls == []


def test_output_both_uses_excel_as_checkpoint(tmp_path):
    path = tmp_path / "both.xlsx"
    evaluator = ScriptedEvaluator("quality", {})
    framework = EvaluationFramework(
        [evaluator], output="both", excel_path=str(path), resume=True
    )
    checkpoint = framework._checkpoint
    phoenix = RecordingWriter()
    framework._writers = [phoenix, checkpoint]
    framework.evaluate(EvaluationCase(case_id="a"))
    assert evaluator.calls == ["a"]
    assert phoenix.case_ids == ["a"]

    resumed = ScriptedEvaluator("quality", {})
    second = EvaluationFramework(
        [resumed], output="both", excel_path=str(path), resume=True
    )
    second._writers = [RecordingWriter(), second._checkpoint]
    assert second.evaluate(EvaluationCase(case_id="a"))["quality"].score == 1.0
    assert resumed.calls == []
    assert second._writers[0].case_ids == []


def test_output_both_resume_skips_traces_and_republishing(monkeypatch, tmp_path):
    from idp_eval import framework as framework_module

    root_spans = []

    @contextmanager
    def fake_case_span(case_id, *args, **kwargs):
        root_spans.append(case_id)
        index = len(root_spans)
        yield SimpleNamespace(
            trace_id=f"{index:032x}", span_id=f"{index:016x}"
        )

    monkeypatch.setattr(
        framework_module.tracing, "case_evaluation_span", fake_case_span
    )
    path = tmp_path / "phoenix-resume.xlsx"
    case = EvaluationCase(case_id="case")

    failed = ScriptedEvaluator("quality", {"case": [TimeoutError("throttled")]})
    first = EvaluationFramework(
        [failed], output="both", excel_path=str(path), resume=True
    )
    first_phoenix = RecordingWriter()
    first._writers = [first_phoenix, first._checkpoint]
    first.evaluate(case, on_error="continue")
    assert root_spans == ["case"]
    assert first_phoenix.case_ids == ["case"]
    assert _rows(path)[0]["status"] == "error"

    successful = ScriptedEvaluator("quality", {"case": [1.0]})
    second = EvaluationFramework(
        [successful], output="both", excel_path=str(path), resume=True
    )
    second_phoenix = RecordingWriter()
    second._writers = [second_phoenix, second._checkpoint]
    second.evaluate(case, on_error="continue")
    assert successful.calls == ["case"]
    assert root_spans == ["case", "case"]
    assert second_phoenix.case_ids == ["case"]
    assert len(_rows(path)) == 1
    assert _rows(path)[0]["status"] == "success"

    skipped = ScriptedEvaluator("quality", {})
    third = EvaluationFramework(
        [skipped], output="both", excel_path=str(path), resume=True
    )
    third_phoenix = RecordingWriter()
    third._writers = [third_phoenix, third._checkpoint]
    result = third.evaluate(case)
    assert result["quality"].score == 1.0
    assert skipped.calls == []
    assert root_spans == ["case", "case"]
    assert third_phoenix.case_ids == []
    assert len(_rows(path)) == 1


def test_progress_distinguishes_operational_error_and_resumed_case(
    tmp_path, capsys
):
    error_framework = EvaluationFramework(
        [ScriptedEvaluator("quality", {"error": [TimeoutError("throttled")]})]
    )
    error_framework.evaluate_many(
        [EvaluationCase(case_id="error")],
        on_error="continue",
        show_progress=True,
    )
    error_output = capsys.readouterr()
    error_text = error_output.out + error_output.err
    assert "completed with operational error" in error_text
    assert "quality=ERROR (TimeoutError)" in error_text

    path = tmp_path / "progress.xlsx"
    EvaluationFramework(
        [ScriptedEvaluator("quality", {})],
        output="excel",
        excel_path=str(path),
    ).evaluate(EvaluationCase(case_id="resumed"))
    EvaluationFramework(
        [ScriptedEvaluator("quality", {})],
        output="excel",
        excel_path=str(path),
        resume=True,
    ).evaluate_many(
        [EvaluationCase(case_id="resumed")], show_progress=True
    )
    resumed_output = capsys.readouterr()
    resumed_text = resumed_output.out + resumed_output.err
    assert "resumed from checkpoint" in resumed_text
    assert "resumed=1" in resumed_text


def test_error_details_are_sanitized_in_excel(tmp_path):
    path = tmp_path / "safe.xlsx"
    evaluator = ScriptedEvaluator(
        "quality", {"a": [TimeoutError("Bearer SECRET_TOKEN password=oops")]}
    )
    EvaluationFramework(
        [evaluator], output="excel", excel_path=str(path)
    ).evaluate(EvaluationCase(case_id="a"), on_error="continue")
    row = _rows(path)[0]
    payload = json.dumps(row)
    assert "SECRET_TOKEN" not in payload
    assert "password=oops" not in payload


def test_invalid_on_error_is_rejected_before_work():
    evaluator = ScriptedEvaluator("quality", {})
    with pytest.raises(ValueError, match="Unknown on_error"):
        EvaluationFramework([evaluator]).evaluate(
            EvaluationCase(case_id="a"), on_error="ignore"
        )
    assert evaluator.calls == []


def test_group_helpers_propagate_continue_mode():
    sync = ScriptedEvaluator("quality", {"g:0": [TimeoutError("throttled")]})
    result = EvaluationFramework([sync]).evaluate_groups(
        [{"group_id": "g", "outputs": ["answer"]}],
        on_error="continue",
    )
    assert result[0]["quality"].label == "error"

    class AsyncFailure(Evaluator):
        name = "async_quality"

        def evaluate(self, case):
            raise AssertionError("sync path")

        async def a_evaluate(self, case, *, judge_limiter):
            async with judge_limiter:
                raise TimeoutError("throttled")

    async_result = asyncio.run(
        EvaluationFramework([AsyncFailure()]).a_evaluate_groups(
            [{"group_id": "g", "outputs": ["answer"]}],
            on_error="continue",
        )
    )
    assert async_result[0]["async_quality"].label == "error"


def test_case_fingerprint_is_canonical_and_preserves_rank_order():
    first = EvaluationCase(
        case_id="one",
        context={"b": 2, "a": 1},
        retrieved_documents=[{"text": "first"}, {"text": "second"}],
    )
    same = EvaluationCase(
        case_id="different-id",
        context={"a": 1, "b": 2},
        retrieved_documents=[{"text": "first"}, {"text": "second"}],
    )
    reordered = EvaluationCase(
        context={"a": 1, "b": 2},
        retrieved_documents=[{"text": "second"}, {"text": "first"}],
    )
    assert case_fingerprint(first) == case_fingerprint(same)
    assert case_fingerprint(first) != case_fingerprint(reordered)


def test_builtin_evaluation_fingerprint_tracks_retrieval_configuration():
    class Judge:
        model = "offline-model"

    case_hash = case_fingerprint(
        EvaluationCase(
            input="query",
            retrieved_documents=[{"body": "one", "text": "one"}],
        )
    )
    k5 = RelevanceAtKEvaluator(5, Judge())
    k10 = RelevanceAtKEvaluator(10, Judge())
    body_key = RelevanceAtKEvaluator(5, Judge(), document_text_key="body")
    verbose = RelevanceAtKEvaluator(5, Judge(), verbose=True)
    fingerprints = {
        evaluation_fingerprint(case_hash, evaluator.name, evaluator)
        for evaluator in (k5, k10, body_key, verbose)
    }
    # Verbose uses a different prompt/schema/result-detail contract, so it must
    # not reuse a compact checkpoint even though both make one judge call.
    assert len(fingerprints) == 4


def test_core_evaluator_resume_signatures_use_contract_version_four():
    assert CoverageEvaluator().resume_signature()["contract_version"] == 4
    assert FaithfulnessEvaluator().resume_signature()["contract_version"] == 4


@pytest.mark.parametrize(
    "evaluator_type", [CoverageEvaluator, FaithfulnessEvaluator]
)
def test_v4_semantic_contract_changes_fingerprint_from_v3(evaluator_type):
    class PreviousV3Contract(evaluator_type):
        def resume_signature(self):
            signature = super().resume_signature()
            signature["contract_version"] = 3
            return signature

    class Judge:
        model = "same-offline-model"

    case_hash = case_fingerprint(
        EvaluationCase(case_id="same", context="fact", output="fact")
    )
    current = evaluator_type(
        Judge(), verbose=True, max_items=5, reason_mode="overall"
    )
    previous = PreviousV3Contract(
        Judge(), verbose=True, max_items=5, reason_mode="overall"
    )

    assert current.resume_signature() | {"contract_version": 3} == (
        previous.resume_signature()
    )
    assert evaluation_fingerprint(
        case_hash, current.name, current
    ) != evaluation_fingerprint(case_hash, previous.name, previous)


@pytest.mark.parametrize(
    "evaluator_type,limit_name,response",
    [
        (
            CoverageEvaluator,
            "max_items",
            {
                "items": [
                    {
                        "source_item": "fact",
                        "meaningfully_present": True,
                        "fully_present": True,
                        "reason": "",
                    }
                ],
                "overall_reason": "The fact is represented.",
            },
        ),
        (
            FaithfulnessEvaluator,
            "max_items",
            {
                "claims": [
                    {"claim": "fact", "status": "supported", "reason": ""}
                ],
                "overall_reason": "The fact is grounded in context.",
            },
        ),
    ],
)
def test_extraction_limits_change_resume_identity_and_exact_limit_resumes(
    tmp_path, evaluator_type, limit_name, response
):
    path = tmp_path / f"{limit_name}.xlsx"
    case = EvaluationCase(case_id="same", context="fact", output="fact")

    first_judge = QueuedJudge(response)
    EvaluationFramework(
        [evaluator_type(first_judge, **{limit_name: 5})],
        output="excel",
        excel_path=str(path),
        resume=True,
    ).evaluate(case)
    assert len(first_judge.calls) == 1

    same_judge = QueuedJudge(response)
    EvaluationFramework(
        [evaluator_type(same_judge, **{limit_name: 5})],
        output="excel",
        excel_path=str(path),
        resume=True,
    ).evaluate(case)
    assert same_judge.calls == []

    ten_judge = QueuedJudge(response)
    EvaluationFramework(
        [evaluator_type(ten_judge, **{limit_name: 10})],
        output="excel",
        excel_path=str(path),
        resume=True,
    ).evaluate(case)
    assert len(ten_judge.calls) == 1

    unlimited_judge = QueuedJudge(response)
    EvaluationFramework(
        [evaluator_type(unlimited_judge, **{limit_name: None})],
        output="excel",
        excel_path=str(path),
        resume=True,
    ).evaluate(case)
    assert len(unlimited_judge.calls) == 1


@pytest.mark.parametrize(
    "evaluator_type,response",
    [
        (CoverageEvaluator, _covered_response("fact")),
        (FaithfulnessEvaluator, _supported_response("fact")),
    ],
)
def test_reason_mode_changes_resume_identity_and_exact_mode_resumes(
    tmp_path, evaluator_type, response
):
    path = tmp_path / f"{evaluator_type.name}-reason-mode.xlsx"
    case = EvaluationCase(case_id="same", context="fact", output="fact")

    first = QueuedJudge(response)
    EvaluationFramework(
        [evaluator_type(first, reason_mode="overall")],
        output="excel",
        excel_path=str(path),
        resume=True,
    ).evaluate(case)
    assert len(first.calls) == 1

    same = QueuedJudge(response)
    EvaluationFramework(
        [evaluator_type(same, reason_mode="overall")],
        output="excel",
        excel_path=str(path),
        resume=True,
    ).evaluate(case)
    assert same.calls == []

    none_response = (
        {
            "items": [
                {
                    "source_item": "fact",
                    "meaningfully_present": True,
                    "fully_present": True,
                }
            ]
        }
        if evaluator_type is CoverageEvaluator
        else {"claims": [{"claim": "fact", "status": "supported"}]}
    )
    changed = QueuedJudge(none_response)
    EvaluationFramework(
        [evaluator_type(changed, reason_mode="none")],
        output="excel",
        excel_path=str(path),
        resume=True,
    ).evaluate(case)
    assert len(changed.calls) == 1


def test_report_fields_do_not_participate_in_evaluation_fingerprint():
    judge = QueuedJudge()
    evaluator = CoverageEvaluator(judge, max_items=5)
    case_hash = case_fingerprint(
        EvaluationCase(case_id="same", context="fact", output="fact")
    )
    before = evaluation_fingerprint(case_hash, evaluator.name, evaluator)
    EvaluationFramework(
        [evaluator], report_fields=["context", "output"]
    )
    after = evaluation_fingerprint(case_hash, evaluator.name, evaluator)
    assert before == after
