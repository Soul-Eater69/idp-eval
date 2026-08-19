"""Async one-call source coverage behavior (offline)."""

import asyncio
import threading
import time

import pytest

from idp_eval import EvaluationCase, EvaluationFramework, CoverageEvaluator


class ProbeJudge:
    """Tracks concurrency and returns output-sensitive one-call judgments."""

    def __init__(self, *, delay=0.05, explode_on: str | None = None):
        self.delay = delay
        self.explode_on = explode_on
        self._lock = threading.Lock()
        self._current = 0
        self.max_concurrent = 0
        self.calls = 0

    def generate_object(self, prompt, schema):
        user = prompt[1]["content"]
        with self._lock:
            self._current += 1
            self.max_concurrent = max(self.max_concurrent, self._current)
            self.calls += 1
        try:
            time.sleep(self.delay)
            if self.explode_on and self.explode_on in user:
                raise RuntimeError("gateway timeout")
            covered = "GOOD" in user.split("[OUTPUT]\n", 1)[1]
            return {
                "items": [
                    {
                        "source_item": "Important source item.",
                        "meaningfully_present": covered,
                        "fully_present": covered,
                    }
                ]
            }
        finally:
            with self._lock:
                self._current -= 1


def _framework(judge):
    return EvaluationFramework(evaluators=[CoverageEvaluator(judge, mode="g_eval")])


def _case(output, case_id):
    return EvaluationCase(context="source", output=output, case_id=case_id)


def test_sync_and_async_each_use_one_call():
    sync_judge = ProbeJudge(delay=0)
    async_judge = ProbeJudge(delay=0)
    sync = _framework(sync_judge).evaluate(_case("GOOD", "s"))
    async_result = asyncio.run(
        _framework(async_judge).a_evaluate(_case("GOOD", "a"))
    )
    assert sync["coverage"].score == 1.0
    assert async_result["coverage"].score == 1.0
    assert sync_judge.calls == async_judge.calls == 1


def test_a_evaluate_many_preserves_order_and_scores():
    judge = ProbeJudge(delay=0)
    cases = [_case("GOOD", "c1"), _case("BAD", "c2"), _case("GOOD", "c3")]
    results = asyncio.run(_framework(judge).a_evaluate_many(cases))
    assert [result["coverage"].score for result in results] == [1.0, 0.0, 1.0]
    assert judge.calls == 3


def test_cases_overlap_but_respect_global_max_concurrency():
    judge = ProbeJudge(delay=0.05)
    cases = [_case("GOOD", f"c{i}") for i in range(6)]
    asyncio.run(
        _framework(judge).a_evaluate_many(cases, max_concurrency=2)
    )
    assert judge.max_concurrent == 2
    assert judge.calls == 6


@pytest.mark.parametrize("bad", [0, -1, 1.5, "4", True, None])
def test_invalid_max_concurrency_rejected(bad):
    framework = _framework(ProbeJudge(delay=0))
    with pytest.raises(ValueError, match="max_concurrency must be a positive"):
        asyncio.run(framework.a_evaluate(_case("GOOD", "c1"), max_concurrency=bad))


def test_async_exception_propagates_and_is_not_not_applicable():
    framework = _framework(ProbeJudge(delay=0, explode_on="BOOM"))
    with pytest.raises(RuntimeError, match="gateway timeout"):
        asyncio.run(framework.a_evaluate(_case("BOOM", "c1")))


def test_async_many_exception_propagates():
    framework = _framework(ProbeJudge(delay=0, explode_on="BOOM"))
    with pytest.raises(RuntimeError, match="gateway timeout"):
        asyncio.run(
            framework.a_evaluate_many(
                [_case("GOOD", "c1"), _case("BOOM", "c2")]
            )
        )


@pytest.fixture
def spans():
    trace = pytest.importorskip("opentelemetry").trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace._TRACER_PROVIDER = provider  # type: ignore[attr-defined]
    yield exporter
    exporter.clear()


def test_concurrent_cases_keep_separate_traces_and_one_stage_span(spans):
    judge = ProbeJudge(delay=0.02)
    cases = [_case("GOOD", f"c{i}") for i in range(3)]
    asyncio.run(_framework(judge).a_evaluate_many(cases, max_concurrency=2))

    finished = spans.get_finished_spans()
    roots = [span for span in finished if span.name == "idp_eval.evaluate"]
    stages = [span for span in finished if span.name == "coverage.evaluate"]
    assert len(roots) == len(stages) == 3
    assert len({span.context.trace_id for span in roots}) == 3
    assert not any(
        span.name in {"coverage.extract", "coverage.classify"}
        for span in finished
    )


def test_root_span_records_one_call_source_summary(spans):
    asyncio.run(_framework(ProbeJudge(delay=0)).a_evaluate(_case("GOOD", "c1")))
    root = next(
        span for span in spans.get_finished_spans() if span.name == "idp_eval.evaluate"
    )
    assert root.attributes["coverage.item_count"] == 1
    assert root.attributes["coverage.judge_call_count"] == 1
    assert root.attributes["coverage.verbose"] is False
    assert "coverage.total_ms" in root.attributes
