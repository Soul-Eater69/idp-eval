"""Async coverage evaluation: concurrency bounding, ordering, and tracing.

Offline: a scripted judge with a small blocking sleep (run in worker threads via
the async path) lets us observe overlap and the global concurrency cap. No LLM,
Phoenix, or gateway calls.
"""

import asyncio
import json
import threading
import time

import pytest

from idp_eval import (
    EvaluationCase,
    EvaluationFramework,
    SourceCoverageEvaluator,
)


def _requirements_from_prompt(user_content: str) -> list[dict]:
    block = user_content.split("[REQUIREMENTS]\n", 1)[1].split("\n\n[OUTPUT]", 1)[0]
    return json.loads(block)


class ProbeJudge:
    """Sleeps briefly per call and tracks peak concurrency; output-sensitive.

    Classification marks every requested id covered iff the rendered OUTPUT
    contains ``covered_token`` — so different cases get different scores, which
    lets us assert result ordering.
    """

    def __init__(self, items, *, delay=0.05, covered_token="GOOD"):
        self.items = list(items)
        self.delay = delay
        self.covered_token = covered_token
        self._lock = threading.Lock()
        self._current = 0
        self._classify_current = 0
        self.max_concurrent = 0
        self.classify_max_concurrent = 0
        self.order: list[str] = []

    def generate_object(self, prompt, schema):
        user = prompt[1]["content"]
        is_classify = "[REQUIREMENTS]" in user
        with self._lock:
            self._current += 1
            self.max_concurrent = max(self.max_concurrent, self._current)
            if is_classify:
                self._classify_current += 1
                self.classify_max_concurrent = max(
                    self.classify_max_concurrent, self._classify_current
                )
            self.order.append("classify" if is_classify else "extract")
        try:
            time.sleep(self.delay)
            if not is_classify:
                if "source_items" in schema["properties"]:
                    return {"source_items": [{"source_item": t} for t in self.items]}
                return {"requirements": [{"requirement": t} for t in self.items]}
            covered = self.covered_token in user.split("[OUTPUT]\n", 1)[1]
            return {
                "requirements": [
                    {
                        "id": r["id"],
                        "meaningfully_present": covered,
                        "fully_present": covered,
                    }
                    for r in _requirements_from_prompt(user)
                ]
            }
        finally:
            with self._lock:
                self._current -= 1
                if is_classify:
                    self._classify_current -= 1


def _items(n):
    return [f"item number {i}" for i in range(1, n + 1)]


def _framework(judge, **kwargs):
    return EvaluationFramework(
        evaluators=[SourceCoverageEvaluator(judge, **kwargs)]
    )


def _case(output, case_id):
    return EvaluationCase(context="src", output=output, case_id=case_id)


# --- basic async behavior ---------------------------------------------------


def test_a_evaluate_runs_one_case():
    judge = ProbeJudge(_items(3), delay=0.0)
    results = asyncio.run(_framework(judge).a_evaluate(_case("GOOD", "c1")))
    assert results["source_coverage"].score == 1.0


def test_sync_still_works():
    judge = ProbeJudge(_items(3), delay=0.0)
    results = _framework(judge).evaluate(_case("GOOD", "c1"))
    assert results["source_coverage"].score == 1.0


def test_a_evaluate_many_preserves_order_and_scores():
    judge = ProbeJudge(_items(2), delay=0.0)
    cases = [_case("GOOD", "c1"), _case("BAD", "c2"), _case("GOOD", "c3")]
    results = asyncio.run(_framework(judge).a_evaluate_many(cases))
    assert [r["source_coverage"].score for r in results] == [1.0, 0.0, 1.0]


# --- concurrency bounding ---------------------------------------------------


def test_cases_overlap_but_respect_max_concurrency():
    judge = ProbeJudge(_items(1), delay=0.05)
    cases = [_case("GOOD", f"c{i}") for i in range(6)]
    asyncio.run(_framework(judge).a_evaluate_many(cases, max_concurrency=2))
    assert judge.max_concurrent >= 2       # real overlap
    assert judge.max_concurrent <= 2       # never exceeds the cap


def test_global_cap_applies_across_classification_batches():
    # One case, 30 items, batch_size 10 -> 3 classify batches; cap 2.
    judge = ProbeJudge(_items(30), delay=0.05)
    framework = _framework(judge, classification_batch_size=10)
    asyncio.run(framework.a_evaluate(_case("GOOD", "c1"), max_concurrency=2))
    assert judge.classify_max_concurrent >= 2   # batches overlap
    assert judge.classify_max_concurrent <= 2   # bounded by the global cap


def test_independent_batches_overlap_under_higher_cap():
    judge = ProbeJudge(_items(20), delay=0.05)
    framework = _framework(judge, classification_batch_size=5)  # 4 batches
    asyncio.run(framework.a_evaluate(_case("GOOD", "c1"), max_concurrency=4))
    assert judge.classify_max_concurrent >= 2


def test_stage2_waits_for_stage1_within_a_case():
    judge = ProbeJudge(_items(6), delay=0.02)
    framework = _framework(judge, classification_batch_size=2)  # 3 batches
    asyncio.run(framework.a_evaluate(_case("GOOD", "c1"), max_concurrency=4))
    # Extraction happens once, first, before any classification for the case.
    assert judge.order[0] == "extract"
    assert judge.order.count("extract") == 1
    assert all(kind == "classify" for kind in judge.order[1:])


# --- validation & failure semantics -----------------------------------------


@pytest.mark.parametrize("bad", [0, -1, 1.5, "4", True, None])
def test_invalid_max_concurrency_rejected(bad):
    judge = ProbeJudge(_items(1), delay=0.0)
    framework = _framework(judge)
    with pytest.raises(ValueError, match="max_concurrency must be a positive"):
        asyncio.run(framework.a_evaluate(_case("GOOD", "c1"), max_concurrency=bad))
    with pytest.raises(ValueError, match="max_concurrency must be a positive"):
        asyncio.run(
            framework.a_evaluate_many([_case("GOOD", "c1")], max_concurrency=bad)
        )


class _GatewayTimeout(RuntimeError):
    """Stands in for an operational gateway timeout/error (not semantic N/A)."""


class ExplodingJudge:
    def __init__(self, *, on="classify"):
        self.on = on

    def generate_object(self, prompt, schema):
        is_classify = "[REQUIREMENTS]" in prompt[1]["content"]
        if (self.on == "classify") == is_classify:
            raise _GatewayTimeout("gateway timeout after 60s")
        return {"source_items": [{"source_item": "A"}]}


def test_async_evaluator_exception_propagates():
    framework = _framework(ExplodingJudge(on="classify"))
    with pytest.raises(_GatewayTimeout):
        asyncio.run(framework.a_evaluate(_case("GOOD", "c1")))


def test_async_evaluate_many_fails_whole_operation_on_one_case():
    framework = _framework(ExplodingJudge(on="extract"))
    cases = [_case("GOOD", "c1"), _case("GOOD", "c2")]
    with pytest.raises(_GatewayTimeout):
        asyncio.run(framework.a_evaluate_many(cases))


def test_timeout_error_is_not_converted_to_not_applicable():
    # Sync path too: an operational error is raised, never turned into a result.
    framework = _framework(ExplodingJudge(on="classify"))
    with pytest.raises(_GatewayTimeout):
        framework.evaluate(_case("GOOD", "c1"))


# --- tracing under concurrency ----------------------------------------------


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


def test_concurrent_cases_keep_separate_traces(spans):
    judge = ProbeJudge(_items(2), delay=0.02)
    cases = [_case("GOOD", f"c{i}") for i in range(3)]
    asyncio.run(_framework(judge).a_evaluate_many(cases))

    finished = spans.get_finished_spans()
    roots = [s for s in finished if s.name == "idp_eval.evaluate"]
    assert len(roots) == 3
    assert len({s.context.trace_id for s in roots}) == 3
    # Each case has its own extract + classify child spans under its trace.
    assert len([s for s in finished if s.name == "source_coverage.extract"]) == 3
    assert len([s for s in finished if s.name == "source_coverage.classify"]) == 3


def test_batched_classify_spans_have_no_duplicate_stage_spans(spans):
    judge = ProbeJudge(_items(16), delay=0.0)
    framework = _framework(judge, classification_batch_size=15)  # 2 batches
    asyncio.run(framework.a_evaluate(_case("GOOD", "c1"), max_concurrency=4))

    finished = spans.get_finished_spans()
    names = [s.name for s in finished]
    assert names.count("idp_eval.evaluate") == 1
    assert names.count("source_coverage.extract") == 1
    assert names.count("source_coverage.classify") == 1          # one grouping span
    assert names.count("source_coverage.classify.batch") == 2    # two batch spans
    # Batch spans carry batch_index / batch_count attributes.
    batch_spans = [s for s in finished if s.name == "source_coverage.classify.batch"]
    assert {s.attributes["coverage.batch_index"] for s in batch_spans} == {0, 1}
    assert all(s.attributes["coverage.batch_count"] == 2 for s in batch_spans)


def test_root_span_records_coverage_summary_attributes(spans):
    judge = ProbeJudge(_items(3), delay=0.0)
    asyncio.run(_framework(judge).a_evaluate(_case("GOOD", "c1")))
    root = next(
        s for s in spans.get_finished_spans() if s.name == "idp_eval.evaluate"
    )
    assert root.attributes["coverage.item_count"] == 3
    assert root.attributes["coverage.batch_count"] == 1
    assert root.attributes["coverage.verbose"] is False
    assert "coverage.total_ms" in root.attributes
