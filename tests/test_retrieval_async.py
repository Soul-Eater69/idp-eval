"""Async retrieval: concurrent document judgments, bounding, ordering, failures."""

import asyncio
import threading
import time

import pytest

import idp_eval.evaluators.retrieval as retrieval_mod
from idp_eval import (
    EvaluationCase,
    EvaluationFramework,
    NDCGAtKEvaluator,
    RelevanceAtKEvaluator,
)


class FakeScore:
    def __init__(self, score, label):
        self.score = score
        self.label = label
        self.explanation = "because"


class ProbeHarness:
    """Concurrency-tracking fake relevance evaluator with a small blocking sleep."""

    def __init__(self, *, delay=0.05, raise_on=None):
        self.delay = delay
        self.raise_on = raise_on
        self._lock = threading.Lock()
        self._current = 0
        self.max_concurrent = 0
        self.calls = 0

    def factory(self, llm):
        harness = self

        class _FakeEvaluator:
            def evaluate(self, record):
                with harness._lock:
                    harness.calls += 1
                    harness._current += 1
                    harness.max_concurrent = max(
                        harness.max_concurrent, harness._current
                    )
                try:
                    if harness.raise_on and harness.raise_on in record["document_text"]:
                        raise RuntimeError("gateway timeout")
                    time.sleep(harness.delay)
                    rel = 1.0 if "GOOD" in record["document_text"] else 0.0
                    return [FakeScore(rel, "relevant" if rel else "unrelated")]
                finally:
                    with harness._lock:
                        harness._current -= 1

        return _FakeEvaluator()


@pytest.fixture
def probe(monkeypatch):
    h = ProbeHarness()
    monkeypatch.setattr(
        retrieval_mod, "_build_document_relevance_evaluator", h.factory
    )
    return h


def _docs(n, pattern="GOOD"):
    return [{"text": f"{pattern} {i}"} for i in range(n)]


def _framework(**kwargs):
    return EvaluationFramework(
        evaluators=[RelevanceAtKEvaluator(k=10)], judge=object(), **kwargs
    )


def test_a_evaluate_runs(probe):
    case = EvaluationCase(input="q", retrieved_documents=_docs(3))
    results = asyncio.run(_framework().a_evaluate(case))
    assert results["relevance_at_10"].score == 1.0


def test_documents_overlap_but_respect_max_concurrency(probe):
    case = EvaluationCase(input="q", retrieved_documents=_docs(6))
    asyncio.run(_framework().a_evaluate(case, max_concurrency=2))
    assert probe.max_concurrent >= 2       # documents judged concurrently
    assert probe.max_concurrent <= 2       # bounded by the global cap


def test_global_cap_shared_across_metrics(probe):
    # Two retrieval metrics still share ONE relevance pass -> 5 judgments, cap 2.
    case = EvaluationCase(input="q", retrieved_documents=_docs(5))
    framework = EvaluationFramework(
        evaluators=[RelevanceAtKEvaluator(k=5), NDCGAtKEvaluator(k=5)],
        judge=object(),
    )
    asyncio.run(framework.a_evaluate(case, max_concurrency=2))
    assert probe.calls == 5                # not 10
    assert probe.max_concurrent <= 2


def test_result_ordering_follows_document_rank(probe):
    # Relevance by position: docs 0 and 2 relevant, 1 and 3 not.
    docs = [
        {"text": "GOOD 0"}, {"text": "bad 1"}, {"text": "GOOD 2"}, {"text": "bad 3"},
    ]
    case = EvaluationCase(input="q", retrieved_documents=docs)
    results = asyncio.run(_framework().a_evaluate(case, max_concurrency=4))
    scores = [d["relevance_score"] for d in results["relevance_at_10"].details["documents"]]
    assert scores == [1.0, 0.0, 1.0, 0.0]   # rank order preserved despite overlap


def test_evaluator_exception_propagates(probe):
    probe.raise_on = "bad"
    case = EvaluationCase(input="q", retrieved_documents=[{"text": "bad 0"}])
    with pytest.raises(RuntimeError, match="gateway timeout"):
        asyncio.run(_framework().a_evaluate(case))


def test_error_not_converted_to_zero_or_not_applicable(probe):
    # A failed judgment must raise, never become a silent 0.0 / not_applicable.
    probe.raise_on = "GOOD"
    case = EvaluationCase(input="q", retrieved_documents=_docs(3))
    with pytest.raises(RuntimeError):
        asyncio.run(_framework().a_evaluate(case))
    # Sync path too.
    with pytest.raises(RuntimeError):
        _framework().evaluate(case)


def test_a_evaluate_many_separate_traces(probe):
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

    cases = [
        EvaluationCase(input="q", retrieved_documents=_docs(2), case_id=f"c{i}")
        for i in range(3)
    ]
    results = asyncio.run(_framework().a_evaluate_many(cases))
    assert len(results) == 3
    roots = [s for s in exporter.get_finished_spans() if s.name == "idp_eval.evaluate"]
    assert len(roots) == 3
    assert len({s.context.trace_id for s in roots}) == 3
    exporter.clear()
