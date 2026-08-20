"""Async one-call retrieval relevance, concurrency, ordering, and failures."""

import asyncio
import re

import pytest

from idp_eval import (
    EvaluationCase,
    EvaluationFramework,
    HitRateAtKEvaluator,
    MRRAtKEvaluator,
    NDCGAtKEvaluator,
    RelevanceAtKEvaluator,
)


class AsyncProbeJudge:
    def __init__(self, *, delay=0.02, explode_on=None):
        self.delay = delay
        self.explode_on = explode_on
        self.calls = 0
        self.sync_calls = 0
        self.current = 0
        self.max_concurrent = 0

    def generate_object(self, prompt, schema):
        self.sync_calls += 1
        raise AssertionError("native async path should be used")

    async def async_generate_object(self, prompt, schema):
        self.calls += 1
        self.current += 1
        self.max_concurrent = max(self.max_concurrent, self.current)
        try:
            await asyncio.sleep(self.delay)
            user = prompt[1]["content"]
            if self.explode_on and self.explode_on in user:
                raise RuntimeError("gateway timeout")
            matches = re.findall(
                r"\[RANK (\d+)\]\n(.*?)(?=\n\n\[RANK |\n\n\[END DATA])",
                user,
                re.DOTALL,
            )
            return {
                "documents": [
                    {
                        "rank": int(rank),
                        "relevant": "GOOD" in text,
                        "reason": "useful" if "GOOD" in text else "irrelevant",
                    }
                    for rank, text in matches
                ]
            }
        finally:
            self.current -= 1


class SyncJudge:
    def __init__(self):
        self.calls = 0

    def generate_object(self, prompt, schema):
        self.calls += 1
        user = prompt[1]["content"]
        matches = re.findall(
            r"\[RANK (\d+)\]\n(.*?)(?=\n\n\[RANK |\n\n\[END DATA])",
            user,
            re.DOTALL,
        )
        return {
            "documents": [
                {
                    "rank": int(rank),
                    "relevant": "GOOD" in text,
                    "reason": "judged",
                }
                for rank, text in matches
            ]
        }


def _docs(n, pattern="GOOD"):
    return [{"text": f"{pattern} {index}"} for index in range(n)]


def _framework(judge):
    return EvaluationFramework(
        evaluators=[
            RelevanceAtKEvaluator(k=5),
            HitRateAtKEvaluator(k=5),
            MRRAtKEvaluator(k=5),
            NDCGAtKEvaluator(k=5),
        ],
        judge=judge,
    )


def test_native_async_four_metrics_use_one_call_and_no_sync_bridge():
    judge = AsyncProbeJudge(delay=0)
    result = asyncio.run(
        _framework(judge).a_evaluate(
            EvaluationCase(input="q", retrieved_documents=_docs(5))
        )
    )
    assert judge.calls == 1
    assert judge.sync_calls == 0
    assert set(result) == {
        "relevance_at_5",
        "hit_rate_at_5",
        "mrr_at_5",
        "ndcg_at_5",
    }


def test_sync_only_judge_falls_back_to_one_worker_thread_call():
    judge = SyncJudge()
    result = asyncio.run(
        EvaluationFramework(
            evaluators=[RelevanceAtKEvaluator(k=3)], judge=judge
        ).a_evaluate(EvaluationCase(input="q", retrieved_documents=_docs(3)))
    )
    assert result["relevance_at_3"].score == 1.0
    assert judge.calls == 1


def test_one_holistic_call_consumes_one_shared_concurrency_slot_per_case():
    judge = AsyncProbeJudge(delay=0.05)
    cases = [
        EvaluationCase(input="q", retrieved_documents=_docs(5), case_id=f"c{i}")
        for i in range(6)
    ]
    asyncio.run(_framework(judge).a_evaluate_many(cases, max_concurrency=2))
    assert judge.calls == 6
    assert judge.max_concurrent == 2


def test_async_result_document_order_follows_rank():
    judge = AsyncProbeJudge(delay=0)
    documents = [
        {"text": "GOOD 0"},
        {"text": "bad 1"},
        {"text": "GOOD 2"},
        {"text": "bad 3"},
    ]
    result = asyncio.run(
        EvaluationFramework(
            evaluators=[RelevanceAtKEvaluator(k=4)], judge=judge
        ).a_evaluate(EvaluationCase(input="q", retrieved_documents=documents))
    )
    assert [
        document["relevance_score"]
        for document in result["relevance_at_4"].details["documents"]
    ] == [1.0, 0.0, 1.0, 0.0]


def test_async_error_propagates_and_is_not_converted_to_score():
    judge = AsyncProbeJudge(delay=0, explode_on="FAIL")
    case = EvaluationCase(input="q", retrieved_documents=[{"text": "FAIL"}])
    with pytest.raises(RuntimeError, match="gateway timeout"):
        asyncio.run(_framework(judge).a_evaluate(case))
    assert judge.calls == 1


def test_async_empty_documents_skip_both_native_and_sync_calls():
    judge = AsyncProbeJudge(delay=0)
    results = asyncio.run(
        _framework(judge).a_evaluate(
            EvaluationCase(input="q", retrieved_documents=[])
        )
    )
    assert judge.calls == judge.sync_calls == 0
    assert all(result.label == "not_applicable" for result in results.values())


def test_a_evaluate_many_preserves_order_and_separate_root_traces():
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

    judge = AsyncProbeJudge(delay=0)
    cases = [
        EvaluationCase(
            input="q",
            retrieved_documents=[{"text": text}],
            case_id=case_id,
        )
        for case_id, text in (("c1", "GOOD"), ("c2", "bad"), ("c3", "GOOD"))
    ]
    results = asyncio.run(
        EvaluationFramework(
            evaluators=[RelevanceAtKEvaluator(k=1)], judge=judge
        ).a_evaluate_many(cases, max_concurrency=2)
    )
    assert [result["relevance_at_1"].score for result in results] == [1.0, 0.0, 1.0]
    roots = [
        span
        for span in exporter.get_finished_spans()
        if span.name == "idp_eval.evaluate"
    ]
    stages = [
        span
        for span in exporter.get_finished_spans()
        if span.name == "retrieval.relevance.evaluate"
    ]
    assert len(roots) == len(stages) == 3
    assert len({span.context.trace_id for span in roots}) == 3
    exporter.clear()
