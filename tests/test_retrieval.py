"""Retrieval evaluators: Phoenix wrapper, shared relevance pass, results, Excel.

Offline: Phoenix's DocumentRelevanceEvaluator is replaced by a fake via the
``_build_document_relevance_evaluator`` seam. No LLM/Phoenix/gateway calls.
"""

import json

import pytest

import idp_eval.evaluators.retrieval as retrieval_mod
from idp_eval import (
    EvaluationCase,
    EvaluationFramework,
    NDCGAtKEvaluator,
    RelevanceAtKEvaluator,
)

openpyxl = pytest.importorskip("openpyxl")


class FakeScore:
    def __init__(self, score, label, explanation="because"):
        self.score = score
        self.label = label
        self.explanation = explanation


class RelevanceHarness:
    """Fake Phoenix DocumentRelevanceEvaluator factory + call recorder.

    A document is relevant (1.0) iff its text contains ``"GOOD"`` unless a
    ``relevance_map`` overrides it. Records every ``evaluate`` record so tests can
    assert query/document, ordering, and call counts.
    """

    def __init__(self, *, relevance_map=None):
        self.relevance_map = relevance_map or {}
        self.records: list[dict] = []

    def _relevance(self, text: str) -> float:
        if text in self.relevance_map:
            return float(self.relevance_map[text])
        return 1.0 if "GOOD" in text else 0.0

    def factory(self, llm):
        harness = self

        class _FakeEvaluator:
            def evaluate(self, record):
                harness.records.append(record)
                rel = harness._relevance(record["document_text"])
                label = "relevant" if rel >= 1.0 else "unrelated"
                return [FakeScore(rel, label)]

        return _FakeEvaluator()

    @property
    def call_count(self) -> int:
        return len(self.records)


@pytest.fixture
def harness(monkeypatch):
    h = RelevanceHarness()
    monkeypatch.setattr(
        retrieval_mod, "_build_document_relevance_evaluator", h.factory
    )
    return h


def _docs(*texts):
    return [{"text": t} for t in texts]


def _case(*texts, query="q"):
    return EvaluationCase(input=query, retrieved_documents=list(texts))


# --- Phoenix wrapper (item 19) ----------------------------------------------


def test_query_and_document_sent_one_at_a_time(harness):
    case = EvaluationCase(
        input="How do I reset my password?",
        context="SHOULD_NOT_BE_SENT",
        output="SHOULD_NOT_BE_SENT",
        retrieved_documents=_docs("GOOD a", "bad b"),
    )
    RelevanceAtKEvaluator(k=2, llm=object()).evaluate(case)
    assert harness.call_count == 2  # one per document
    for record in harness.records:
        assert record["input"] == "How do I reset my password?"
        # Phoenix contract: input + document_text only.
        assert set(record) == {"input", "document_text"}
        assert "document" not in record       # old key must be gone
        assert "SHOULD_NOT_BE_SENT" not in json.dumps(record)  # no context/output


def test_exact_phoenix_payload_contract(harness):
    # Exactly what the Phoenix DocumentRelevanceEvaluator receives, including that
    # retrieval metadata (id / similarity score / nested metadata) is NOT sent.
    case = EvaluationCase(
        input="How do I reset my password?",
        retrieved_documents=[
            {
                "text": "Use the reset-password page.",
                "document_id": "doc-1",
                "score": 0.94,
                "metadata": {"source": "help-center"},
            }
        ],
    )
    RelevanceAtKEvaluator(k=1, llm=object()).evaluate(case)
    assert harness.records == [
        {
            "input": "How do I reset my password?",
            "document_text": "Use the reset-password page.",
        }
    ]
    payload = harness.records[0]
    assert "document_text" in payload and "document" not in payload
    # None of the retrieval metadata leaks into the relevance judgment.
    for leaked in ("document_id", "doc-1", "0.94", "score", "metadata", "help-center"):
        assert leaked not in json.dumps(payload)


def test_document_order_and_scores_preserved(harness):
    case = _case(*_docs("GOOD a", "bad b", "GOOD c"))
    result = RelevanceAtKEvaluator(k=3, llm=object()).evaluate(case)
    docs = result.details["documents"]
    assert [d["rank"] for d in docs] == [1, 2, 3]
    assert [d["relevance_score"] for d in docs] == [1.0, 0.0, 1.0]
    assert [d["relevance_label"] for d in docs] == [
        "relevant", "unrelated", "relevant"
    ]


def test_similarity_score_not_sent_but_kept_as_metadata(harness):
    case = EvaluationCase(
        input="q",
        retrieved_documents=[{"text": "GOOD a", "score": 0.91, "document_id": "d1"}],
    )
    result = RelevanceAtKEvaluator(k=1, llm=object()).evaluate(case)
    assert "score" not in harness.records[0]           # similarity not sent to LLM
    doc = result.details["documents"][0]
    assert doc["retrieval_score"] == 0.91              # retained as diagnostics
    assert doc["document_id"] == "d1"


# --- Relevance@K (item 20) --------------------------------------------------


@pytest.mark.parametrize(
    "rel,k,expected,label",
    [
        (["GOOD", "GOOD", "GOOD"], 3, 1.0, "all_relevant"),
        (["GOOD", "bad", "GOOD"], 3, 2 / 3, "partially_relevant"),
        (["bad", "bad", "bad"], 3, 0.0, "none_relevant"),
    ],
)
def test_relevance_at_k_scores_and_labels(harness, rel, k, expected, label):
    result = RelevanceAtKEvaluator(k=k, llm=object()).evaluate(_case(*_docs(*rel)))
    assert result.score == pytest.approx(expected)
    assert result.label == label
    assert result.metric == f"relevance_at_{k}"


def test_relevance_fewer_docs_than_k_uses_effective_k(harness):
    result = RelevanceAtKEvaluator(k=5, llm=object()).evaluate(
        _case(*_docs("GOOD", "bad"))
    )
    assert result.details["effective_k"] == 2   # min(5, 2)
    assert result.details["document_count"] == 2
    assert result.score == 0.5                   # 1 of 2, not 1 of 5


def test_zero_documents_is_not_applicable_no_judge_calls(harness):
    case = EvaluationCase(input="q", retrieved_documents=[])
    result = RelevanceAtKEvaluator(k=3, llm=object()).evaluate(case)
    assert result.score is None
    assert result.label == "not_applicable"
    assert harness.call_count == 0


@pytest.mark.parametrize("bad", [0, -1, 1.5, "3", True, None])
def test_k_validation(bad):
    with pytest.raises(ValueError, match="k must be a positive integer"):
        RelevanceAtKEvaluator(k=bad)
    with pytest.raises(ValueError, match="k must be a positive integer"):
        NDCGAtKEvaluator(k=bad)


# --- nDCG@K result wiring (values covered in test_retrieval_scoring) ---------


def test_ndcg_result_details(harness):
    result = NDCGAtKEvaluator(k=3, llm=object()).evaluate(
        _case(*_docs("GOOD", "bad", "GOOD"))
    )
    assert result.metric == "ndcg_at_3"
    assert result.score == pytest.approx(0.9197207891481876)
    assert result.label == "suboptimal_ranking"
    assert result.details["relevance_scores"] == [1.0, 0.0, 1.0]
    assert result.details["dcg"] == pytest.approx(1.5)


def test_ndcg_all_irrelevant_is_zero(harness):
    result = NDCGAtKEvaluator(k=3, llm=object()).evaluate(
        _case(*_docs("bad", "bad", "bad"))
    )
    assert result.score == 0.0
    assert result.label == "no_relevant_retrieved"


# --- shared judge pass (item 22, 17) ----------------------------------------


def test_two_metrics_same_k_judge_once(harness):
    case = _case(*_docs("GOOD", "bad", "GOOD", "bad", "GOOD"))
    framework = EvaluationFramework(
        evaluators=[RelevanceAtKEvaluator(k=5), NDCGAtKEvaluator(k=5)],
        judge=object(),
    )
    results = framework.evaluate(case)
    assert harness.call_count == 5            # NOT 10
    assert set(results) == {"relevance_at_5", "ndcg_at_5"}


def test_mixed_k_judges_max_depth_once(harness):
    case = _case(*_docs("GOOD", "bad", "GOOD", "bad", "GOOD"))
    framework = EvaluationFramework(
        evaluators=[RelevanceAtKEvaluator(k=3), NDCGAtKEvaluator(k=5)],
        judge=object(),
    )
    results = framework.evaluate(case)
    assert harness.call_count == 5            # max(3, 5)
    assert results["relevance_at_3"].score == pytest.approx(2 / 3)  # top 3: 1,0,1


def test_subset_selection_limits_judged_depth(harness):
    case = _case(*_docs("GOOD", "bad", "GOOD", "bad", "GOOD"))
    framework = EvaluationFramework(
        evaluators=[RelevanceAtKEvaluator(k=3), NDCGAtKEvaluator(k=5)],
        judge=object(),
    )
    framework.evaluate(case, metrics=["relevance_at_3"])
    assert harness.call_count == 3            # only top 3 needed


def test_retrieval_shares_but_faithfulness_independent(harness):
    # Retrieval-only selection must not require output; faithfulness excluded.
    case = _case(*_docs("GOOD", "bad"))
    framework = EvaluationFramework(
        evaluators=[RelevanceAtKEvaluator(k=2), NDCGAtKEvaluator(k=2)],
        judge=object(),
    )
    results = framework.evaluate(case)
    assert set(results) == {"relevance_at_2", "ndcg_at_2"}
    assert harness.call_count == 2


# --- structured documents (item 24) -----------------------------------------


def test_string_documents_supported(harness):
    case = EvaluationCase(input="q", retrieved_documents=["GOOD a", "bad b"])
    result = RelevanceAtKEvaluator(k=2, llm=object()).evaluate(case)
    assert result.score == 0.5


def test_structured_document_with_metadata(harness):
    case = EvaluationCase(
        input="q",
        retrieved_documents=[
            {"text": "GOOD a", "document_id": "d1", "score": 0.9,
             "metadata": {"source": "kb"}},
        ],
    )
    result = RelevanceAtKEvaluator(k=1, llm=object()).evaluate(case)
    # Only the text is judged; nested metadata is not rendered into it.
    assert harness.records[0]["document_text"] == "GOOD a"
    assert result.details["documents"][0]["document_id"] == "d1"


def test_missing_document_text_raises(harness):
    case = EvaluationCase(input="q", retrieved_documents=[{"score": 0.9}])
    with pytest.raises(ValueError, match="missing non-empty 'text'"):
        RelevanceAtKEvaluator(k=1, llm=object()).evaluate(case)


def test_custom_text_key(harness):
    case = EvaluationCase(input="q", retrieved_documents=[{"body": "GOOD a"}])
    result = RelevanceAtKEvaluator(
        k=1, llm=object(), document_text_key="body"
    ).evaluate(case)
    assert result.score == 1.0
    # Custom text key -> document_text is the "body" field, still one field only.
    assert harness.records[0] == {"input": "q", "document_text": "GOOD a"}


# --- validation & required fields (item 11) ---------------------------------


def test_requires_input_and_documents_list():
    ev = RelevanceAtKEvaluator(k=3, llm=object())
    with pytest.raises(ValueError, match="requires non-empty `input`"):
        ev.validate_case(EvaluationCase(retrieved_documents=[]))
    with pytest.raises(ValueError, match="requires `retrieved_documents`"):
        ev.validate_case(EvaluationCase(input="q"))  # None documents


def test_extra_context_output_instructions_allowed(harness):
    case = EvaluationCase(
        input="q", context="c", output="o", instructions="i",
        retrieved_documents=_docs("GOOD"),
    )
    RelevanceAtKEvaluator(k=1, llm=object()).validate_case(case)  # no raise
    assert RelevanceAtKEvaluator(k=1, llm=object()).evaluate(case).score == 1.0


def test_missing_judge_raises_clear_error(harness):
    # No judge at construction and none injected -> clear error at judge time.
    ev = RelevanceAtKEvaluator(k=1)  # llm=None
    with pytest.raises(ValueError, match="needs a judge"):
        ev.evaluate(_case(*_docs("GOOD")))


# --- verbose document text ---------------------------------------------------


def test_verbose_includes_text_default_excludes_it(harness):
    case = _case(*_docs("GOOD a"))
    compact = RelevanceAtKEvaluator(k=1, llm=object()).evaluate(case)
    verbose = RelevanceAtKEvaluator(k=1, llm=object(), verbose=True).evaluate(case)
    assert "text" not in compact.details["documents"][0]
    assert verbose.details["documents"][0]["text"] == "GOOD a"
    assert compact.score == verbose.score  # verbose is diagnostics only


# --- tracing -----------------------------------------------------------------


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


def test_tracing_relevance_span_and_per_document_children(harness, spans):
    case = _case(*_docs("GOOD", "bad", "GOOD"))
    framework = EvaluationFramework(
        evaluators=[RelevanceAtKEvaluator(k=3), NDCGAtKEvaluator(k=3)],
        judge=object(),
    )
    framework.evaluate(case)

    finished = spans.get_finished_spans()
    names = [s.name for s in finished]
    assert names.count("idp_eval.evaluate") == 1
    assert names.count("retrieval.relevance") == 1                # shared, once
    assert names.count("retrieval.relevance.document") == 3       # one per doc
    root = next(s for s in finished if s.name == "idp_eval.evaluate")
    assert root.attributes["retrieval.document_count"] == 3
    assert root.attributes["retrieval.judged_count"] == 3
    assert root.attributes["retrieval.relevant_count"] == 2
    doc_spans = [s for s in finished if s.name == "retrieval.relevance.document"]
    assert {s.attributes["retrieval.rank"] for s in doc_spans} == {1, 2, 3}


# --- Excel (item 16) ---------------------------------------------------------


def _rows(path, sheet):
    rows = list(openpyxl.load_workbook(path)[sheet].iter_rows(values_only=True))
    header, *data = rows
    return header, [dict(zip(header, r)) for r in data]


def test_excel_shared_retrieval_documents_sheet_no_duplication(harness, tmp_path):
    path = tmp_path / "retrieval.xlsx"
    framework = EvaluationFramework(
        evaluators=[RelevanceAtKEvaluator(k=3), NDCGAtKEvaluator(k=3)],
        judge=object(),
        output="excel",
        excel_path=str(path),
    )
    framework.evaluate(
        EvaluationCase(
            input="q", case_id="c1",
            retrieved_documents=[
                {"text": "GOOD a", "document_id": "d1", "score": 0.9},
                {"text": "bad b", "document_id": "d2", "score": 0.8},
                {"text": "GOOD c", "document_id": "d3", "score": 0.7},
            ],
        )
    )
    names = openpyxl.load_workbook(path).sheetnames
    assert "retrieval_documents" in names
    header, rows = _rows(path, "retrieval_documents")
    # 3 documents, written ONCE despite two retrieval metrics (not 6).
    assert [r["rank"] for r in rows] == [1, 2, 3]
    assert [r["document_id"] for r in rows] == ["d1", "d2", "d3"]
    assert [r["relevance_score"] for r in rows] == [1.0, 0.0, 1.0]
    assert "metric" not in header  # shared identity, no per-metric column
    # Summary still has one row per metric.
    summary = _rows(path, "evaluations")[1]
    assert {r["metric"] for r in summary} == {"relevance_at_3", "ndcg_at_3"}
