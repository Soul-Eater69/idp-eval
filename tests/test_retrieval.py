"""One-call retrieval relevance and deterministic metric behavior (offline)."""

import json
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

openpyxl = pytest.importorskip("openpyxl")


class Judge:
    """Structured fake: document text containing GOOD is relevant."""

    def __init__(self, response=None):
        self.response = response
        self.calls: list[dict] = []

    def generate_object(self, prompt, schema):
        self.calls.append({"prompt": prompt, "schema": schema})
        if self.response is not None:
            return self.response
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
                    "reason": "Materially useful." if "GOOD" in text else "Unrelated.",
                }
                for rank, text in matches
            ]
        }


def _docs(*texts):
    return [{"text": text} for text in texts]


def _case(*texts, query="q"):
    return EvaluationCase(input=query, retrieved_documents=_docs(*texts))


def _all_metrics(k=5):
    return [
        RelevanceAtKEvaluator(k=k),
        HitRateAtKEvaluator(k=k),
        MRRAtKEvaluator(k=k),
        NDCGAtKEvaluator(k=k),
    ]


def test_one_query_multiple_documents_uses_exactly_one_structured_call():
    judge = Judge()
    result = RelevanceAtKEvaluator(k=3, llm=judge).evaluate(
        _case("GOOD reset instructions", "billing", "GOOD recovery advice")
    )
    assert len(judge.calls) == 1
    assert result.score == pytest.approx(2 / 3)
    assert result.details["judge_call_count"] == 1
    item_schema = judge.calls[0]["schema"]["properties"]["documents"]["items"]
    assert set(item_schema["properties"]) == {"rank", "relevant", "reason"}
    assert set(item_schema["required"]) == {"rank", "relevant", "reason"}


def test_prompt_contains_only_rendered_query_rank_and_document_text():
    judge = Judge()
    case = EvaluationCase(
        input={"question": "How do I reset my password?"},
        context="CONTEXT-MUST-NOT-LEAK",
        output="OUTPUT-MUST-NOT-LEAK",
        instructions="INSTRUCTIONS-MUST-NOT-LEAK",
        metadata={"case_secret": "CASE-METADATA-MUST-NOT-LEAK"},
        retrieved_documents=[
            {
                "text": "GOOD Use the reset-password page.",
                "document_id": "doc-private-1",
                "score": 0.94,
                "metadata": {"source": "help-center-private"},
            }
        ],
    )
    RelevanceAtKEvaluator(k=1, llm=judge).evaluate(case)
    payload = json.dumps(judge.calls[0]["prompt"])
    assert "Question: How do I reset my password?" in payload
    assert "[RANK 1]" in payload
    assert "GOOD Use the reset-password page." in payload
    for forbidden in (
        "CONTEXT-MUST-NOT-LEAK",
        "OUTPUT-MUST-NOT-LEAK",
        "INSTRUCTIONS-MUST-NOT-LEAK",
        "CASE-METADATA-MUST-NOT-LEAK",
        "doc-private-1",
        "0.94",
        "help-center-private",
    ):
        assert forbidden not in payload


def test_document_order_binary_scores_and_diagnostics_are_preserved():
    judge = Judge()
    case = EvaluationCase(
        input="q",
        retrieved_documents=[
            {"text": "GOOD a", "document_id": "d1", "score": 0.9},
            {"text": "bad b", "document_id": "d2", "score": 0.8},
            {"text": "GOOD c", "document_id": "d3", "score": 0.7},
        ],
    )
    documents = RelevanceAtKEvaluator(k=3, llm=judge).evaluate(case).details[
        "documents"
    ]
    assert [document["rank"] for document in documents] == [1, 2, 3]
    assert [document["relevant"] for document in documents] == [True, False, True]
    assert [document["relevance_score"] for document in documents] == [1.0, 0.0, 1.0]
    assert [document["document_id"] for document in documents] == ["d1", "d2", "d3"]
    assert [document["retrieval_score"] for document in documents] == [0.9, 0.8, 0.7]


@pytest.mark.parametrize(
    "texts,score,label",
    [
        (("GOOD", "GOOD"), 1.0, "all_relevant"),
        (("GOOD", "bad"), 0.5, "partially_relevant"),
        (("bad", "bad"), 0.0, "none_relevant"),
    ],
)
def test_relevance_at_k_is_binary_precision(texts, score, label):
    result = RelevanceAtKEvaluator(k=2, llm=Judge()).evaluate(_case(*texts))
    assert result.score == score
    assert result.label == label
    assert result.details["relevant_count"] == sum("GOOD" in text for text in texts)


@pytest.mark.parametrize(
    "texts,score,label,first_rank",
    [
        (("bad", "GOOD"), 1.0, "hit", 2),
        (("bad", "bad"), 0.0, "miss", None),
    ],
)
def test_hit_rate_at_k(texts, score, label, first_rank):
    result = HitRateAtKEvaluator(k=2, llm=Judge()).evaluate(_case(*texts))
    assert result.metric == "hit_rate_at_2"
    assert result.score == score
    assert result.label == label
    assert result.details["first_relevant_rank"] == first_rank


@pytest.mark.parametrize(
    "texts,score,label,first_rank",
    [
        (("GOOD", "bad", "GOOD"), 1.0, "first_result_relevant", 1),
        (("bad", "bad", "GOOD"), 1 / 3, "relevant_found", 3),
        (("bad", "bad", "bad"), 0.0, "no_relevant_result", None),
    ],
)
def test_mrr_at_k_is_per_query_reciprocal_rank(texts, score, label, first_rank):
    result = MRRAtKEvaluator(k=3, llm=Judge()).evaluate(_case(*texts))
    assert result.metric == "mrr_at_3"
    assert result.score == pytest.approx(score)
    assert result.label == label
    assert result.details["first_relevant_rank"] == first_rank


def test_ndcg_result_uses_shared_binary_order():
    result = NDCGAtKEvaluator(k=3, llm=Judge()).evaluate(
        _case("GOOD", "bad", "GOOD")
    )
    assert result.metric == "ndcg_at_3"
    assert result.score == pytest.approx(0.9197207891481876)
    assert result.label == "suboptimal_ranking"
    assert result.details["relevance_scores"] == [1.0, 0.0, 1.0]
    assert result.details["dcg"] == pytest.approx(1.5)


def test_ndcg_ideal_and_all_irrelevant_labels():
    ideal = NDCGAtKEvaluator(k=3, llm=Judge()).evaluate(
        _case("GOOD", "GOOD", "bad")
    )
    empty_gain = NDCGAtKEvaluator(k=3, llm=Judge()).evaluate(
        _case("bad", "bad", "bad")
    )
    assert ideal.score == 1.0 and ideal.label == "ideal_ranking"
    assert empty_gain.score == 0.0
    assert empty_gain.label == "no_relevant_retrieved"


def test_effective_k_uses_returned_document_count_not_requested_k():
    result = RelevanceAtKEvaluator(k=5, llm=Judge()).evaluate(
        _case("GOOD", "bad")
    )
    assert result.details["requested_k"] == 5
    assert result.details["effective_k"] == 2
    assert result.details["document_count"] == 2
    assert result.score == 0.5


def test_empty_documents_all_metrics_not_applicable_without_judge():
    framework = EvaluationFramework(evaluators=_all_metrics(k=5))
    results = framework.evaluate(EvaluationCase(input="q", retrieved_documents=[]))
    assert set(results) == {
        "relevance_at_5",
        "hit_rate_at_5",
        "mrr_at_5",
        "ndcg_at_5",
    }
    for result in results.values():
        assert result.score is None
        assert result.label == "not_applicable"
        assert result.details["effective_k"] == 0
        assert result.details["document_count"] == 0
        assert result.details["judge_call_count"] == 0
        assert result.details["documents"] == []


@pytest.mark.parametrize("bad", [0, -1, 1.5, "3", True, None])
@pytest.mark.parametrize(
    "evaluator", [RelevanceAtKEvaluator, HitRateAtKEvaluator, MRRAtKEvaluator, NDCGAtKEvaluator]
)
def test_k_validation(evaluator, bad):
    with pytest.raises(ValueError, match="k must be a positive integer"):
        evaluator(k=bad)


def test_plain_strings_structured_documents_and_custom_text_key():
    assert RelevanceAtKEvaluator(k=2, llm=Judge()).evaluate(
        EvaluationCase(input="q", retrieved_documents=["GOOD a", "bad b"])
    ).score == 0.5
    judge = Judge()
    result = RelevanceAtKEvaluator(
        k=1, llm=judge, document_text_key="body"
    ).evaluate(
        EvaluationCase(
            input="q",
            retrieved_documents=[
                {"body": "GOOD custom", "document_id": "d1", "score": 0.4}
            ],
        )
    )
    assert result.score == 1.0
    assert "GOOD custom" in judge.calls[0]["prompt"][1]["content"]


def test_missing_or_empty_document_text_fails_before_judge_work():
    judge = Judge()
    with pytest.raises(ValueError, match="missing non-empty 'text'"):
        RelevanceAtKEvaluator(k=1, llm=judge).evaluate(
            EvaluationCase(input="q", retrieved_documents=[{"score": 0.9}])
        )
    with pytest.raises(ValueError, match="empty text"):
        RelevanceAtKEvaluator(k=1, llm=judge).evaluate(
            EvaluationCase(input="q", retrieved_documents=["  "])
        )
    assert judge.calls == []


def test_requires_non_empty_input_and_documents_list():
    evaluator = RelevanceAtKEvaluator(k=3, llm=Judge())
    with pytest.raises(ValueError, match="requires non-empty `input`"):
        evaluator.validate_case(EvaluationCase(retrieved_documents=[]))
    with pytest.raises(ValueError, match="requires `retrieved_documents`"):
        evaluator.validate_case(EvaluationCase(input="q"))
    with pytest.raises(ValueError, match="requires `retrieved_documents`"):
        evaluator.validate_case(
            EvaluationCase(input="q", retrieved_documents={"text": "not-list"})
        )


def test_missing_judge_fails_only_when_nonempty_documents_need_judging():
    evaluator = RelevanceAtKEvaluator(k=1)
    assert evaluator.evaluate(
        EvaluationCase(input="q", retrieved_documents=[])
    ).label == "not_applicable"
    with pytest.raises(ValueError, match="needs a judge"):
        evaluator.evaluate(_case("GOOD"))


def test_four_metrics_same_k_share_one_call():
    judge = Judge()
    framework = EvaluationFramework(evaluators=_all_metrics(k=5), judge=judge)
    results = framework.evaluate(_case("GOOD", "bad", "GOOD", "bad", "GOOD"))
    assert len(judge.calls) == 1
    assert set(results) == {
        "relevance_at_5",
        "hit_rate_at_5",
        "mrr_at_5",
        "ndcg_at_5",
    }


def test_mixed_k_shares_one_call_through_max_selected_depth():
    judge = Judge()
    framework = EvaluationFramework(
        evaluators=[
            RelevanceAtKEvaluator(k=3),
            HitRateAtKEvaluator(k=5),
            MRRAtKEvaluator(k=5),
            NDCGAtKEvaluator(k=10),
        ],
        judge=judge,
    )
    framework.evaluate(_case("GOOD", "bad", "GOOD", "bad", "GOOD", "bad"))
    assert len(judge.calls) == 1
    prompt = judge.calls[0]["prompt"][1]["content"]
    assert "[RANK 6]" in prompt
    assert "[RANK 7]" not in prompt


def test_metric_subset_only_judges_required_depth():
    judge = Judge()
    framework = EvaluationFramework(
        evaluators=[RelevanceAtKEvaluator(k=3), NDCGAtKEvaluator(k=5)], judge=judge
    )
    framework.evaluate(
        _case("GOOD", "bad", "GOOD", "bad", "GOOD"),
        metrics=["relevance_at_3"],
    )
    prompt = judge.calls[0]["prompt"][1]["content"]
    assert "[RANK 3]" in prompt
    assert "[RANK 4]" not in prompt


def test_multiple_relevance_k_values_share_one_call_and_slice():
    judge = Judge()
    framework = EvaluationFramework(
        evaluators=[RelevanceAtKEvaluator(k=2), RelevanceAtKEvaluator(k=4)],
        judge=judge,
    )
    results = framework.evaluate(_case("GOOD", "bad", "GOOD", "bad"))
    assert len(judge.calls) == 1
    assert results["relevance_at_2"].score == 0.5
    assert results["relevance_at_4"].score == 0.5


@pytest.mark.parametrize(
    "response,match",
    [
        ({}, "missing `documents`"),
        ({"documents": None}, "must be a list"),
        ({"documents": ["bad"]}, "expected an object"),
        ({"documents": [{"rank": "1", "relevant": True, "reason": "x"}]}, "rank.*integer"),
        ({"documents": [{"rank": 1, "relevant": 1, "reason": "x"}]}, "relevant.*boolean"),
        ({"documents": [{"rank": 1, "relevant": True, "reason": ""}]}, "reason.*non-empty"),
        (
            {"documents": [
                {"rank": 1, "relevant": True, "reason": "x"},
                {"rank": 1, "relevant": False, "reason": "y"},
            ]},
            "duplicate rank 1",
        ),
        (
            {"documents": [
                {"rank": 1, "relevant": True, "reason": "x"},
                {"rank": 3, "relevant": False, "reason": "y"},
            ]},
            r"missing=\[2\].*out_of_range=\[3\]",
        ),
        (
            {"documents": [
                {"rank": 1, "relevant": True, "reason": "x"},
            ]},
            r"missing=\[2\].*returned=1",
        ),
    ],
)
def test_malformed_judge_response_fails_clearly(response, match):
    with pytest.raises(ValueError, match=match):
        RelevanceAtKEvaluator(k=2, llm=Judge(response)).evaluate(
            _case("GOOD", "bad")
        )


def test_reordered_valid_ranks_are_reconstructed_in_rank_order():
    judge = Judge(
        {
            "documents": [
                {"rank": 2, "relevant": False, "reason": "second"},
                {"rank": 1, "relevant": True, "reason": "first"},
            ]
        }
    )
    result = RelevanceAtKEvaluator(k=2, llm=judge, verbose=True).evaluate(
        _case("GOOD", "bad")
    )
    assert [document["rank"] for document in result.details["documents"]] == [1, 2]
    assert [document["reason"] for document in result.details["documents"]] == [
        "first",
        "second",
    ]


def test_verbose_adds_text_and_reason_without_changing_score():
    case = _case("GOOD a", "bad b")
    compact = RelevanceAtKEvaluator(k=2, llm=Judge()).evaluate(case)
    verbose = RelevanceAtKEvaluator(k=2, llm=Judge(), verbose=True).evaluate(case)
    assert compact.score == verbose.score
    assert "text" not in compact.details["documents"][0]
    assert "reason" not in compact.details["documents"][0]
    assert verbose.details["documents"][0]["text"] == "GOOD a"
    assert verbose.details["documents"][0]["reason"] == "Materially useful."


def test_scope_expansion_remains_framework_owned_and_one_call_per_logical_case():
    judge = Judge()
    framework = EvaluationFramework(
        evaluators=[RelevanceAtKEvaluator(k=2)], judge=judge
    )
    results = framework.evaluate(
        EvaluationCase(
            input="q",
            retrieved_documents=_docs("GOOD", "bad"),
            output=["a", "b"],
            evaluation_scope="both",
        )
    )
    assert len(judge.calls) == 3  # combined + two individual logical cases
    assert results["combined"]["relevance_at_2"].score == 0.5
    assert len(results["individual"]) == 2


def test_sync_evaluate_many_uses_one_call_per_case_and_preserves_order():
    judge = Judge()
    framework = EvaluationFramework(
        evaluators=[RelevanceAtKEvaluator(k=1)], judge=judge
    )
    results = framework.evaluate_many([_case("GOOD"), _case("bad")])
    assert len(judge.calls) == 2
    assert [result["relevance_at_1"].score for result in results] == [1.0, 0.0]


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


def test_tracing_has_one_relevance_stage_and_no_per_document_spans(spans):
    framework = EvaluationFramework(evaluators=_all_metrics(k=3), judge=Judge())
    framework.evaluate(_case("GOOD", "bad", "GOOD"))
    finished = spans.get_finished_spans()
    names = [span.name for span in finished]
    assert names.count("idp_eval.evaluate") == 1
    assert names.count("retrieval.relevance.evaluate") == 1
    assert not any(name == "retrieval.relevance.document" for name in names)
    root = next(span for span in finished if span.name == "idp_eval.evaluate")
    assert root.attributes["retrieval.document_count"] == 3
    assert root.attributes["retrieval.judged_count"] == 3
    assert root.attributes["retrieval.relevant_count"] == 2
    assert root.attributes["retrieval.max_k"] == 3


def _rows(path, sheet):
    rows = list(openpyxl.load_workbook(path)[sheet].iter_rows(values_only=True))
    header, *data = rows
    return header, [dict(zip(header, row)) for row in data]


def test_excel_persists_all_metrics_and_shared_documents_once(tmp_path):
    path = tmp_path / "retrieval.xlsx"
    framework = EvaluationFramework(
        evaluators=_all_metrics(k=3),
        judge=Judge(),
        output="excel",
        excel_path=str(path),
    )
    framework.evaluate(
        EvaluationCase(
            input="q",
            case_id="c1",
            retrieved_documents=[
                {"text": "GOOD a", "document_id": "d1", "score": 0.9},
                {"text": "bad b", "document_id": "d2", "score": 0.8},
                {"text": "GOOD c", "document_id": "d3", "score": 0.7},
            ],
        )
    )
    header, rows = _rows(path, "retrieval_documents")
    assert len(rows) == 3
    assert [row["rank"] for row in rows] == [1, 2, 3]
    assert [row["relevant"] for row in rows] == [True, False, True]
    assert "metric" not in header
    summary = _rows(path, "evaluations")[1]
    assert {row["metric"] for row in summary} == {
        "relevance_at_3",
        "hit_rate_at_3",
        "mrr_at_3",
        "ndcg_at_3",
    }
