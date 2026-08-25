"""Retrieval metrics derived from one shared binary relevance judgment.

For one evaluation case, the framework asks the configured judge to classify all
needed ranked documents in one structured call. Relevance@K, Hit Rate@K, MRR@K,
nDCG@K, and Contextual Precision@K then slice those shared judgments and
calculate their scores in Python. Contextual Relevancy and Contextual Recall use
separate one-call semantic contracts because they evaluate different units. The
internal relevance pass is intentionally not part of the public API.
"""

from __future__ import annotations

import asyncio
import time

from idp_eval import tracing
from idp_eval.models import EvaluationCase, EvaluationResult, Evaluator
from idp_eval.prompts.retrieval import (
    CONTEXTUAL_RECALL_SCHEMA_V1,
    CONTEXTUAL_RELEVANCY_SCHEMA_V1,
    RETRIEVAL_RELEVANCE_SCHEMA_V1,
    render_contextual_recall_prompt,
    render_contextual_relevancy_prompt,
    render_retrieval_relevance_prompt,
)
from idp_eval.rendering import render_value
from idp_eval.scoring import (
    contextual_precision_at_k,
    contextual_precision_at_k_label,
    contextual_recall_label,
    contextual_recall_score,
    contextual_relevancy_label,
    contextual_relevancy_score,
    hit_rate_at_k,
    hit_rate_at_k_label,
    mrr_at_k_label,
    ndcg_at_k,
    ndcg_at_k_label,
    reciprocal_rank_at_k,
    relevance_at_k,
    relevance_at_k_label,
)

DEFAULT_DOCUMENT_TEXT_KEY = "text"


def _elapsed_ms(started: float) -> float:
    return (time.monotonic() - started) * 1000.0


def _validate_k(k: int) -> None:
    """Rejects a non-positive integer cutoff (including ``bool``)."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}.")


def _document_text(document, text_key: str, rank: int) -> str:
    """Extracts only the document text that may be sent to the judge."""
    if isinstance(document, str):
        if not document.strip():
            raise ValueError(f"Retrieved document at rank {rank} has empty text.")
        return document
    if isinstance(document, dict):
        text = document.get(text_key)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"Retrieved document at rank {rank} is missing non-empty "
                f"{text_key!r} text."
            )
        return text
    raise ValueError(
        f"Retrieved document at rank {rank} must be a string or a mapping with "
        f"a {text_key!r} field."
    )


def _document_id(document) -> str | None:
    if isinstance(document, dict):
        value = document.get("document_id", document.get("id"))
        return value if isinstance(value, str) else None
    return None


def _retrieval_score(document):
    if isinstance(document, dict):
        return document.get("score")
    return None


def _validate_documents_list(case: EvaluationCase, evaluator_name: str) -> list:
    documents = case.retrieved_documents
    if documents is None or not isinstance(documents, list):
        raise ValueError(
            f"{evaluator_name} requires `retrieved_documents` as a list "
            "(rank order); got "
            f"{type(documents).__name__ if documents is not None else 'None'}."
        )
    return documents


def _validate_document_text_key(document_text_key: str) -> None:
    if not isinstance(document_text_key, str) or not document_text_key:
        raise ValueError("document_text_key must be a non-empty string.")


def _document_texts(documents: list, text_key: str) -> list[str]:
    return [
        _document_text(document, text_key, rank)
        for rank, document in enumerate(documents, start=1)
    ]


def _validate_relevance_response(
    response: object, expected_count: int
) -> list[dict]:
    """Validates one holistic response and reconstructs exact rank order."""
    if not isinstance(response, dict) or "documents" not in response:
        raise ValueError(
            "Malformed retrieval relevance response: missing `documents`."
        )
    raw_documents = response["documents"]
    if not isinstance(raw_documents, list):
        raise ValueError(
            "Malformed retrieval relevance response: `documents` must be a list."
        )

    by_rank: dict[int, dict] = {}
    for index, item in enumerate(raw_documents, start=1):
        if not isinstance(item, dict):
            raise ValueError(
                f"Malformed retrieval relevance item {index}: expected an object."
            )
        rank = item.get("rank")
        relevant = item.get("relevant")
        reason = item.get("reason")
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise ValueError(
                f"Malformed retrieval relevance item {index}: `rank` must be an "
                "integer."
            )
        if rank in by_rank:
            raise ValueError(
                f"Malformed retrieval relevance response: duplicate rank {rank}."
            )
        if not isinstance(relevant, bool):
            raise ValueError(
                f"Malformed retrieval relevance item {index}: `relevant` must be "
                "a boolean."
            )
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(
                f"Malformed retrieval relevance item {index}: `reason` must be a "
                "non-empty string."
            )
        by_rank[rank] = {
            "rank": rank,
            "relevant": relevant,
            "reason": reason,
        }

    expected = set(range(1, expected_count + 1))
    actual = set(by_rank)
    if len(raw_documents) != expected_count or actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(
            "Malformed retrieval relevance response: ranks must appear exactly "
            f"once from 1 through {expected_count}; missing={missing}, "
            f"out_of_range={unknown}, returned={len(raw_documents)}."
        )
    return [by_rank[rank] for rank in range(1, expected_count + 1)]


class _RetrievalRelevancePass:
    """One internal batched relevance call reused by selected retrieval metrics."""

    def __init__(self, llm, query: str, documents: list, text_key: str):
        self._llm = llm
        self._query = query
        self._documents = documents
        self._text_key = text_key
        self._judgments: list[dict] = []

    def _prepare(self, depth: int) -> tuple[int, list[str], list[dict]]:
        depth = min(depth, len(self._documents))
        documents = self._documents[:depth]
        texts = [
            _document_text(document, self._text_key, rank)
            for rank, document in enumerate(documents, start=1)
        ]
        prompt = render_retrieval_relevance_prompt(self._query, texts)
        return depth, texts, prompt

    def run(self, depth: int) -> None:
        """Makes one synchronous structured call for all documents through depth."""
        depth = min(depth, len(self._documents))
        if depth == 0:
            self._set_trace_attributes(depth)
            return
        if self._judgments:
            if depth > len(self._judgments):
                raise RuntimeError("Shared retrieval relevance pass already ran.")
            self._set_trace_attributes(len(self._judgments))
            return
        if self._llm is None:
            raise ValueError(
                "Retrieval evaluator needs a judge; pass one to the constructor "
                "or via EvaluationFramework(judge=...)."
            )
        depth, texts, prompt = self._prepare(depth)
        with tracing.judge_span(
            "retrieval.relevance.evaluate",
            {"idp_eval.metric": "retrieval_relevance", "idp_eval.stage": "evaluate"},
        ):
            response = self._llm.generate_object(
                prompt=prompt, schema=RETRIEVAL_RELEVANCE_SCHEMA_V1
            )
        self._complete(response, depth, texts)

    async def a_run(self, depth: int, limiter: asyncio.Semaphore) -> None:
        """Makes one async structured call under one shared-concurrency slot."""
        depth = min(depth, len(self._documents))
        if depth == 0:
            self._set_trace_attributes(depth)
            return
        if self._judgments:
            if depth > len(self._judgments):
                raise RuntimeError("Shared retrieval relevance pass already ran.")
            self._set_trace_attributes(len(self._judgments))
            return
        if self._llm is None:
            raise ValueError(
                "Retrieval evaluator needs a judge; pass one to the constructor "
                "or via EvaluationFramework(judge=...)."
            )
        depth, texts, prompt = self._prepare(depth)
        async with limiter:
            with tracing.judge_span(
                "retrieval.relevance.evaluate",
                {
                    "idp_eval.metric": "retrieval_relevance",
                    "idp_eval.stage": "evaluate",
                },
            ):
                async_generate = getattr(self._llm, "async_generate_object", None)
                if callable(async_generate):
                    response = await async_generate(
                        prompt=prompt, schema=RETRIEVAL_RELEVANCE_SCHEMA_V1
                    )
                else:
                    response = await asyncio.to_thread(
                        self._llm.generate_object,
                        prompt=prompt,
                        schema=RETRIEVAL_RELEVANCE_SCHEMA_V1,
                    )
        self._complete(response, depth, texts)

    def _complete(self, response: object, depth: int, texts: list[str]) -> None:
        raw = _validate_relevance_response(response, depth)
        self._judgments = []
        for judgment, document, text in zip(
            raw, self._documents[:depth], texts, strict=True
        ):
            relevant = judgment["relevant"]
            self._judgments.append(
                {
                    "rank": judgment["rank"],
                    "document_id": _document_id(document),
                    "relevant": relevant,
                    "relevance_score": 1.0 if relevant else 0.0,
                    "reason": judgment["reason"],
                    "retrieval_score": _retrieval_score(document),
                    "text": text,
                }
            )
        self._set_trace_attributes(depth)

    def judgments(self, depth: int) -> list[dict]:
        return self._judgments[:depth]

    def _set_trace_attributes(self, depth: int) -> None:
        tracing.set_current_span_attributes(
            {
                "retrieval.document_count": len(self._documents),
                "retrieval.judged_count": len(self._judgments),
                "retrieval.relevant_count": sum(
                    judgment["relevant"] for judgment in self._judgments
                ),
                "retrieval.max_k": depth,
            }
        )


class _RetrievalEvaluator(Evaluator):
    """Small shared base for deterministic metrics over batched judgments."""

    uses_retrieval_relevance = True
    required_fields = ("input",)
    name = "retrieval"

    def __init__(
        self,
        k: int,
        llm=None,
        *,
        verbose: bool = False,
        document_text_key: str = DEFAULT_DOCUMENT_TEXT_KEY,
    ):
        _validate_k(k)
        _validate_document_text_key(document_text_key)
        self._k = k
        self._llm = llm
        self._verbose = verbose
        self._document_text_key = document_text_key

    def _bind_judge(self, judge) -> None:
        if self._llm is None:
            self._llm = judge

    def resume_signature(self) -> dict:
        return {
            "contract_version": 1,
            "k": self._k,
            "verbose": self._verbose,
            "document_text_key": self._document_text_key,
            "judge": self.judge_resume_signature(self._llm),
        }

    def validate_case(self, case: EvaluationCase) -> None:
        super().validate_case(case)
        _validate_documents_list(case, type(self).__name__)

    def _documents(self, case: EvaluationCase) -> list:
        return case.retrieved_documents or []

    def _relevance_depth(self, case: EvaluationCase) -> int:
        return min(self._k, len(self._documents(case)))

    def _build_relevance_pass(self, case: EvaluationCase) -> _RetrievalRelevancePass:
        return _RetrievalRelevancePass(
            self._llm,
            render_value(case.input),
            self._documents(case),
            self._document_text_key,
        )

    def evaluate_shared(
        self, case: EvaluationCase, relevance_pass: _RetrievalRelevancePass
    ) -> EvaluationResult:
        document_count = len(self._documents(case))
        if document_count == 0:
            return self._not_applicable_result()
        effective_k = self._relevance_depth(case)
        return self._score(
            relevance_pass.judgments(effective_k), effective_k, document_count
        )

    def evaluate(self, case: EvaluationCase) -> EvaluationResult:
        self.validate_case(case)
        relevance_pass = self._build_relevance_pass(case)
        relevance_pass.run(self._relevance_depth(case))
        return self.evaluate_shared(case, relevance_pass)

    async def a_evaluate(
        self, case: EvaluationCase, *, judge_limiter: asyncio.Semaphore
    ) -> EvaluationResult:
        self.validate_case(case)
        relevance_pass = self._build_relevance_pass(case)
        await relevance_pass.a_run(self._relevance_depth(case), judge_limiter)
        return self.evaluate_shared(case, relevance_pass)

    def _score(self, judgments, effective_k, document_count):
        raise NotImplementedError

    def _empty_metric_details(self) -> dict:
        return {}

    def _not_applicable_result(self) -> EvaluationResult:
        details = self._common_details([], 0, 0)
        details.update(self._empty_metric_details())
        return EvaluationResult(
            metric=self.name,
            score=None,
            label="not_applicable",
            explanation="No documents were retrieved for the query.",
            details=details,
        )

    def _common_details(
        self, judgments: list[dict], effective_k: int, document_count: int
    ) -> dict:
        return {
            "requested_k": self._k,
            "effective_k": effective_k,
            "document_count": document_count,
            "judge_call_count": 1 if judgments else 0,
            "verbose": self._verbose,
            "documents": self._documents_diagnostic(judgments),
        }

    def _documents_diagnostic(self, judgments: list[dict]) -> list[dict]:
        compact_keys = (
            "rank",
            "document_id",
            "relevant",
            "relevance_score",
            "retrieval_score",
        )
        documents = [
            {key: judgment.get(key) for key in compact_keys}
            for judgment in judgments
        ]
        if self._verbose:
            for document, judgment in zip(documents, judgments, strict=True):
                document["text"] = judgment["text"]
                document["reason"] = judgment["reason"]
        return documents


class RelevanceAtKEvaluator(_RetrievalEvaluator):
    """Fraction of the top effective-K documents that are relevant."""

    def __init__(self, k: int, llm=None, **kwargs):
        super().__init__(k, llm, **kwargs)
        self.name = f"relevance_at_{k}"

    def _score(self, judgments, effective_k, document_count):
        scores = [judgment["relevance_score"] for judgment in judgments]
        score = relevance_at_k(scores)
        relevant_count = int(sum(scores))
        details = self._common_details(judgments, effective_k, document_count)
        details["relevant_count"] = relevant_count
        return EvaluationResult(
            metric=self.name,
            score=score,
            label=relevance_at_k_label(score),
            explanation=(
                f"{relevant_count} of the top {effective_k} retrieved documents "
                f"are relevant (Relevance@{self._k})."
            ),
            details=details,
        )

    def _empty_metric_details(self) -> dict:
        return {"relevant_count": 0}


class HitRateAtKEvaluator(_RetrievalEvaluator):
    """Whether at least one top effective-K document is relevant."""

    def __init__(self, k: int, llm=None, **kwargs):
        super().__init__(k, llm, **kwargs)
        self.name = f"hit_rate_at_{k}"

    def _score(self, judgments, effective_k, document_count):
        scores = [judgment["relevance_score"] for judgment in judgments]
        score = hit_rate_at_k(scores)
        relevant_count = int(sum(scores))
        first_rank = _first_relevant_rank(judgments)
        details = self._common_details(judgments, effective_k, document_count)
        details.update(
            {
                "relevant_count": relevant_count,
                "first_relevant_rank": first_rank,
            }
        )
        return EvaluationResult(
            metric=self.name,
            score=score,
            label=hit_rate_at_k_label(score),
            explanation=(
                f"A relevant document was found at rank {first_rank}."
                if first_rank is not None
                else f"No relevant document was found in the top {effective_k}."
            ),
            details=details,
        )

    def _empty_metric_details(self) -> dict:
        return {"relevant_count": 0, "first_relevant_rank": None}


class MRRAtKEvaluator(_RetrievalEvaluator):
    """Per-query reciprocal rank of the first relevant top-K document."""

    def __init__(self, k: int, llm=None, **kwargs):
        super().__init__(k, llm, **kwargs)
        self.name = f"mrr_at_{k}"

    def _score(self, judgments, effective_k, document_count):
        scores = [judgment["relevance_score"] for judgment in judgments]
        score = reciprocal_rank_at_k(scores)
        first_rank = _first_relevant_rank(judgments)
        details = self._common_details(judgments, effective_k, document_count)
        details["first_relevant_rank"] = first_rank
        return EvaluationResult(
            metric=self.name,
            score=score,
            label=mrr_at_k_label(first_rank),
            explanation=(
                f"The first relevant document is at rank {first_rank}; "
                f"reciprocal rank is {score:.4g}."
                if first_rank is not None
                else f"No relevant document was found in the top {effective_k}."
            ),
            details=details,
        )

    def _empty_metric_details(self) -> dict:
        return {"first_relevant_rank": None}


class NDCGAtKEvaluator(_RetrievalEvaluator):
    """Binary nDCG ranking quality over the top effective-K documents."""

    def __init__(self, k: int, llm=None, **kwargs):
        super().__init__(k, llm, **kwargs)
        self.name = f"ndcg_at_{k}"

    def _score(self, judgments, effective_k, document_count):
        scores = [judgment["relevance_score"] for judgment in judgments]
        score, dcg_value, idcg_value = ndcg_at_k(scores)
        details = self._common_details(judgments, effective_k, document_count)
        details.update(
            {
                "relevance_scores": scores,
                "dcg": dcg_value,
                "idcg": idcg_value,
            }
        )
        return EvaluationResult(
            metric=self.name,
            score=score,
            label=ndcg_at_k_label(score),
            explanation=(
                f"nDCG@{self._k} = {score:.4g} over the top {effective_k} "
                f"documents (DCG {dcg_value:.4g} / IDCG {idcg_value:.4g})."
            ),
            details=details,
        )

    def _empty_metric_details(self) -> dict:
        return {"relevance_scores": [], "dcg": 0.0, "idcg": 0.0}


class ContextualPrecisionAtKEvaluator(_RetrievalEvaluator):
    """AP-style ranking quality over shared top-K relevance judgments.

    This metric evaluates only the returned top-K list. It does not assume
    knowledge of relevant documents outside that list and is therefore not
    corpus-level recall-aware Average Precision.
    """

    def __init__(self, k: int, llm=None, **kwargs):
        super().__init__(k, llm, **kwargs)
        self.name = f"contextual_precision_at_{k}"

    def _score(self, judgments, effective_k, document_count):
        scores = [judgment["relevance_score"] for judgment in judgments]
        score = contextual_precision_at_k(scores)
        relevant_count = int(sum(scores))
        relevant_seen = 0
        precision_at_relevant_ranks = []
        for judgment in judgments:
            if judgment["relevant"]:
                relevant_seen += 1
                precision_at_relevant_ranks.append(
                    {
                        "rank": judgment["rank"],
                        "precision": relevant_seen / judgment["rank"],
                    }
                )
        details = self._common_details(judgments, effective_k, document_count)
        details.update(
            {
                "relevant_count": relevant_count,
                "precision_at_relevant_ranks": precision_at_relevant_ranks,
            }
        )
        return EvaluationResult(
            metric=self.name,
            score=score,
            label=contextual_precision_at_k_label(score),
            explanation=(
                f"Contextual Precision@{self._k} = {score:.4g} across "
                f"{relevant_count} relevant documents in the top {effective_k}."
            ),
            details=details,
        )

    def _empty_metric_details(self) -> dict:
        return {"relevant_count": 0, "precision_at_relevant_ranks": []}


class ContextualRelevancyEvaluator(Evaluator):
    """Fraction of meaningful information units in retrieval that are useful."""

    name = "contextual_relevancy"
    required_fields = ("input",)

    def __init__(
        self,
        llm=None,
        *,
        verbose: bool = False,
        document_text_key: str = DEFAULT_DOCUMENT_TEXT_KEY,
    ):
        _validate_document_text_key(document_text_key)
        self._llm = llm
        self._verbose = verbose
        self._document_text_key = document_text_key

    def _bind_judge(self, judge) -> None:
        if self._llm is None:
            self._llm = judge

    def resume_signature(self) -> dict:
        return {
            "contract_version": 1,
            "verbose": self._verbose,
            "document_text_key": self._document_text_key,
            "judge": self.judge_resume_signature(self._llm),
        }

    def validate_case(self, case: EvaluationCase) -> None:
        super().validate_case(case)
        _validate_documents_list(case, type(self).__name__)

    def _prepare(self, case: EvaluationCase) -> tuple[list[dict], int]:
        texts = _document_texts(
            case.retrieved_documents or [], self._document_text_key
        )
        prompt = render_contextual_relevancy_prompt(
            render_value(case.input), texts
        )
        return prompt, len(texts)

    def evaluate(self, case: EvaluationCase) -> EvaluationResult:
        self.validate_case(case)
        if not case.retrieved_documents:
            return self._empty_retrieval_result()
        if self._llm is None:
            raise ValueError(
                "ContextualRelevancyEvaluator needs a judge; pass one to the "
                "constructor or via EvaluationFramework(judge=...)."
            )
        started = time.monotonic()
        prompt, document_count = self._prepare(case)
        with tracing.judge_span(
            "contextual_relevancy.evaluate",
            {"idp_eval.metric": self.name, "idp_eval.stage": "evaluate"},
        ):
            response = self._llm.generate_object(
                prompt=prompt, schema=CONTEXTUAL_RELEVANCY_SCHEMA_V1
            )
        return self._result_from_response(
            response, document_count, _elapsed_ms(started)
        )

    async def a_evaluate(
        self, case: EvaluationCase, *, judge_limiter: asyncio.Semaphore
    ) -> EvaluationResult:
        self.validate_case(case)
        if not case.retrieved_documents:
            return self._empty_retrieval_result()
        if self._llm is None:
            raise ValueError(
                "ContextualRelevancyEvaluator needs a judge; pass one to the "
                "constructor or via EvaluationFramework(judge=...)."
            )
        started = time.monotonic()
        prompt, document_count = self._prepare(case)
        async with judge_limiter:
            with tracing.judge_span(
                "contextual_relevancy.evaluate",
                {"idp_eval.metric": self.name, "idp_eval.stage": "evaluate"},
            ):
                async_generate = getattr(
                    self._llm, "async_generate_object", None
                )
                if callable(async_generate):
                    response = await async_generate(
                        prompt=prompt, schema=CONTEXTUAL_RELEVANCY_SCHEMA_V1
                    )
                else:
                    response = await asyncio.to_thread(
                        self._llm.generate_object,
                        prompt=prompt,
                        schema=CONTEXTUAL_RELEVANCY_SCHEMA_V1,
                    )
        return self._result_from_response(
            response, document_count, _elapsed_ms(started)
        )

    def _validate_response(self, response: object, document_count: int) -> list[dict]:
        if not isinstance(response, dict) or set(response) != {"items"}:
            raise ValueError(
                "Malformed contextual relevancy response: expected only an "
                "`items` list."
            )
        raw_items = response["items"]
        if not isinstance(raw_items, list):
            raise ValueError(
                "Malformed contextual relevancy response: `items` must be a list."
            )
        validated = []
        expected_keys = {"document_rank", "context_item", "relevant", "reason"}
        for index, item in enumerate(raw_items, start=1):
            if not isinstance(item, dict) or set(item) != expected_keys:
                raise ValueError(
                    f"Malformed contextual relevancy item {index}: expected "
                    f"exactly {sorted(expected_keys)}."
                )
            rank = item["document_rank"]
            context_item = item["context_item"]
            relevant = item["relevant"]
            reason = item["reason"]
            if (
                isinstance(rank, bool)
                or not isinstance(rank, int)
                or not 1 <= rank <= document_count
            ):
                raise ValueError(
                    f"Malformed contextual relevancy item {index}: "
                    f"`document_rank` must be from 1 through {document_count}."
                )
            if not isinstance(context_item, str) or not context_item.strip():
                raise ValueError(
                    f"Malformed contextual relevancy item {index}: "
                    "`context_item` must be a non-empty string."
                )
            if not isinstance(relevant, bool):
                raise ValueError(
                    f"Malformed contextual relevancy item {index}: `relevant` "
                    "must be a boolean."
                )
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(
                    f"Malformed contextual relevancy item {index}: `reason` "
                    "must be a non-empty string."
                )
            validated.append(
                {
                    "document_rank": rank,
                    "context_item": " ".join(context_item.split()),
                    "relevant": relevant,
                    "reason": reason,
                }
            )
        return validated

    def _result_from_response(
        self, response: object, document_count: int, total_ms: float
    ) -> EvaluationResult:
        items = self._validate_response(response, document_count)
        relevant_count = sum(item["relevant"] for item in items)
        details = {
            "item_count": len(items),
            "relevant_count": relevant_count,
            "judge_call_count": 1,
            "total_ms": total_ms,
            "verbose": self._verbose,
        }
        if self._verbose:
            details["items"] = items
        self._set_trace_attributes(len(items), relevant_count, 1, total_ms)
        if not items:
            return EvaluationResult(
                metric=self.name,
                score=None,
                label="not_applicable",
                explanation=(
                    "No materially evaluable context items were identified."
                ),
                details=details,
            )
        score = contextual_relevancy_score(
            [item["relevant"] for item in items]
        )
        return EvaluationResult(
            metric=self.name,
            score=score,
            label=contextual_relevancy_label(score),
            explanation=(
                f"{relevant_count} of {len(items)} retrieved context items are "
                "relevant to the information need."
            ),
            details=details,
        )

    def _empty_retrieval_result(self) -> EvaluationResult:
        details = {
            "item_count": 0,
            "relevant_count": 0,
            "judge_call_count": 0,
            "total_ms": 0.0,
            "verbose": self._verbose,
        }
        if self._verbose:
            details["items"] = []
        self._set_trace_attributes(0, 0, 0, 0.0)
        return EvaluationResult(
            metric=self.name,
            score=None,
            label="not_applicable",
            explanation="No documents were retrieved for the information need.",
            details=details,
        )

    def _set_trace_attributes(
        self, item_count: int, relevant_count: int, calls: int, total_ms: float
    ) -> None:
        tracing.set_current_span_attributes(
            {
                "contextual_relevancy.item_count": item_count,
                "contextual_relevancy.relevant_count": relevant_count,
                "contextual_relevancy.judge_call_count": calls,
                "contextual_relevancy.total_ms": total_ms,
            }
        )


class ContextualRecallEvaluator(Evaluator):
    """Fraction of query-relevant reference items captured by retrieval."""

    name = "contextual_recall"
    required_fields = ("input", "context")

    def __init__(
        self,
        llm=None,
        *,
        verbose: bool = False,
        document_text_key: str = DEFAULT_DOCUMENT_TEXT_KEY,
    ):
        _validate_document_text_key(document_text_key)
        self._llm = llm
        self._verbose = verbose
        self._document_text_key = document_text_key

    def _bind_judge(self, judge) -> None:
        if self._llm is None:
            self._llm = judge

    def resume_signature(self) -> dict:
        return {
            "contract_version": 1,
            "verbose": self._verbose,
            "document_text_key": self._document_text_key,
            "judge": self.judge_resume_signature(self._llm),
        }

    def validate_case(self, case: EvaluationCase) -> None:
        super().validate_case(case)
        _validate_documents_list(case, type(self).__name__)

    def _prepare(self, case: EvaluationCase) -> list[dict]:
        texts = _document_texts(
            case.retrieved_documents or [], self._document_text_key
        )
        return render_contextual_recall_prompt(
            render_value(case.input), render_value(case.context), texts
        )

    def evaluate(self, case: EvaluationCase) -> EvaluationResult:
        self.validate_case(case)
        if self._llm is None:
            raise ValueError(
                "ContextualRecallEvaluator needs a judge; pass one to the "
                "constructor or via EvaluationFramework(judge=...)."
            )
        started = time.monotonic()
        prompt = self._prepare(case)
        with tracing.judge_span(
            "contextual_recall.evaluate",
            {"idp_eval.metric": self.name, "idp_eval.stage": "evaluate"},
        ):
            response = self._llm.generate_object(
                prompt=prompt, schema=CONTEXTUAL_RECALL_SCHEMA_V1
            )
        return self._result_from_response(response, _elapsed_ms(started))

    async def a_evaluate(
        self, case: EvaluationCase, *, judge_limiter: asyncio.Semaphore
    ) -> EvaluationResult:
        self.validate_case(case)
        if self._llm is None:
            raise ValueError(
                "ContextualRecallEvaluator needs a judge; pass one to the "
                "constructor or via EvaluationFramework(judge=...)."
            )
        started = time.monotonic()
        prompt = self._prepare(case)
        async with judge_limiter:
            with tracing.judge_span(
                "contextual_recall.evaluate",
                {"idp_eval.metric": self.name, "idp_eval.stage": "evaluate"},
            ):
                async_generate = getattr(
                    self._llm, "async_generate_object", None
                )
                if callable(async_generate):
                    response = await async_generate(
                        prompt=prompt, schema=CONTEXTUAL_RECALL_SCHEMA_V1
                    )
                else:
                    response = await asyncio.to_thread(
                        self._llm.generate_object,
                        prompt=prompt,
                        schema=CONTEXTUAL_RECALL_SCHEMA_V1,
                    )
        return self._result_from_response(response, _elapsed_ms(started))

    def _validate_response(self, response: object) -> list[dict]:
        if not isinstance(response, dict) or set(response) != {"items"}:
            raise ValueError(
                "Malformed contextual recall response: expected only an `items` "
                "list."
            )
        raw_items = response["items"]
        if not isinstance(raw_items, list):
            raise ValueError(
                "Malformed contextual recall response: `items` must be a list."
            )
        validated = []
        expected_keys = {"reference_item", "captured", "reason"}
        for index, item in enumerate(raw_items, start=1):
            if not isinstance(item, dict) or set(item) != expected_keys:
                raise ValueError(
                    f"Malformed contextual recall item {index}: expected exactly "
                    f"{sorted(expected_keys)}."
                )
            reference_item = item["reference_item"]
            captured = item["captured"]
            reason = item["reason"]
            if not isinstance(reference_item, str) or not reference_item.strip():
                raise ValueError(
                    f"Malformed contextual recall item {index}: `reference_item` "
                    "must be a non-empty string."
                )
            if not isinstance(captured, bool):
                raise ValueError(
                    f"Malformed contextual recall item {index}: `captured` must "
                    "be a boolean."
                )
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(
                    f"Malformed contextual recall item {index}: `reason` must be "
                    "a non-empty string."
                )
            validated.append(
                {
                    "reference_item": " ".join(reference_item.split()),
                    "captured": captured,
                    "reason": reason,
                }
            )
        return validated

    def _result_from_response(
        self, response: object, total_ms: float
    ) -> EvaluationResult:
        items = self._validate_response(response)
        captured_count = sum(item["captured"] for item in items)
        details = {
            "reference_item_count": len(items),
            "captured_count": captured_count,
            "missing_count": len(items) - captured_count,
            "judge_call_count": 1,
            "total_ms": total_ms,
            "verbose": self._verbose,
        }
        if self._verbose:
            details["items"] = items
        self._set_trace_attributes(len(items), captured_count, total_ms)
        if not items:
            return EvaluationResult(
                metric=self.name,
                score=None,
                label="not_applicable",
                explanation=(
                    "No query-relevant reference items were identified in context."
                ),
                details=details,
            )
        score = contextual_recall_score([item["captured"] for item in items])
        return EvaluationResult(
            metric=self.name,
            score=score,
            label=contextual_recall_label(score),
            explanation=(
                f"Retrieval captured {captured_count} of {len(items)} relevant "
                "reference items."
            ),
            details=details,
        )

    def _set_trace_attributes(
        self, item_count: int, captured_count: int, total_ms: float
    ) -> None:
        tracing.set_current_span_attributes(
            {
                "contextual_recall.item_count": item_count,
                "contextual_recall.captured_count": captured_count,
                "contextual_recall.judge_call_count": 1,
                "contextual_recall.total_ms": total_ms,
            }
        )


def _first_relevant_rank(judgments: list[dict]) -> int | None:
    return next(
        (judgment["rank"] for judgment in judgments if judgment["relevant"]),
        None,
    )
