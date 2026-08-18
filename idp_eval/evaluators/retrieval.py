"""Retrieval evaluators: Relevance@K and nDCG@K over ranked retrieved documents.

Both metrics judge each retrieved document's relevance to the query **once**,
using Phoenix's modern ``DocumentRelevanceEvaluator`` (binary relevant/unrelated
per document), and then compute their scores deterministically in Python:

- ``RelevanceAtKEvaluator(k)`` -> metric ``relevance_at_{k}``: fraction of the top
  effective-K documents that are relevant. Under binary relevance this is exactly
  **Precision@K**.
- ``NDCGAtKEvaluator(k)`` -> metric ``ndcg_at_{k}``: ranking quality from the same
  per-document relevance judgments (no extra LLM call). v1 is **binary-relevance
  nDCG** because ``DocumentRelevanceEvaluator`` is binary; the internal math also
  accepts graded ``[0, 1]`` relevance, so graded relevance can be added later
  without changing the public evaluators.

Sharing: when several retrieval metrics run for one case, a single
:class:`_RetrievalRelevancePass` judges documents up to the deepest required rank
once and every retrieval metric reads from it (see
:mod:`idp_eval.framework`). The pass is internal and not exported.

The retrieval query is ``EvaluationCase.input``; the ranked documents are
``EvaluationCase.retrieved_documents`` (list order = rank). No generated output is
required. The retrieval similarity ``score`` is kept as diagnostics only and is
never sent to the relevance judge.
"""

from __future__ import annotations

import asyncio

from idp_eval import tracing
from idp_eval.models import EvaluationCase, EvaluationResult, Evaluator
from idp_eval.rendering import render_value
from idp_eval.scoring import (
    ndcg_at_k,
    ndcg_at_k_label,
    relevance_at_k,
    relevance_at_k_label,
)

DEFAULT_DOCUMENT_TEXT_KEY = "text"


def _validate_k(k: int) -> None:
    """Rejects a non-positive-integer ``k`` (``bool`` is not an int here)."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}.")


def _document_text(document, text_key: str, rank: int) -> str:
    """Extracts the judged text from a document (string or mapping).

    Only the text is sent to the relevance judge — never the similarity score or
    other metadata.
    """
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
    """Returns the document id if present (``document_id`` or ``id``)."""
    if isinstance(document, dict):
        value = document.get("document_id", document.get("id"))
        return value if isinstance(value, str) else None
    return None


def _retrieval_score(document):
    """Returns the retrieval similarity score metadata if present."""
    if isinstance(document, dict):
        return document.get("score")
    return None


def _build_document_relevance_evaluator(llm):
    """Builds Phoenix's modern ``DocumentRelevanceEvaluator`` (patchable in tests).

    Imported lazily so the rest of the framework works without Phoenix installed.
    """
    from phoenix.evals.metrics import DocumentRelevanceEvaluator

    return DocumentRelevanceEvaluator(llm=llm)


class _DocumentRelevanceJudge:
    """Thin wrapper over Phoenix ``DocumentRelevanceEvaluator`` (one doc at a time).

    Sends ``{"input": query, "document_text": text}`` only (the Phoenix
    DocumentRelevanceEvaluator contract); returns Phoenix's own score/label/
    explanation without reinterpreting the label. Phoenix's binary semantics are
    relevant -> 1.0, unrelated -> 0.0.
    """

    def __init__(self, llm):
        self._evaluator = _build_document_relevance_evaluator(llm)

    def judge(self, query: str, document_text: str) -> dict:
        # Phoenix DocumentRelevanceEvaluator contract: the query is ``input`` and
        # the document text is ``document_text``. Only these two fields are sent —
        # never document_id, similarity score, or other retrieval metadata.
        result = self._evaluator.evaluate(
            {"input": query, "document_text": document_text}
        )[0]
        score = result.score
        label = getattr(result, "label", None)
        if score is None:
            # Fall back to Phoenix's label only if it gave no numeric score.
            score = 1.0 if str(label).strip().lower() == "relevant" else 0.0
        return {
            "score": float(score),
            "label": label,
            "explanation": getattr(result, "explanation", None),
        }


class _RetrievalRelevancePass:
    """One shared per-case relevance pass over the ranked documents.

    Judges documents ``[0, depth)`` once (lazily building the Phoenix judge), in
    rank order, and caches the per-document judgments so multiple retrieval
    metrics (and different K values) reuse them. Not exported.
    """

    def __init__(self, judge_factory, query, documents, text_key, verbose):
        self._judge_factory = judge_factory   # zero-arg; built on first judge
        self._judge = None
        self._query = query
        self._documents = documents
        self._text_key = text_key
        self._verbose = verbose
        self._judgments: list[dict] = []

    def run(self, depth: int) -> None:
        """Synchronously judges documents up to ``depth`` (serial)."""
        depth = min(depth, len(self._documents))
        if depth <= len(self._judgments):
            self._set_trace_attributes()
            return
        with tracing.judge_span(
            "retrieval.relevance", {"idp_eval.stage": "relevance"}
        ):
            for index in range(len(self._judgments), depth):
                self._judgments.append(self._judge_one(index))
        self._set_trace_attributes()

    async def a_run(self, depth: int, limiter: asyncio.Semaphore) -> None:
        """Judges documents up to ``depth`` concurrently under ``limiter``."""
        depth = min(depth, len(self._documents))
        if depth <= len(self._judgments):
            self._set_trace_attributes()
            return

        async def judge_index(index: int) -> dict:
            async with limiter:
                return await asyncio.to_thread(self._judge_one, index)

        with tracing.judge_span(
            "retrieval.relevance", {"idp_eval.stage": "relevance"}
        ):
            self._judgments = list(
                await asyncio.gather(*(judge_index(i) for i in range(depth)))
            )
        self._set_trace_attributes()

    def _judge_one(self, index: int) -> dict:
        document = self._documents[index]
        rank = index + 1
        text = _document_text(document, self._text_key, rank)
        if self._judge is None:
            self._judge = self._judge_factory()
        with tracing.judge_span(
            "retrieval.relevance.document",
            {"idp_eval.stage": "relevance", "retrieval.rank": rank},
        ):
            judgment = self._judge.judge(self._query, text)
        record = {
            "rank": rank,
            "document_id": _document_id(document),
            "relevance_score": judgment["score"],
            "relevance_label": judgment["label"],
            "explanation": judgment["explanation"],
            "retrieval_score": _retrieval_score(document),
        }
        if self._verbose:
            record["text"] = text
        return record

    def judgments(self, depth: int) -> list[dict]:
        """Returns the cached judgments for the top ``depth`` documents."""
        return self._judgments[:depth]

    def _set_trace_attributes(self) -> None:
        relevant = sum(1 for j in self._judgments if j["relevance_score"] > 0)
        tracing.set_current_span_attributes(
            {
                "retrieval.document_count": len(self._documents),
                "retrieval.judged_count": len(self._judgments),
                "retrieval.relevant_count": relevant,
            }
        )


class _RetrievalEvaluator(Evaluator):
    """Shared base for retrieval metrics over ranked retrieved documents.

    Configured by subclasses via :meth:`_score`. Requires ``input`` (the query)
    and ``retrieved_documents`` (a list; may be empty -> not_applicable). Never
    requires ``context`` / ``output`` / ``instructions``.
    """

    uses_retrieval_relevance = True
    required_fields = ("input",)
    # Class-level placeholder satisfies the abstract ``name``; each instance sets
    # its own ``relevance_at_{k}`` / ``ndcg_at_{k}`` in ``__init__``.
    name = "retrieval"

    def __init__(
        self,
        k: int,
        llm=None,
        *,
        verbose: bool = False,
        document_text_key: str = DEFAULT_DOCUMENT_TEXT_KEY,
    ):
        """Args:
            k: Positive integer cutoff rank.
            llm: Shared judge; may be ``None`` at construction and injected by the
                framework (``EvaluationFramework(evaluators=[...], judge=judge)``).
            verbose: When ``True``, include each document's text in result
                details (off by default to avoid duplicating large payloads).
            document_text_key: Mapping key holding a document's text.

        Raises:
            ValueError: If ``k`` is not a positive integer.
        """
        _validate_k(k)
        self._k = k
        self._llm = llm
        self._verbose = verbose
        self._document_text_key = document_text_key
        self._relevance_judge_cache: _DocumentRelevanceJudge | None = None

    # --- judge wiring (shared judge; injected by the framework if unset) -----

    def _bind_judge(self, judge) -> None:
        """Binds the shared judge when none was provided at construction."""
        if self._llm is None:
            self._llm = judge

    def _relevance_judge(self) -> _DocumentRelevanceJudge:
        if self._relevance_judge_cache is None:
            if self._llm is None:
                raise ValueError(
                    f"{type(self).__name__} needs a judge; pass one to the "
                    "constructor or via EvaluationFramework(judge=...)."
                )
            self._relevance_judge_cache = _DocumentRelevanceJudge(self._llm)
        return self._relevance_judge_cache

    # --- validation ---------------------------------------------------------

    def validate_case(self, case: EvaluationCase) -> None:
        """Requires a non-empty ``input`` and a ``retrieved_documents`` list.

        An empty list is allowed and yields a not-applicable result (no judge
        calls); ``None`` or a non-list value is a validation error.
        """
        super().validate_case(case)  # input required and non-empty
        documents = case.retrieved_documents
        if documents is None or not isinstance(documents, list):
            raise ValueError(
                f"{type(self).__name__} requires `retrieved_documents` as a "
                "list (rank order); got "
                f"{type(documents).__name__ if documents is not None else 'None'}."
            )

    # --- shared-pass hooks (used by the framework) --------------------------

    def _documents(self, case: EvaluationCase) -> list:
        return case.retrieved_documents or []

    def _relevance_depth(self, case: EvaluationCase) -> int:
        """effective_k = min(k, number_of_documents)."""
        return min(self._k, len(self._documents(case)))

    def _build_relevance_pass(self, case: EvaluationCase) -> _RetrievalRelevancePass:
        return _RetrievalRelevancePass(
            self._relevance_judge,
            render_value(case.input),
            self._documents(case),
            self._document_text_key,
            self._verbose,
        )

    def evaluate_shared(
        self, case: EvaluationCase, relevance_pass: _RetrievalRelevancePass
    ) -> EvaluationResult:
        """Computes the metric from an already-run shared relevance pass.

        Deterministic: no judge calls happen here (the pass judged the documents).
        """
        effective_k = self._relevance_depth(case)
        document_count = len(self._documents(case))
        if document_count == 0:
            return self._not_applicable_result()
        judgments = relevance_pass.judgments(effective_k)
        return self._score(case, judgments, effective_k, document_count)

    # --- standalone entry points (direct use, own relevance pass) -----------

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

    # --- subclass responsibilities ------------------------------------------

    def _score(self, case, judgments, effective_k, document_count):
        raise NotImplementedError

    def _not_applicable_result(self) -> EvaluationResult:
        raise NotImplementedError

    def _documents_diagnostic(self, judgments: list[dict]) -> list[dict]:
        """Per-document diagnostics for result details (no text unless verbose)."""
        return judgments


class RelevanceAtKEvaluator(_RetrievalEvaluator):
    """Fraction of the top-K retrieved documents that are relevant to the query.

    Metric name: ``relevance_at_{k}``. Under binary relevance this equals
    Precision@K. Higher is better.
    """

    def __init__(self, k: int, llm=None, **kwargs):
        super().__init__(k, llm, **kwargs)
        self.name = f"relevance_at_{k}"

    def _score(self, case, judgments, effective_k, document_count):
        scores = [j["relevance_score"] for j in judgments]
        value = relevance_at_k(scores)
        relevant_count = sum(1 for s in scores if s > 0)
        return EvaluationResult(
            metric=self.name,
            score=value,
            label=relevance_at_k_label(value),
            explanation=(
                f"{relevant_count} of the top {effective_k} retrieved documents "
                f"are relevant (Relevance@{self._k}"
                + (f", {document_count} retrieved" if document_count != effective_k
                   else "")
                + ")."
            ),
            details={
                "requested_k": self._k,
                "effective_k": effective_k,
                "document_count": document_count,
                "relevant_count": relevant_count,
                "documents": self._documents_diagnostic(judgments),
            },
        )

    def _not_applicable_result(self) -> EvaluationResult:
        return EvaluationResult(
            metric=self.name,
            score=None,
            label="not_applicable",
            explanation="No documents were retrieved for the query.",
            details={
                "requested_k": self._k,
                "effective_k": 0,
                "document_count": 0,
                "relevant_count": 0,
                "documents": [],
            },
        )


class NDCGAtKEvaluator(_RetrievalEvaluator):
    """Ranking quality of relevant documents within the top K (binary nDCG v1).

    Metric name: ``ndcg_at_{k}``. Uses the same per-document relevance judgments
    as Relevance@K (no extra LLM call) and computes nDCG deterministically.
    Higher is better.
    """

    def __init__(self, k: int, llm=None, **kwargs):
        super().__init__(k, llm, **kwargs)
        self.name = f"ndcg_at_{k}"

    def _score(self, case, judgments, effective_k, document_count):
        scores = [j["relevance_score"] for j in judgments]
        value, dcg_value, idcg_value = ndcg_at_k(scores)
        return EvaluationResult(
            metric=self.name,
            score=value,
            label=ndcg_at_k_label(value),
            explanation=(
                f"nDCG@{self._k} = {round(value, 4):g} over the top {effective_k} "
                f"retrieved documents (DCG {round(dcg_value, 4):g} / IDCG "
                f"{round(idcg_value, 4):g})."
            ),
            details={
                "requested_k": self._k,
                "effective_k": effective_k,
                "document_count": document_count,
                "relevance_scores": scores,
                "dcg": dcg_value,
                "idcg": idcg_value,
                "documents": self._documents_diagnostic(judgments),
            },
        )

    def _not_applicable_result(self) -> EvaluationResult:
        return EvaluationResult(
            metric=self.name,
            score=None,
            label="not_applicable",
            explanation="No documents were retrieved for the query.",
            details={
                "requested_k": self._k,
                "effective_k": 0,
                "document_count": 0,
                "relevance_scores": [],
                "dcg": 0.0,
                "idcg": 0.0,
                "documents": [],
            },
        )
