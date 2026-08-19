"""Deterministic scoring functions.

The judge LLM classifies semantics (covered / partial / missing for coverage,
followed / violated for instruction adherence, relevant / unrelated per document
for retrieval). Python turns those classifications into numbers. Never ask the
LLM to produce the final score directly.
"""

from __future__ import annotations

import math

# Semantic weights for coverage.
COVERAGE_VALUES = {
    "covered": 1.0,
    "partial": 0.5,
    "missing": 0.0,
}

# Binary weights for instruction adherence.
INSTRUCTION_ADHERENCE_VALUES = {
    "followed": 1.0,
    "violated": 0.0,
}


def coverage_status_score(status: str) -> float:
    """Maps one coverage status to its deterministic numeric score.

    Args:
        status: One of ``covered``, ``partial``, or ``missing``.

    Returns:
        ``1.0`` / ``0.5`` / ``0.0`` respectively.

    Raises:
        ValueError: If ``status`` is not a recognized coverage status. Unknown
            statuses fail loudly rather than silently receiving a score.
    """
    try:
        return COVERAGE_VALUES[status]
    except KeyError:
        raise ValueError(f"Unknown coverage status: {status!r}") from None


def coverage_status_from_binary(
    meaningfully_present: bool, fully_present: bool
) -> str:
    """Derives the three-way coverage status from two binary judgments.

    The judge returns only the two booleans; Python derives the category:

        not meaningfully_present            -> "missing"
        meaningfully_present, fully_present -> "covered"
        meaningfully_present, not full      -> "partial"

    Args:
        meaningfully_present: Whether any meaningful part of the requirement is
            represented in the output.
        fully_present: Whether the full material requirement (including
            qualifiers) is represented.

    Returns:
        One of ``"covered"``, ``"partial"``, or ``"missing"``.

    Raises:
        ValueError: For the logically inconsistent combination
            ``fully_present=True`` with ``meaningfully_present=False``.
    """
    if fully_present and not meaningfully_present:
        raise ValueError(
            "Invalid coverage classification: fully_present=True requires "
            "meaningfully_present=True."
        )
    if not meaningfully_present:
        return "missing"
    if fully_present:
        return "covered"
    return "partial"


def calculate_coverage(items: list[dict]) -> float:
    """Aggregates item-level coverage classifications into a single score.

    Each item's ``"status"`` is mapped deterministically to a numeric score and
    the mean is returned. The LLM never produces this number.

    Args:
        items: Requirements/items classified with a ``"status"`` of ``covered``,
            ``partial``, or ``missing``.

    Returns:
        Coverage score between ``0`` and ``1``. Higher is better.

    Raises:
        ValueError: If ``items`` is empty or any item carries an unrecognized
            status. The evaluator handles an empty judge result as
            not-applicable before calling this helper.
    """
    if not items:
        raise ValueError(
            "At least one source item is required to calculate coverage."
        )

    total = sum(coverage_status_score(item["status"]) for item in items)
    return total / len(items)


def calculate_instruction_adherence(instructions: list[dict]) -> float:
    """Calculates the binary instruction-adherence score.

    Args:
        instructions: Instructions classified as ``followed`` or ``violated``.

    Returns:
        Score between ``0`` and ``1`` equal to the fraction followed.

    Raises:
        ValueError: If the list is empty or a status is not recognized.
    """
    if not instructions:
        raise ValueError(
            "At least one instruction is required to calculate instruction "
            "adherence."
        )

    try:
        total = sum(
            INSTRUCTION_ADHERENCE_VALUES[item["status"]]
            for item in instructions
        )
    except KeyError as exc:
        status = exc.args[0]
        raise ValueError(
            f"Unknown instruction-adherence status: {status!r}"
        ) from None
    return total / len(instructions)


def coverage_label(score: float) -> str:
    """Descriptive coverage label derived directly from the covered fraction.

    The label restates what the score already means — the boundaries are the
    only two defensible ones (everything vs. nothing covered), not arbitrary
    thresholds:

        1.0            -> "complete"    (every item represented)
        0 < score < 1  -> "incomplete"  (some items missing/partial)
        0.0            -> "missing"     (nothing represented)

    Not-applicable results (no score) are labeled ``"not_applicable"`` by the
    evaluator and never reach this function.
    """
    if score >= 1.0:
        return "complete"
    if score <= 0.0:
        return "missing"
    return "incomplete"


def instruction_adherence_label(score: float) -> str:
    """Descriptive adherence label derived from the fraction of instructions
    followed.

        1.0            -> "fully_followed"
        0 < score < 1  -> "violations_present"  (at least one violated)
        0.0            -> "violated"

    Unlike a generic high/medium/low bucket, this never calls a result with an
    explicit violation "high": any violation yields ``"violations_present"``.
    """
    if score >= 1.0:
        return "fully_followed"
    if score <= 0.0:
        return "violated"
    return "violations_present"


# --- retrieval metrics ------------------------------------------------------
#
# Both retrieval metrics consume the SAME per-document relevance scores (one per
# top-K document, in rank order). The LLM produces only per-document relevance;
# these functions turn that into the metric numbers. The scores may be binary
# (Phoenix's DocumentRelevanceEvaluator is binary today) or graded in ``[0, 1]``
# — the math below works for both, so graded relevance needs no formula change.


def relevance_at_k(relevance_scores: list[float]) -> float:
    """Mean relevance over the already-sliced top-K documents.

    ``relevance_scores`` must already be the top-``effective_k`` scores in rank
    order. Under binary relevance this is exactly Precision@K
    (relevant count / evaluated top-K count).

    Raises:
        ValueError: If the list is empty (the caller handles zero documents as
            not-applicable, so this should never be reached with an empty list).
    """
    if not relevance_scores:
        raise ValueError("relevance_at_k requires at least one relevance score.")
    return sum(relevance_scores) / len(relevance_scores)


def dcg(relevance_scores: list[float]) -> float:
    """Discounted cumulative gain with ranks starting at 1.

    ``DCG = sum(rel_i / log2(rank_i + 1))`` so the rank-1 document is undiscounted
    (``log2(2) = 1``). Works for binary or graded relevance.
    """
    return sum(
        rel / math.log2(rank + 1)
        for rank, rel in enumerate(relevance_scores, start=1)
    )


def ndcg_at_k(relevance_scores: list[float]) -> tuple[float, float, float]:
    """Normalized DCG over the already-sliced top-K scores.

    Returns ``(ndcg, dcg, idcg)``. IDCG is the DCG of the same scores sorted
    descending (the ideal ranking). When IDCG is 0 (no retrieved document was
    relevant) nDCG is defined as ``0.0`` rather than raising on divide-by-zero.

    Raises:
        ValueError: If the list is empty (zero documents is handled upstream as
            not-applicable).
    """
    if not relevance_scores:
        raise ValueError("ndcg_at_k requires at least one relevance score.")
    actual = dcg(relevance_scores)
    ideal = dcg(sorted(relevance_scores, reverse=True))
    if ideal == 0:
        return 0.0, actual, ideal
    return actual / ideal, actual, ideal


def relevance_at_k_label(score: float) -> str:
    """Descriptive Relevance@K label derived from the fraction relevant.

    ``1.0`` -> "all_relevant", ``0.0`` -> "none_relevant", between ->
    "partially_relevant". Not-applicable (no documents) is labeled by the
    evaluator.
    """
    if score >= 1.0:
        return "all_relevant"
    if score <= 0.0:
        return "none_relevant"
    return "partially_relevant"


def ndcg_at_k_label(score: float) -> str:
    """Descriptive nDCG@K label derived from the ranking quality.

    ``1.0`` -> "ideal_ranking" (relevant docs ranked first), ``0.0`` ->
    "no_relevant_retrieved" (IDCG is 0), between -> "suboptimal_ranking".
    """
    if score >= 1.0:
        return "ideal_ranking"
    if score <= 0.0:
        return "no_relevant_retrieved"
    return "suboptimal_ranking"
