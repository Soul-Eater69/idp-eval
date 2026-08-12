"""Deterministic scoring functions.

The judge LLM classifies semantics (supported / unsupported, covered / partial /
missing). Python turns those classifications into numbers. Never ask the LLM to
produce the final score directly.
"""

from __future__ import annotations

# Statuses that count as hallucinated when scoring hallucination.
HALLUCINATED_STATUSES = {"unsupported", "contradicted"}

# Semantic weights for input coverage.
COVERAGE_VALUES = {
    "covered": 1.0,
    "partial": 0.5,
    "missing": 0.0,
}


def calculate_hallucination_score(claims: list[dict]) -> float:
    """Calculates the unsupported-claim ratio.

    Args:
        claims: Claims classified with a ``"status"`` of ``supported``,
            ``unsupported``, or ``contradicted``.

    Returns:
        Fraction of claims that are unsupported or contradicted, in ``[0, 1]``.
        Returns ``0.0`` when there are no claims (nothing unsupported). Lower is
        better.
    """
    if not claims:
        return 0.0

    unsupported = sum(
        1 for claim in claims if claim["status"] in HALLUCINATED_STATUSES
    )
    return unsupported / len(claims)


def calculate_coverage(items: list[dict]) -> float:
    """Calculates semantic source coverage.

    Args:
        items: Source items classified with a ``"status"`` of ``covered``,
            ``partial``, or ``missing``.

    Returns:
        Coverage score between ``0`` and ``1``. Returns ``1.0`` when there are
        no important source items to cover. Higher is better.
    """
    if not items:
        return 1.0

    total = sum(COVERAGE_VALUES[item["status"]] for item in items)
    return total / len(items)


def score_to_label(score: float, high: float = 0.66, low: float = 0.33) -> str:
    """Buckets a ``[0, 1]`` score into ``high`` / ``medium`` / ``low``.

    Args:
        score: Score to bucket.
        high: Inclusive lower bound for the ``"high"`` bucket.
        low: Inclusive lower bound for the ``"medium"`` bucket.

    Returns:
        One of ``"high"``, ``"medium"``, or ``"low"``.
    """
    if score >= high:
        return "high"
    if score >= low:
        return "medium"
    return "low"
