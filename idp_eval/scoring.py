"""Deterministic scoring functions.

The judge LLM classifies semantics (covered / partial / missing, followed /
partial / violated). Python turns those classifications into numbers. Never ask
the LLM to produce the final score directly.
"""

from __future__ import annotations

# Semantic weights for coverage.
COVERAGE_VALUES = {
    "covered": 1.0,
    "partial": 0.5,
    "missing": 0.0,
}

# Semantic weights for instruction following.
INSTRUCTION_FOLLOWING_VALUES = {
    "followed": 1.0,
    "partial": 0.5,
    "violated": 0.0,
}


def calculate_coverage(items: list[dict]) -> float:
    """Calculates semantic source coverage.

    Args:
        items: Source items classified with a ``"status"`` of ``covered``,
            ``partial``, or ``missing``.

    Returns:
        Coverage score between ``0`` and ``1``. Returns ``1.0`` when there are
        no relevant source items to cover. Higher is better.
    """
    if not items:
        return 1.0

    total = sum(COVERAGE_VALUES[item["status"]] for item in items)
    return total / len(items)


def calculate_instruction_following(instructions: list[dict]) -> float:
    """Calculates the instruction-following score.

    Args:
        instructions: Instructions classified with a ``"status"`` of
            ``followed``, ``partial``, or ``violated``. Must be non-empty; the
            evaluator handles the "no instructions supplied" case separately as
            not-applicable, so this function is never called with an empty list.

    Returns:
        Score between ``0`` and ``1`` equal to the mean of the per-instruction
        status values. Higher is better.
    """
    total = sum(INSTRUCTION_FOLLOWING_VALUES[i["status"]] for i in instructions)
    return total / len(instructions)


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
