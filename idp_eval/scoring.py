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

# Semantic weights for instruction adherence. ``not_applicable`` is deliberately
# absent: those instructions are excluded from scoring, not weighted.
INSTRUCTION_ADHERENCE_VALUES = {
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


def calculate_instruction_adherence(instructions: list[dict]) -> float:
    """Calculates the instruction-adherence score.

    Instructions classified ``not_applicable`` are excluded before scoring: they
    are neither a success nor a failure, so they must not affect the denominator.

    Args:
        instructions: Instructions classified with a ``"status"`` of
            ``followed``, ``partial``, ``violated``, or ``not_applicable``.

    Returns:
        Score between ``0`` and ``1`` equal to the mean of the applicable
        instructions' status values. Higher is better.

    Raises:
        ValueError: If no applicable instruction remains after excluding
            ``not_applicable`` ones. The evaluator is responsible for handling
            that case as not-applicable before calling this function.
    """
    applicable = [
        i for i in instructions if i["status"] != "not_applicable"
    ]
    if not applicable:
        raise ValueError(
            "At least one applicable instruction is required to calculate "
            "instruction adherence."
        )

    total = sum(INSTRUCTION_ADHERENCE_VALUES[i["status"]] for i in applicable)
    return total / len(applicable)


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
