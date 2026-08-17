"""Deterministic scoring functions.

The judge LLM classifies semantics (covered / partial / missing for coverage,
followed / violated for instruction adherence). Python turns those
classifications into numbers. Never ask the LLM to produce the final score
directly.
"""

from __future__ import annotations

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
        Coverage score between ``0`` and ``1``. Returns ``1.0`` when there are
        no task-relevant requirements to cover. Higher is better.

    Raises:
        ValueError: If any item carries an unrecognized status.
    """
    if not items:
        return 1.0

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
