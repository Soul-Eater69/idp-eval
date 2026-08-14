"""Development-only coverage stability benchmark.

Runs the SAME case through the coverage evaluator many times against the
configured real judge and reports how stable the single-call result is (score
spread, requirement-count spread, and normalized-exact requirement overlap).

This is a manual developer tool. It is NOT part of the public evaluation API and
is NOT exercised by the unit test suite (the unit tests only cover the pure
``summarize_runs`` statistics via fake data). LLM behavior is not deterministic;
this tool is for *observing* stability, not asserting it.

Usage:
    python -m scripts.coverage_stability --runs 20
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass


@dataclass
class RunRecord:
    """One evaluation run's observable coverage outputs.

    Attributes:
        score: Aggregate coverage score, or ``None`` for a not-applicable result.
        requirements: The requirement texts the judge derived this run.
    """

    score: float | None
    requirements: list[str]


def _normalize(text: str) -> str:
    """Normalizes requirement text the same way the evaluator dedups."""
    return " ".join(text.lower().split())


def _mean_pairwise_overlap(run_requirement_sets: list[set[str]]) -> float | None:
    """Mean pairwise normalized-string Jaccard overlap across runs.

    Args:
        run_requirement_sets: One normalized requirement set per run.

    Returns:
        Mean Jaccard over all run pairs in ``[0, 1]``, or ``None`` when there is
        fewer than one comparable pair.
    """
    jaccards: list[float] = []
    for i in range(len(run_requirement_sets)):
        for j in range(i + 1, len(run_requirement_sets)):
            a, b = run_requirement_sets[i], run_requirement_sets[j]
            union = a | b
            if not union:
                continue
            jaccards.append(len(a & b) / len(union))
    if not jaccards:
        return None
    return sum(jaccards) / len(jaccards)


def summarize_runs(records: list[RunRecord]) -> dict:
    """Computes stability statistics over repeated runs of the same case.

    Args:
        records: Per-run coverage observations.

    Returns:
        A dict of stability statistics. Score statistics ignore not-applicable
        (``None``) scores; requirement-overlap uses normalized-exact matching.
    """
    scores = [r.score for r in records if r.score is not None]
    counts = [len(r.requirements) for r in records]
    overlap = _mean_pairwise_overlap(
        [{_normalize(t) for t in r.requirements} for r in records]
    )

    def _stats(values: list[float]) -> dict:
        if not values:
            return {"mean": None, "min": None, "max": None, "range": None, "stddev": None}
        return {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "range": max(values) - min(values),
            "stddev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        }

    return {
        "runs": len(records),
        "applicable_runs": len(scores),
        "score": _stats(scores),
        "requirement_count": _stats([float(c) for c in counts]),
        "mean_pairwise_requirement_overlap": overlap,
    }


def _sample_case():
    """Builds a small illustrative case (avoids importing at module load)."""
    from idp_eval import EvaluationCase

    return EvaluationCase(
        input="Generate a Jira Epic from the provided Theme.",
        context=(
            "Theme: Improve Customer Onboarding\n"
            "Goals:\n"
            "- Reduce onboarding time by 25%\n"
            "- Reduce abandoned registrations\n"
            "- Automate manual identity verification\n"
            "- Reduce manual verification effort by 40%\n"
            "- Give customers real-time onboarding status updates\n"
        ),
        output=(
            "Title: Improve Customer Onboarding\n"
            "Description: Streamline onboarding by automating identity "
            "verification and improving visibility into onboarding progress.\n"
            "Success Criteria:\n"
            "- Reduce onboarding time by 25%\n"
            "- Reduce manual verification effort\n"
            "- Customers can view onboarding status updates\n"
        ),
    )


def main() -> None:
    """Runs the same case ``--runs`` times against the real judge and reports."""
    parser = argparse.ArgumentParser(description="Coverage stability benchmark.")
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args()

    # Imported here so importing this module (for unit tests) needs no judge.
    from idp_eval import CoverageEvaluator, create_judge

    evaluator = CoverageEvaluator(llm=create_judge())
    case = _sample_case()

    records: list[RunRecord] = []
    for run in range(1, args.runs + 1):
        result = evaluator.evaluate(case)
        requirements = [item["requirement"] for item in result.details["items"]]
        records.append(RunRecord(score=result.score, requirements=requirements))
        print(f"Run {run}: requirements={len(requirements)} score={result.score}")

    summary = summarize_runs(records)
    print("\n=== Stability summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
