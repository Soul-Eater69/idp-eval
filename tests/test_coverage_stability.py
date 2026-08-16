"""Unit tests for the pure stability-summary helper (no LLM involved)."""

from scripts.coverage_stability import RunRecord, summarize_runs


def test_summary_stable_runs():
    records = [
        RunRecord(score=0.6, requirements=["a", "b", "c"]),
        RunRecord(score=0.6, requirements=["a", "b", "c"]),
        RunRecord(score=0.6, requirements=["a", "b", "c"]),
    ]
    summary = summarize_runs(records)

    assert summary["runs"] == 3
    assert summary["applicable_runs"] == 3
    assert summary["score"]["mean"] == 0.6
    assert summary["score"]["range"] == 0.0
    assert summary["score"]["stddev"] == 0.0
    assert summary["requirement_count"]["mean"] == 3.0
    # Identical requirement sets -> perfect overlap.
    assert summary["mean_pairwise_requirement_overlap"] == 1.0


def test_summary_unstable_runs():
    records = [
        RunRecord(score=0.60, requirements=["a", "b", "c", "d", "e"]),
        RunRecord(score=0.50, requirements=["a", "b", "c", "d", "e", "f"]),
    ]
    summary = summarize_runs(records)

    assert summary["score"]["min"] == 0.50
    assert summary["score"]["max"] == 0.60
    assert round(summary["score"]["range"], 2) == 0.10
    assert summary["requirement_count"]["range"] == 1.0
    # 5 shared of 6 union -> 5/6 overlap.
    assert round(summary["mean_pairwise_requirement_overlap"], 4) == round(5 / 6, 4)


def test_summary_normalizes_requirements_for_overlap():
    records = [
        RunRecord(score=1.0, requirements=["Reduce Onboarding Time"]),
        RunRecord(score=1.0, requirements=[" reduce   onboarding time "]),
    ]
    summary = summarize_runs(records)
    assert summary["mean_pairwise_requirement_overlap"] == 1.0


def test_summary_ignores_not_applicable_scores():
    records = [
        RunRecord(score=None, requirements=[]),
        RunRecord(score=0.8, requirements=["a"]),
    ]
    summary = summarize_runs(records)
    assert summary["applicable_runs"] == 1
    assert summary["score"]["mean"] == 0.8
    # No statuses recorded -> consistency is not computable.
    assert summary["status_consistency"] is None


def test_status_consistency_across_runs():
    records = [
        RunRecord(score=0.5, requirements=["a", "b"],
                  statuses={"a": "covered", "b": "missing"}),
        RunRecord(score=0.5, requirements=["a", "b"],
                  statuses={"a": "covered", "b": "partial"}),
    ]
    # "a" agrees (covered/covered), "b" disagrees -> 1 of 2 consistent.
    assert summarize_runs(records)["status_consistency"] == 0.5
