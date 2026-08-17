"""Offline tests for retained production-coverage benchmark helpers."""

from scripts.coverage_benchmark import estimate_calls, status_agreement
from scripts.coverage_benchmark_utils import (
    exact_set_summary,
    load_cases,
    normalize,
    numeric_stats,
)


def test_normalize_and_exact_set_summary():
    assert normalize(" Reduce   Onboarding TIME ") == "reduce onboarding time"
    summary = exact_set_summary(
        [["Requirement A", "Requirement B"], [" requirement   a", "Requirement B"]]
    )
    assert summary["count_per_run"] == [2, 2]
    assert summary["mean_exact_jaccard"] == 1.0


def test_numeric_stats():
    stats = numeric_stats([0.5, 0.75, 1.0])
    assert stats["mean"] == 0.75
    assert stats["median"] == 0.75
    assert stats["min"] == 0.5
    assert stats["max"] == 1.0
    assert stats["range"] == 0.5
    assert stats["stddev"] > 0
    assert numeric_stats([])["mean"] is None


def test_estimate_calls():
    assert estimate_calls(3, 4, False) == {
        "extraction": 12,
        "classification": 12,
        "end_to_end": 0,
        "total": 24,
    }
    assert estimate_calls(3, 4, True)["total"] == 48


def test_status_agreement():
    runs = [
        {"r1": {"status": "covered"}, "r2": {"status": "partial"}},
        {"r1": {"status": "covered"}, "r2": {"status": "missing"}},
        {"r1": {"status": "covered"}, "r2": {"status": "partial"}},
    ]
    result = status_agreement(runs, ["r1", "r2"])
    assert result["per_requirement"]["r1"]["agreement"] == 1.0
    assert result["per_requirement"]["r2"]["agreement"] == 2 / 3


def test_gt_fixture_is_valid_benchmark_input():
    assert len(load_cases("gt.json")) == 10


def test_general_fixture_has_ten_cases():
    assert len(load_cases("benchmarks/coverage_cases.json")) == 10
