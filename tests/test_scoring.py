"""Unit tests for the deterministic scoring functions (no LLM needed)."""

from idp_eval.scoring import calculate_coverage, score_to_label


def test_coverage_score():
    items = [
        {"status": "covered"},
        {"status": "covered"},
        {"status": "missing"},
    ]
    assert calculate_coverage(items) == 2 / 3


def test_coverage_with_partial():
    items = [
        {"status": "covered"},
        {"status": "covered"},
        {"status": "partial"},
        {"status": "missing"},
    ]
    assert calculate_coverage(items) == 2.5 / 4


def test_coverage_all_covered():
    items = [{"status": "covered"}, {"status": "covered"}]
    assert calculate_coverage(items) == 1.0


def test_coverage_all_missing():
    items = [{"status": "missing"}, {"status": "missing"}]
    assert calculate_coverage(items) == 0.0


def test_coverage_empty_is_full():
    assert calculate_coverage([]) == 1.0


def test_score_to_label():
    assert score_to_label(0.9) == "high"
    assert score_to_label(0.5) == "medium"
    assert score_to_label(0.1) == "low"
