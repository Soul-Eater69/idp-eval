"""Unit tests for the deterministic scoring functions (no LLM needed)."""

from idp_eval.scoring import (
    calculate_coverage,
    calculate_hallucination_score,
    score_to_label,
)


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


def test_coverage_empty_is_full():
    assert calculate_coverage([]) == 1.0


def test_hallucination_score():
    claims = [
        {"status": "supported"},
        {"status": "supported"},
        {"status": "supported"},
        {"status": "supported"},
        {"status": "unsupported"},
    ]
    assert calculate_hallucination_score(claims) == 0.20


def test_hallucination_counts_contradicted():
    claims = [
        {"status": "supported"},
        {"status": "contradicted"},
    ]
    assert calculate_hallucination_score(claims) == 0.5


def test_hallucination_empty_is_zero():
    assert calculate_hallucination_score([]) == 0.0


def test_score_to_label():
    assert score_to_label(0.9) == "high"
    assert score_to_label(0.5) == "medium"
    assert score_to_label(0.1) == "low"
