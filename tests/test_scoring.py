"""Unit tests for the deterministic scoring functions (no LLM needed)."""

import pytest

from idp_eval.scoring import (
    calculate_coverage,
    calculate_instruction_adherence,
    coverage_status_score,
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


def test_coverage_all_covered():
    items = [{"status": "covered"}, {"status": "covered"}]
    assert calculate_coverage(items) == 1.0


def test_coverage_all_missing():
    items = [{"status": "missing"}, {"status": "missing"}]
    assert calculate_coverage(items) == 0.0


def test_coverage_single_statuses():
    assert calculate_coverage([{"status": "covered"}]) == 1.0
    assert calculate_coverage([{"status": "partial"}]) == 0.5
    assert calculate_coverage([{"status": "missing"}]) == 0.0


def test_coverage_covered_and_missing():
    assert calculate_coverage(
        [{"status": "covered"}, {"status": "missing"}]
    ) == 0.5


def test_coverage_mixed_four():
    items = [
        {"status": "covered"},
        {"status": "covered"},
        {"status": "partial"},
        {"status": "missing"},
    ]
    assert calculate_coverage(items) == 0.625


def test_coverage_empty_is_full():
    assert calculate_coverage([]) == 1.0


def test_coverage_status_score_mapping():
    assert coverage_status_score("covered") == 1.0
    assert coverage_status_score("partial") == 0.5
    assert coverage_status_score("missing") == 0.0


def test_coverage_unknown_status_raises():
    with pytest.raises(ValueError, match="Unknown coverage status"):
        coverage_status_score("mostly")
    with pytest.raises(ValueError, match="Unknown coverage status"):
        calculate_coverage([{"status": "covered"}, {"status": "bogus"}])


def test_instruction_adherence_mixed():
    instructions = [
        {"status": "followed"},
        {"status": "followed"},
        {"status": "partial"},
        {"status": "violated"},
    ]
    assert calculate_instruction_adherence(instructions) == 0.625


def test_instruction_adherence_all_followed():
    instructions = [{"status": "followed"}, {"status": "followed"}]
    assert calculate_instruction_adherence(instructions) == 1.0


def test_instruction_adherence_all_violated():
    instructions = [{"status": "violated"}, {"status": "violated"}]
    assert calculate_instruction_adherence(instructions) == 0.0


def test_score_to_label():
    assert score_to_label(0.9) == "high"
    assert score_to_label(0.5) == "medium"
    assert score_to_label(0.1) == "low"
