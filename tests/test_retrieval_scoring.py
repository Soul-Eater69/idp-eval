"""Deterministic retrieval scoring unit tests (hand-calculated, no LLM)."""

import math

import pytest

from idp_eval.scoring import (
    contextual_precision_at_k,
    contextual_recall_score,
    contextual_relevancy_score,
    dcg,
    hit_rate_at_k,
    hit_rate_at_k_label,
    mrr_at_k_label,
    ndcg_at_k,
    ndcg_at_k_label,
    reciprocal_rank_at_k,
    relevance_at_k,
    relevance_at_k_label,
)


@pytest.mark.parametrize(
    "values,expected",
    [
        ([1, 1, 1], 1.0),
        ([1, 0, 1, 0, 1], (1 + 2 / 3 + 3 / 5) / 3),
        ([0, 0, 1], 1 / 3),
        ([0, 0, 0], 0.0),
    ],
)
def test_contextual_precision_at_k(values, expected):
    assert contextual_precision_at_k(values) == pytest.approx(expected)


def test_contextual_precision_rewards_earlier_results_with_same_relevant_count():
    assert contextual_precision_at_k([1, 1, 0, 0]) > contextual_precision_at_k(
        [0, 0, 1, 1]
    )


@pytest.mark.parametrize(
    "helper",
    [contextual_precision_at_k, contextual_relevancy_score, contextual_recall_score],
)
def test_context_scoring_helpers_reject_empty_values(helper):
    with pytest.raises(ValueError, match="requires at least one"):
        helper([])


@pytest.mark.parametrize(
    "helper,values,expected",
    [
        (contextual_relevancy_score, [True, True, False], 2 / 3),
        (contextual_recall_score, [True, False, False], 1 / 3),
    ],
)
def test_context_fraction_helpers(helper, values, expected):
    assert helper(values) == pytest.approx(expected)


# --- Relevance@K (== Precision@K under binary relevance) ---------------------


@pytest.mark.parametrize(
    "scores,expected",
    [
        ([1, 1, 1], 1.0),
        ([1, 0, 1], 2 / 3),
        ([0, 0, 0], 0.0),
        ([1, 0], 0.5),
    ],
)
def test_relevance_at_k(scores, expected):
    assert relevance_at_k(scores) == pytest.approx(expected)


def test_relevance_at_k_empty_raises():
    with pytest.raises(ValueError, match="at least one relevance score"):
        relevance_at_k([])


def test_relevance_at_k_labels():
    assert relevance_at_k_label(1.0) == "all_relevant"
    assert relevance_at_k_label(0.5) == "partially_relevant"
    assert relevance_at_k_label(0.0) == "none_relevant"


# --- Hit Rate@K -------------------------------------------------------------


@pytest.mark.parametrize(
    "scores,expected",
    [([1], 1.0), ([0, 0, 1], 1.0), ([0, 0, 0], 0.0)],
)
def test_hit_rate_at_k(scores, expected):
    assert hit_rate_at_k(scores) == expected


def test_hit_rate_at_k_empty_raises():
    with pytest.raises(ValueError, match="at least one relevance score"):
        hit_rate_at_k([])


def test_hit_rate_labels():
    assert hit_rate_at_k_label(1.0) == "hit"
    assert hit_rate_at_k_label(0.0) == "miss"


# --- MRR@K (per-query reciprocal rank) -------------------------------------


@pytest.mark.parametrize(
    "scores,expected",
    [([1, 0, 1], 1.0), ([0, 0, 1], 1 / 3), ([0, 0, 0], 0.0)],
)
def test_reciprocal_rank_at_k(scores, expected):
    assert reciprocal_rank_at_k(scores) == pytest.approx(expected)


def test_reciprocal_rank_at_k_empty_raises():
    with pytest.raises(ValueError, match="at least one relevance score"):
        reciprocal_rank_at_k([])


def test_mrr_labels():
    assert mrr_at_k_label(1) == "first_result_relevant"
    assert mrr_at_k_label(3) == "relevant_found"
    assert mrr_at_k_label(None) == "no_relevant_result"


# --- nDCG@K (binary v1; hand-calculated references) --------------------------


def test_dcg_ranks_start_at_one():
    # rank 1 undiscounted (log2(2)=1), rank 2 -> /log2(3), rank 3 -> /log2(4).
    assert dcg([1, 1, 1]) == pytest.approx(1 + 1 / math.log2(3) + 1 / math.log2(4))


@pytest.mark.parametrize(
    "scores,expected_ndcg",
    [
        ([1, 1, 0, 0], 1.0),                    # perfectly ranked
        ([0, 0, 1, 1], 0.5706417189553201),     # poorly ranked
        ([0, 1, 0], 0.6309297535714575),        # one relevant, rank 2
        ([1, 0, 1], 0.9197207891481876),        # mixed
        ([1, 0], 1.0),                          # fewer docs, already ideal
        ([0, 0, 0], 0.0),                       # all irrelevant -> IDCG 0
    ],
)
def test_ndcg_at_k_values(scores, expected_ndcg):
    value, _dcg, _idcg = ndcg_at_k(scores)
    assert value == pytest.approx(expected_ndcg, abs=1e-12)


def test_ndcg_all_irrelevant_no_divide_by_zero():
    value, actual, ideal = ndcg_at_k([0, 0, 0])
    assert value == 0.0 and actual == 0.0 and ideal == 0.0


def test_ndcg_returns_dcg_and_idcg():
    value, actual, ideal = ndcg_at_k([0, 1, 1])
    assert actual == pytest.approx(1 / math.log2(3) + 1 / math.log2(4))
    assert ideal == pytest.approx(1 + 1 / math.log2(3))
    assert value == pytest.approx(actual / ideal)


def test_ndcg_at_k_empty_raises():
    with pytest.raises(ValueError, match="at least one relevance score"):
        ndcg_at_k([])


def test_ndcg_at_k_labels():
    assert ndcg_at_k_label(1.0) == "ideal_ranking"
    assert ndcg_at_k_label(0.5) == "suboptimal_ranking"
    assert ndcg_at_k_label(0.0) == "no_relevant_retrieved"
