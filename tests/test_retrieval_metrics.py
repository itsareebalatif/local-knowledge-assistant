from __future__ import annotations

import pytest

from app.evaluation.retrieval_metrics import (
    context_precision_at_k,
    context_recall,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k_counts_all_relevant_found_within_k():
    retrieved = [5, 1, 2, 3, 4]
    relevant = {1, 2, 99}  # 99 is never retrieved
    assert recall_at_k(retrieved, relevant, k=3) == 2 / 3  # 1 and 2 found within top-3, 99 missed


def test_recall_at_k_respects_k_boundary():
    retrieved = [1, 2, 3, 4, 5]
    relevant = {5}
    assert recall_at_k(retrieved, relevant, k=3) == 0.0  # 5 only shows up at rank 5
    assert recall_at_k(retrieved, relevant, k=5) == 1.0


def test_recall_at_k_with_no_relevant_ids_is_trivially_satisfied():
    assert recall_at_k([1, 2, 3], set(), k=3) == 1.0


def test_precision_at_k_counts_hits_among_top_k():
    retrieved = [1, 99, 2, 99, 99]
    relevant = {1, 2}
    assert precision_at_k(retrieved, relevant, k=4) == 0.5  # 2 of top-4 are relevant


def test_precision_at_k_with_empty_retrieval_is_zero():
    assert precision_at_k([], {1, 2}, k=5) == 0.0


def test_reciprocal_rank_of_first_hit():
    assert reciprocal_rank([9, 9, 1, 9], {1}) == 1 / 3


def test_reciprocal_rank_is_zero_when_nothing_relevant_found():
    assert reciprocal_rank([9, 9, 9], {1}) == 0.0


def test_context_precision_is_perfect_when_every_relevant_item_comes_first():
    # Best possible ordering: both relevant items as early as they can be.
    assert context_precision_at_k([True, True, False]) == 1.0


def test_context_precision_penalizes_relevant_items_buried_lower():
    # One relevant item at rank 1 (Precision@1=1.0) vs. the same single
    # relevant item at rank 3 (Precision@3=1/3) — position matters.
    assert context_precision_at_k([True, False, False]) == 1.0
    assert context_precision_at_k([False, False, True]) == pytest.approx(1 / 3)


def test_context_precision_is_zero_when_nothing_is_relevant():
    assert context_precision_at_k([False, False, False]) == 0.0


def test_context_precision_is_zero_for_an_empty_list():
    assert context_precision_at_k([]) == 0.0


def test_context_precision_averages_correctly_over_multiple_relevant_items():
    # Relevant at ranks 1 and 3: Precision@1=1.0, Precision@3=2/3 -> mean = 5/6.
    assert context_precision_at_k([True, False, True]) == pytest.approx((1.0 + 2 / 3) / 2)


def test_context_recall_is_the_plain_proportion_of_attributable_claims():
    assert context_recall([True, True, False, False]) == 0.5
    assert context_recall([True, True, True]) == 1.0
    assert context_recall([False, False]) == 0.0


def test_context_recall_is_vacuously_perfect_for_an_empty_claim_list():
    assert context_recall([]) == 1.0


def test_reciprocal_rank_rewards_earlier_hits_more():
    assert reciprocal_rank([1, 9, 9], {1}) > reciprocal_rank([9, 1, 9], {1})
