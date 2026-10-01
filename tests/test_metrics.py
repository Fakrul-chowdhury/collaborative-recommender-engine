"""Metric functions checked against values worked out by hand."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.evaluation.evaluate import bootstrap_lift_interval, relative_lift
from src.evaluation.metrics import (
    catalog_coverage,
    intra_list_diversity,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_recall_at_k_hand_computed():
    recommended = ["a", "b", "c", "d", "e"]
    relevant = {"b", "e", "z"}
    # top-3 contains only "b": 1 of 3 relevant items.
    assert recall_at_k(recommended, relevant, 3) == pytest.approx(1 / 3)
    # top-5 contains "b" and "e": 2 of 3.
    assert recall_at_k(recommended, relevant, 5) == pytest.approx(2 / 3)
    # k larger than the list changes nothing.
    assert recall_at_k(recommended, relevant, 10) == pytest.approx(2 / 3)


def test_recall_edge_cases():
    assert recall_at_k(["a", "b"], set(), 2) == 0.0
    assert recall_at_k([], {"a"}, 10) == 0.0
    assert recall_at_k(["a", "b"], {"a", "b"}, 2) == 1.0


def test_precision_at_k_hand_computed():
    recommended = ["a", "b", "c", "d", "e"]
    relevant = {"b", "e", "z"}
    assert precision_at_k(recommended, relevant, 5) == pytest.approx(2 / 5)
    assert precision_at_k(recommended, relevant, 2) == pytest.approx(1 / 2)
    # A short list is still divided by k.
    assert precision_at_k(["b"], relevant, 10) == pytest.approx(1 / 10)


def test_ndcg_at_k_hand_computed():
    # Hits at ranks 2 and 5; three relevant items in total.
    recommended = ["a", "b", "c", "d", "e"]
    relevant = {"b", "e", "z"}
    dcg = 1 / math.log2(3) + 1 / math.log2(6)              # 0.63093 + 0.38685
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)  # 1 + 0.63093 + 0.5
    # 1.01778 / 2.13093 = 0.47762
    assert ndcg_at_k(recommended, relevant, 5) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(recommended, relevant, 5) == pytest.approx(0.47762, abs=1e-5)


def test_ndcg_ideal_is_capped_by_k():
    # Three relevant items but k = 2: the ideal list can hold only two hits.
    recommended = ["x", "a"]
    relevant = {"a", "b", "c"}
    expected = (1 / math.log2(3)) / (1 / math.log2(2) + 1 / math.log2(3))
    assert ndcg_at_k(recommended, relevant, 2) == pytest.approx(expected)
    assert ndcg_at_k(recommended, relevant, 2) == pytest.approx(0.38685, abs=1e-5)


def test_ndcg_extremes():
    assert ndcg_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == pytest.approx(1.0)
    assert ndcg_at_k(["a", "b"], {"b", "a"}, 2) == pytest.approx(1.0)  # order of hits is ideal
    assert ndcg_at_k(["x", "y"], {"a"}, 2) == 0.0
    assert ndcg_at_k(["a"], set(), 5) == 0.0
    # A single hit at rank 1 beats the same hit at rank 3.
    assert ndcg_at_k(["a", "x", "y"], {"a"}, 3) > ndcg_at_k(["x", "y", "a"], {"a"}, 3)


def test_intra_list_diversity_hand_computed():
    identical = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    assert intra_list_diversity(identical) == pytest.approx(0.0)

    orthogonal = np.eye(3)
    assert intra_list_diversity(orthogonal) == pytest.approx(1.0)

    # Pairwise cosines: (a,b)=1, (a,c)=0, (b,c)=0 -> mean 1/3 -> diversity 2/3.
    mixed = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert intra_list_diversity(mixed) == pytest.approx(2 / 3)

    # cos(45 degrees) = sqrt(2)/2 for the only pair.
    pair = np.array([[1.0, 0.0], [1.0, 1.0]])
    assert intra_list_diversity(pair) == pytest.approx(1 - math.sqrt(2) / 2)


def test_intra_list_diversity_undefined_for_single_item():
    assert math.isnan(intra_list_diversity(np.array([[1.0, 0.0]])))


def test_catalog_coverage_ignores_padding():
    recommendations = np.array([[0, 1, 2], [2, 3, -1], [-1, -1, -1]])
    assert catalog_coverage(recommendations, n_items=10) == pytest.approx(4 / 10)
    assert catalog_coverage(np.empty((0, 3), dtype=int), n_items=10) == 0.0


def test_relative_lift_and_bootstrap_interval():
    baseline = np.array([0.1, 0.2, 0.1, 0.2])
    model = np.array([0.2, 0.2, 0.2, 0.3])
    # means: 0.225 vs 0.15 -> +50%
    assert relative_lift(model, baseline) == pytest.approx(0.5)
    low, high = bootstrap_lift_interval(model, baseline, n_resamples=500, seed=0)
    assert low <= 0.5 <= high
    assert bootstrap_lift_interval(model, baseline, 500, seed=0) == (low, high)  # deterministic
