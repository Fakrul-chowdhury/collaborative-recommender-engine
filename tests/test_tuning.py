"""Tuning objective and a tiny end-to-end tuning run."""

from __future__ import annotations

import numpy as np
import pytest

from src.interactions import build_ground_truth
from src.models.content import build_item_embeddings
from src.tuning import (
    CONTENT_GRID,
    balanced_objective,
    build_validation_set,
    simulate_cold_start,
    tune_als,
    tune_content,
)


def test_balanced_objective_weights_precision_and_diversity():
    # Popularity scores exactly 1 against itself.
    assert balanced_objective(0.03, 0.9, 0.03, 0.9) == pytest.approx(1.0)
    # Twice the precision, 10% less diversity, with a 0.7 / 0.3 weighting.
    assert balanced_objective(0.06, 0.81, 0.03, 0.9, accuracy_weight=0.7) == pytest.approx(
        0.7 * 2.0 + 0.3 * 0.9)
    # Pure-accuracy and pure-diversity extremes.
    assert balanced_objective(0.06, 0.81, 0.03, 0.9, accuracy_weight=1.0) == pytest.approx(2.0)
    assert balanced_objective(0.06, 0.81, 0.03, 0.9, accuracy_weight=0.0) == pytest.approx(0.9)
    with pytest.raises(ValueError):
        balanced_objective(0.1, 0.5, 0.1, 0.5, accuracy_weight=1.5)


@pytest.fixture(scope="module")
def validation(split, train_data):
    descriptions = {item: f"PRODUCT {'ABCDEFG'[int(item[:5]) % 7] * 3}" for item in train_data.item_ids}
    embeddings = build_item_embeddings(train_data.item_ids, descriptions, n_components=4,
                                       n_categories=4)
    truth = build_ground_truth(split.val, train_data)
    return build_validation_set(train_data, truth, embeddings.vectors), embeddings


def test_validation_set_only_contains_users_als_would_serve(validation, train_data):
    validation_set, _ = validation
    assert len(validation_set.customer_ids) > 0
    assert all(train_data.history_size(c) >= 3 for c in validation_set.customer_ids)
    assert validation_set.objective(validation_set.reference) == pytest.approx(1.0)


def test_simulated_cold_start_keeps_the_most_recent_items(train_data):
    customers = train_data.user_ids[np.diff(train_data.matrix.indptr) >= 3][:10]
    visible = simulate_cold_start(train_data, customers, n_visible=1)
    for customer, items in zip(customers, visible):
        row = train_data.user_index[int(customer)]
        start, end = train_data.matrix.indptr[row], train_data.matrix.indptr[row + 1]
        latest = train_data.last_purchase[start:end].max()
        position = start + list(train_data.matrix.indices[start:end]).index(items[0])
        assert train_data.last_purchase[position] == latest


def test_tune_als_logs_every_trial_and_is_reproducible(validation):
    validation_set, _ = validation
    best, trials = tune_als(validation_set, n_trials=3, seed=1)
    again, trials_again = tune_als(validation_set, n_trials=3, seed=1)

    assert set(best) == {"factors", "regularization", "iterations", "alpha", "weighting"}
    assert len(trials) == 3
    for column in ("precision@10", "recall@10", "ndcg@10", "diversity", "coverage", "objective"):
        assert column in trials.columns
    assert best == again
    np.testing.assert_allclose(trials["objective"], trials_again["objective"])
    winner = trials.loc[trials["objective"].idxmax()]
    assert winner["factors"] == best["factors"]


def test_tune_content_covers_the_whole_grid(validation):
    validation_set, embeddings = validation
    best, grid = tune_content(validation_set, embeddings)
    n_points = len(CONTENT_GRID["category_weight"]) * len(CONTENT_GRID["popularity_weight"])
    assert len(grid) == n_points * 2  # each point is scored with 1 and with 2 visible items
    assert best["category_weight"] in CONTENT_GRID["category_weight"]
    assert best["popularity_weight"] in CONTENT_GRID["popularity_weight"]
