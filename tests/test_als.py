"""ALS: confidence weighting, purchased-item filtering, determinism and fold-in."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.evaluate import most_recent_items
from src.interactions import build_interaction_data
from src.models.als import ALSRecommender, confidence_matrix
from src.models.base import PAD
from tests.helpers import interaction_frame, no_seen_items


def test_confidence_matrix_forms():
    counts = build_interaction_data(interaction_frame([(1, "10001", 3, "2011-01-01")])).matrix
    # c = 1 + alpha * w(count)
    assert confidence_matrix(counts, alpha=2.0, weighting="count").data.tolist() == [7.0]
    assert confidence_matrix(counts, 2.0, "log").data[0] == pytest.approx(1 + 2 * np.log(4))
    with pytest.raises(ValueError):
        confidence_matrix(counts, 1.0, "nope")


def test_confidences_are_at_least_one_for_every_weighting(train_data):
    for weighting in ("count", "log", "bm25"):
        confidence = confidence_matrix(train_data.matrix, alpha=0.5, weighting=weighting)
        assert confidence.shape == train_data.matrix.shape
        assert confidence.nnz == train_data.matrix.nnz
        assert confidence.data.min() >= 1.0


def test_als_never_recommends_purchased_items_and_is_deterministic(train_data):
    customers = train_data.user_ids[:60].tolist() + [-1]  # -1 is not a known customer
    first = ALSRecommender(factors=16, iterations=5).fit(train_data).recommend(customers, 10)
    second = ALSRecommender(factors=16, iterations=5).fit(train_data).recommend(customers, 10)

    assert first.shape == (61, 10)
    assert (first[-1] == PAD).all()
    assert (first[:-1] != PAD).all()
    assert no_seen_items(train_data, customers, first)
    np.testing.assert_array_equal(first, second)


def test_als_fold_in_uses_only_visible_items(train_data):
    sizes = np.diff(train_data.matrix.indptr)
    customers = train_data.user_ids[sizes >= 3][:20]
    rng = np.random.default_rng(0)
    visible = [most_recent_items(train_data, customer, 2, rng) for customer in customers]
    assert all(len(items) == 2 for items in visible)
    assert all(set(items) <= set(train_data.seen_items(c)) for c, items in zip(customers, visible))

    model = ALSRecommender(factors=16, iterations=5).fit(train_data)
    recommendations = model.recommend_truncated(customers, visible, 10)
    assert recommendations.shape == (20, 10)
    assert no_seen_items(train_data, customers, recommendations)
    assert model.recommend_truncated([], [], 10).shape == (0, 10)
