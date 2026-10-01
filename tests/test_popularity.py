"""Popularity baseline and the shared top-k helper."""

from __future__ import annotations

import numpy as np
import pytest

from src.interactions import build_interaction_data
from src.models.base import PAD, top_k_excluding
from src.models.popularity import PopularityRecommender
from tests.helpers import interaction_frame, no_seen_items


@pytest.fixture()
def toy():
    """Item 10001 has 3 buyers, 10002 has 2, 10003 and 10004 have 1."""
    return build_interaction_data(interaction_frame([
        (1, "10001", 1, "2011-01-01"), (1, "10002", 2, "2011-06-01"),
        (2, "10001", 1, "2011-01-02"), (2, "10003", 1, "2011-06-02"),
        (3, "10001", 4, "2011-01-03"), (3, "10002", 1, "2011-01-04"),
        (3, "10004", 1, "2011-06-03"),
    ]))


def test_top_k_excluding_breaks_ties_by_index_and_pads():
    scores = np.array([0.5, 0.9, 0.9, 0.1])
    assert top_k_excluding(scores, np.array([1]), 2).tolist() == [2, 0]
    assert top_k_excluding(scores, np.array([0, 1, 2]), 3).tolist() == [3, PAD, PAD]


def test_popularity_ranks_by_distinct_buyers_and_skips_owned_items(toy):
    model = PopularityRecommender().fit(toy)
    # Distinct buyers, not purchase volume: customer 3 bought 10001 four times.
    assert model.scores.tolist() == [3, 2, 1, 1]
    # Customer 2 owns 10001 and 10003 -> gets 10002 then 10004.
    assert toy.item_ids[model.recommend([2], 2)[0]].tolist() == ["10002", "10004"]
    # An unknown customer gets the global ranking.
    assert toy.item_ids[model.recommend([99], 3)[0]].tolist() == ["10001", "10002", "10003"]
    # Customer 3 owns three of four items: one recommendation, then padding.
    assert model.recommend([3], 2)[0].tolist() == [toy.item_index["10003"], PAD]


def test_recent_popularity_only_counts_the_window(toy):
    model = PopularityRecommender(window_days=30).fit(toy)
    # Only the purchases from June 2011 fall inside the last 30 days.
    assert model.scores.tolist() == [0, 1, 1, 1]


def test_popularity_never_recommends_purchased_items(train_data):
    customers = train_data.user_ids[:80].tolist()
    recommendations = PopularityRecommender().fit(train_data).recommend(customers, 10)
    assert recommendations.shape == (80, 10)
    assert no_seen_items(train_data, customers, recommendations)
