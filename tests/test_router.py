"""The hybrid router sends users with fewer than 3 training items away from ALS."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.interactions import build_interaction_data
from src.models.hybrid import HybridRecommender, route_user


class _Stub:
    """Recommender that fills every slot with one fixed item index and records its callers."""

    def __init__(self, item: int) -> None:
        self.item = item
        self.served: list[int] = []

    def recommend(self, customer_ids, k):
        self.served.extend(int(c) for c in customer_ids)
        return np.full((len(customer_ids), k), self.item, dtype=np.int64)


def _interactions(history: dict[int, int]) -> pd.DataFrame:
    """One customer per key, with that many distinct items."""
    rows = [(customer, f"{10000 + i}", 1, pd.Timestamp("2011-01-01"))
            for customer, n_items in history.items() for i in range(n_items)]
    return pd.DataFrame(rows, columns=["customer_id", "stock_code", "purchase_count",
                                       "last_purchase"])


def test_default_threshold_is_three():
    assert config.MIN_HISTORY == 3


@pytest.mark.parametrize(("history_size", "expected"), [
    (0, "popularity"),
    (1, "content"),
    (2, "content"),
    (3, "als"),
    (4, "als"),
    (250, "als"),
])
def test_route_user_threshold(history_size, expected):
    assert route_user(history_size) == expected


def test_route_user_custom_threshold_and_validation():
    assert route_user(4, min_history=5) == "content"
    assert route_user(5, min_history=5) == "als"
    assert route_user(1, min_history=1) == "als"
    with pytest.raises(ValueError):
        route_user(-1)


def test_hybrid_dispatches_each_user_to_the_right_model():
    data = build_interaction_data(_interactions({101: 1, 102: 2, 103: 3, 104: 6}))
    als, content, popularity = _Stub(0), _Stub(1), _Stub(2)
    hybrid = HybridRecommender(data, als, content, popularity)

    customers = [101, 102, 103, 104, 999]  # 999 never appears in training
    assert hybrid.routes(customers).tolist() == ["content", "content", "als", "als", "popularity"]

    recommendations = hybrid.recommend(customers, k=4)
    assert recommendations.shape == (5, 4)
    assert (recommendations[[0, 1]] == 1).all()   # content stub
    assert (recommendations[[2, 3]] == 0).all()   # ALS stub
    assert (recommendations[4] == 2).all()        # popularity stub
    assert als.served == [103, 104]
    assert content.served == [101, 102]
    assert popularity.served == [999]


def test_history_counts_distinct_items_not_purchase_volume():
    """Buying one item many times is still a single interaction for routing purposes."""
    frame = _interactions({201: 1})
    frame["purchase_count"] = 25
    data = build_interaction_data(frame)
    assert data.history_size(201) == 1
    assert route_user(data.history_size(201)) == "content"
    assert route_user(data.history_size(555)) == "popularity"


def test_empty_slots_are_backfilled_from_popularity():
    data = build_interaction_data(_interactions({301: 5}))

    class _Empty:
        def recommend(self, customer_ids, k):
            return np.full((len(customer_ids), k), -1, dtype=np.int64)

    class _Ranked:
        def recommend(self, customer_ids, k):
            return np.tile(np.arange(k), (len(customer_ids), 1))

    hybrid = HybridRecommender(data, _Empty(), _Empty(), _Ranked())
    assert hybrid.recommend([301], k=3).tolist() == [[0, 1, 2]]
