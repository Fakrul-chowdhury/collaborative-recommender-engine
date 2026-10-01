"""Hybrid router: ALS for users with enough history, content-based fallback otherwise."""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

from src import config
from src.interactions import InteractionData
from src.models.base import PAD, Recommender

Route = Literal["als", "content", "popularity"]

#: Human-readable user segments, keyed by the route the hybrid takes at the default threshold.
SEGMENTS: dict[Route, str] = {
    "als": "established (>=3 items)",
    "content": "low history (1-2 items)",
    "popularity": "new (0 items)",
}


def route_user(history_size: int, min_history: int = config.MIN_HISTORY) -> Route:
    """Decide which model serves a user with ``history_size`` distinct training items.

    * ``history_size >= min_history`` -> ``"als"``
    * ``0 < history_size < min_history`` -> ``"content"`` (a profile can be built)
    * ``history_size == 0`` -> ``"popularity"`` (nothing to build a profile from)
    """
    if history_size < 0:
        raise ValueError("history_size cannot be negative")
    if history_size >= min_history:
        return "als"
    if history_size > 0:
        return "content"
    return "popularity"


class HybridRecommender:
    """Dispatches each user to ALS, the content model or popularity via :func:`route_user`.

    Any slot a routed model leaves empty is back-filled from the popularity ranking, so
    every user always receives k recommendations.

    Args:
        data: Training interactions (used to look up history sizes).
        als: Collaborative model for users at or above the threshold.
        content: Fallback for users with some, but too little, history.
        popularity: Fallback for users with no history at all.
        min_history: Minimum number of distinct training items required for ALS.
    """

    def __init__(self, data: InteractionData, als: Recommender, content: Recommender,
                 popularity: Recommender, min_history: int = config.MIN_HISTORY) -> None:
        self.data = data
        self.models: dict[Route, Recommender] = {
            "als": als, "content": content, "popularity": popularity,
        }
        self.min_history = min_history

    def routes(self, customer_ids: Sequence[int]) -> np.ndarray:
        """Route label for each customer."""
        return np.array([route_user(self.data.history_size(c), self.min_history)
                         for c in customer_ids], dtype=object)

    def recommend(self, customer_ids: Sequence[int], k: int) -> np.ndarray:
        """Top-k item indices per customer from whichever model the router selects."""
        customer_ids = list(customer_ids)
        routes = self.routes(customer_ids)
        out = np.full((len(customer_ids), k), PAD, dtype=np.int64)
        for route, model in self.models.items():
            positions = np.flatnonzero(routes == route)
            if len(positions):
                out[positions] = model.recommend([customer_ids[p] for p in positions], k)

        incomplete = np.flatnonzero((out == PAD).any(axis=1))
        if len(incomplete):
            backfill = self.models["popularity"].recommend(
                [customer_ids[p] for p in incomplete], 2 * k)
            for position, candidates in zip(incomplete, backfill):
                kept = out[position][out[position] != PAD]
                extra = [c for c in candidates if c != PAD and c not in set(kept)]
                merged = np.concatenate([kept, np.array(extra, dtype=np.int64)])[:k]
                out[position, :len(merged)] = merged
        return out
