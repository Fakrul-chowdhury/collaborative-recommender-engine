"""Popularity baseline: recommend the most widely bought items the user does not own."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from src.interactions import InteractionData
from src.models.base import PAD


class PopularityRecommender:
    """Ranks items by the number of distinct customers who bought them in training.

    Args:
        window_days: If given, only purchases in the last ``window_days`` of the training
            period count ("recent popularity"). ``None`` gives the global baseline.
    """

    def __init__(self, window_days: int | None = None) -> None:
        self.window_days = window_days
        self.scores: np.ndarray | None = None
        self.ranking: np.ndarray | None = None
        self._data: InteractionData | None = None

    def fit(self, data: InteractionData) -> "PopularityRecommender":
        """Count distinct buyers per item and precompute the popularity ranking."""
        indices = data.matrix.indices
        if self.window_days is not None:
            cutoff = pd.Timestamp(data.last_purchase.max()) - pd.Timedelta(days=self.window_days)
            indices = indices[data.last_purchase >= cutoff.to_datetime64()]
        self.scores = np.bincount(indices, minlength=data.n_items).astype(np.float64)
        # Stable sort: ties resolve by item index, keeping the ranking deterministic.
        self.ranking = np.argsort(-self.scores, kind="stable")
        self._data = data
        return self

    def recommend(self, customer_ids: Sequence[int], k: int) -> np.ndarray:
        """Top-k popular items per customer, skipping items they already bought."""
        if self.ranking is None or self._data is None:
            raise RuntimeError("Call fit() before recommend()")
        out = np.full((len(customer_ids), k), PAD, dtype=np.int64)
        for row, customer_id in enumerate(customer_ids):
            seen = self._data.seen_items(customer_id)
            head = self.ranking[:k + len(seen)]
            picks = head[~np.isin(head, seen)][:k]
            out[row, :len(picks)] = picks
        return out
