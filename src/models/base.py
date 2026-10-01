"""Shared recommender interface and helpers."""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

#: Value used to pad recommendation lists that could not be filled.
PAD: int = -1


class Recommender(Protocol):
    """Anything that can produce top-k lists of catalogue item indices."""

    def recommend(self, customer_ids: Sequence[int], k: int) -> np.ndarray:
        """Return an ``(len(customer_ids), k)`` array of item indices, padded with ``PAD``."""
        ...


def top_k_excluding(scores: np.ndarray, exclude: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores, never returning an excluded index.

    Ties are broken by item index so results are deterministic. The result is padded
    with ``PAD`` when fewer than k candidates remain.
    """
    scores = np.array(scores, dtype=np.float64, copy=True)
    scores[exclude] = -np.inf
    k_eff = min(k, len(scores))
    top = np.argpartition(-scores, k_eff - 1)[:k_eff] if k_eff < len(scores) else np.arange(len(scores))
    top = top[np.lexsort((top, -scores[top]))]
    top = top[np.isfinite(scores[top])]
    out = np.full(k, PAD, dtype=np.int64)
    out[:len(top)] = top
    return out
