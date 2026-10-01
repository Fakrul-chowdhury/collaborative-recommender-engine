"""Ranking and beyond-accuracy metrics for top-k recommendation with binary relevance."""

from __future__ import annotations

import math
from typing import Collection, Hashable, Sequence

import numpy as np


def _hits(recommended: Sequence[Hashable], relevant: Collection[Hashable], k: int) -> list[bool]:
    relevant = relevant if isinstance(relevant, (set, frozenset)) else set(relevant)
    return [item in relevant for item in recommended[:k]]


def recall_at_k(recommended: Sequence[Hashable], relevant: Collection[Hashable], k: int) -> float:
    """Fraction of the relevant items that appear in the top-k recommendations.

    Returns 0.0 when there are no relevant items.
    """
    if not relevant:
        return 0.0
    return sum(_hits(recommended, relevant, k)) / len(relevant)


def precision_at_k(recommended: Sequence[Hashable], relevant: Collection[Hashable],
                   k: int) -> float:
    """Fraction of the k recommendation slots filled with a relevant item.

    The denominator is always ``k``, so a list shorter than k is penalised.
    """
    return sum(_hits(recommended, relevant, k)) / k


def ndcg_at_k(recommended: Sequence[Hashable], relevant: Collection[Hashable], k: int) -> float:
    """Normalised discounted cumulative gain with binary relevance.

    ``DCG = sum(1 / log2(rank + 1))`` over the ranks of the hits (rank starts at 1); the
    ideal DCG places ``min(k, len(relevant))`` hits at the top of the list.
    """
    if not relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(rank + 2)
              for rank, hit in enumerate(_hits(recommended, relevant, k)) if hit)
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(k, len(relevant))))
    return dcg / ideal


def intra_list_diversity(vectors: np.ndarray) -> float:
    """One minus the mean pairwise cosine similarity of the given item vectors.

    Args:
        vectors: Array of shape ``(n_items_in_list, dim)``; rows need not be normalised.

    Returns:
        A value in ``[0, 2]`` (0 = identical items); ``nan`` for lists with fewer than
        two items, for which diversity is undefined.
    """
    n = len(vectors)
    if n < 2:
        return float("nan")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = np.divide(vectors, norms, out=np.zeros_like(vectors, dtype=float), where=norms > 0)
    similarity = unit @ unit.T
    mean_pairwise = (similarity.sum() - np.trace(similarity)) / (n * (n - 1))
    return float(1.0 - mean_pairwise)


def catalog_coverage(recommendations: np.ndarray, n_items: int) -> float:
    """Share of the catalogue that appears in at least one recommendation list.

    Args:
        recommendations: Integer item indices of any shape; negative values are padding.
        n_items: Catalogue size.
    """
    if n_items <= 0:
        return 0.0
    flat = np.asarray(recommendations).ravel()
    return len(np.unique(flat[flat >= 0])) / n_items
