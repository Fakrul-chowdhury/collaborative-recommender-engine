"""Scoring recommendation lists against held-out purchases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from src import config
from src.evaluation.metrics import (
    catalog_coverage,
    intra_list_diversity,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.interactions import InteractionData
from src.models.base import PAD

METRIC_NAMES: tuple[str, ...] = ("recall", "precision", "ndcg", "diversity")


@dataclass(frozen=True)
class EvalResult:
    """Per-user metrics for one model on one set of users.

    Per-user arrays are aligned with ``customer_ids``, which lets any user segment be
    summarised afterwards with a boolean mask.
    """

    customer_ids: np.ndarray
    recommendations: np.ndarray
    recall: np.ndarray
    precision: np.ndarray
    ndcg: np.ndarray
    diversity: np.ndarray
    n_items: int
    k: int

    def summary(self, mask: np.ndarray | None = None) -> dict[str, float]:
        """Mean metrics (and catalogue coverage) over all users or the masked subset."""
        mask = np.ones(len(self.customer_ids), dtype=bool) if mask is None else mask
        n_users = int(mask.sum())
        if n_users == 0:
            return {"n_users": 0}
        k = self.k
        diversity = self.diversity[mask]
        return {
            "n_users": n_users,
            f"recall@{k}": float(self.recall[mask].mean()),
            f"precision@{k}": float(self.precision[mask].mean()),
            f"ndcg@{k}": float(self.ndcg[mask].mean()),
            "diversity": float(np.nanmean(diversity)) if np.isfinite(diversity).any() else None,
            "coverage": catalog_coverage(self.recommendations[mask], self.n_items),
        }


def evaluate_recommendations(recommendations: np.ndarray, customer_ids: Sequence[int],
                             truth: Mapping[int, set[str]], data: InteractionData,
                             item_vectors: np.ndarray, k: int = config.K) -> EvalResult:
    """Score top-k lists of item indices against each customer's held-out items.

    Args:
        recommendations: ``(n_users, k)`` item indices, ``PAD`` where a slot is empty.
        customer_ids: Customer for each row of ``recommendations``.
        truth: Customer id -> set of held-out stock codes.
        data: Training interactions, providing the index -> stock code mapping.
        item_vectors: ``(n_items, dim)`` embeddings in which diversity is measured.
        k: List length.
    """
    n = len(customer_ids)
    recall, precision, ndcg, diversity = (np.zeros(n) for _ in range(4))
    for row, customer_id in enumerate(customer_ids):
        indices = recommendations[row][recommendations[row] != PAD]
        items = data.item_ids[indices].tolist()
        relevant = truth[int(customer_id)]
        recall[row] = recall_at_k(items, relevant, k)
        precision[row] = precision_at_k(items, relevant, k)
        ndcg[row] = ndcg_at_k(items, relevant, k)
        diversity[row] = intra_list_diversity(item_vectors[indices])
    return EvalResult(np.asarray(customer_ids), recommendations, recall, precision, ndcg,
                      diversity, data.n_items, k)


def relative_lift(model: np.ndarray, baseline: np.ndarray) -> float:
    """Relative improvement of the mean of ``model`` over the mean of ``baseline``."""
    return float(model.mean() / baseline.mean() - 1.0)


def bootstrap_lift_interval(model: np.ndarray, baseline: np.ndarray, n_resamples: int = 2000,
                            seed: int = config.SEED) -> tuple[float, float]:
    """95% paired-bootstrap confidence interval for :func:`relative_lift`.

    Users are resampled with replacement; the same resample is applied to both models so
    the interval reflects their paired, per-user difference.
    """
    rng = np.random.default_rng(seed)
    n = len(model)
    draws = rng.integers(0, n, size=(n_resamples, n))
    lifts = model[draws].mean(axis=1) / baseline[draws].mean(axis=1) - 1.0
    low, high = np.percentile(lifts, [2.5, 97.5])
    return float(low), float(high)


def most_recent_items(data: InteractionData, customer_id: int, n: int,
                      rng: np.random.Generator) -> np.ndarray:
    """The customer's ``n`` most recently bought training items (random tie-break).

    Many items share a timestamp because they were bought on the same invoice, so ties
    are broken with ``rng`` rather than by stock code.
    """
    row = data.user_index[int(customer_id)]
    start, end = data.matrix.indptr[row], data.matrix.indptr[row + 1]
    items = data.matrix.indices[start:end]
    order = np.lexsort((rng.random(end - start), data.last_purchase[start:end]))
    return items[order[::-1][:n]]
