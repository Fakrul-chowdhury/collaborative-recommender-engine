"""Implicit-feedback ALS matrix factorisation (Hu, Koren & Volinsky, 2008)."""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits

from src import config
from src.interactions import InteractionData
from src.models.base import PAD, top_k_excluding

Weighting = Literal["count", "log", "bm25"]


def confidence_matrix(counts: csr_matrix, alpha: float, weighting: Weighting = "log") -> csr_matrix:
    """Turn purchase counts into ALS confidences ``c_ui = 1 + alpha * w(count_ui)``.

    Purchase logs carry no negative signal, so every observed pair is a positive whose
    confidence grows with how often it was bought, while unobserved pairs keep the
    baseline confidence of 1.

    Args:
        counts: User x item purchase counts.
        alpha: Confidence scale.
        weighting: ``"count"`` uses the raw count, ``"log"`` uses ``log1p(count)`` to damp
            bulk buyers, ``"bm25"`` applies BM25 weighting (damps heavy users and
            blockbuster items).
    """
    weights = counts.astype(np.float32).tocsr(copy=True)
    if weighting == "log":
        weights.data = np.log1p(weights.data)
    elif weighting == "bm25":
        # implicit's helper normalises per row, so it is applied to the item x user view.
        weights = bm25_weight(weights.T.tocsr(), K1=100, B=0.8).T.tocsr()
    elif weighting != "count":
        raise ValueError(f"Unknown weighting: {weighting}")
    weights.data = (1.0 + alpha * np.maximum(weights.data, 0.0)).astype(np.float32)
    weights.sort_indices()
    return weights


class ALSRecommender:
    """ALS over the confidence-weighted purchase matrix.

    Args:
        factors: Latent dimensionality.
        regularization: L2 penalty on the user and item factors.
        iterations: Number of alternating sweeps.
        alpha: Confidence scale, see :func:`confidence_matrix`.
        weighting: How purchase counts are transformed before scaling.
        seed: Seed for factor initialisation.
    """

    def __init__(self, factors: int = 64, regularization: float = 0.05, iterations: int = 20,
                 alpha: float = 10.0, weighting: Weighting = "log",
                 seed: int = config.SEED) -> None:
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.weighting = weighting
        self.seed = seed
        self.model: AlternatingLeastSquares | None = None
        self._confidence: csr_matrix | None = None
        self._data: InteractionData | None = None

    def fit(self, data: InteractionData) -> "ALSRecommender":
        """Factorise the training matrix."""
        self._data = data
        self._confidence = confidence_matrix(data.matrix, self.alpha, self.weighting)
        # ALS parallelises over users itself, so BLAS is pinned to one thread (this mirrors
        # OPENBLAS_NUM_THREADS=1 for processes that imported numpy before this package).
        with threadpool_limits(limits=1, user_api="blas"):
            # alpha stays at 1.0 because the confidences are already fully formed above.
            self.model = AlternatingLeastSquares(
                factors=self.factors,
                regularization=self.regularization,
                iterations=self.iterations,
                alpha=1.0,
                random_state=self.seed,
                use_gpu=False,
            )
            self.model.fit(self._confidence, show_progress=False)
        return self

    def _check_fitted(self) -> tuple[AlternatingLeastSquares, csr_matrix, InteractionData]:
        if self.model is None or self._confidence is None or self._data is None:
            raise RuntimeError("Call fit() before recommend()")
        return self.model, self._confidence, self._data

    def recommend(self, customer_ids: Sequence[int], k: int) -> np.ndarray:
        """Top-k items per customer, never including items bought in training.

        Customers absent from training have no user factors and get an all-``PAD`` row.
        """
        model, confidence, data = self._check_fitted()
        out = np.full((len(customer_ids), k), PAD, dtype=np.int64)
        rows = np.array([data.user_index.get(int(c), -1) for c in customer_ids])
        known = np.flatnonzero(rows >= 0)
        if len(known):
            ids, _ = model.recommend(rows[known], confidence[rows[known]], N=k,
                                     filter_already_liked_items=True)
            out[known, :ids.shape[1]] = ids
        return out

    def recommend_truncated(self, customer_ids: Sequence[int], visible: Sequence[np.ndarray],
                            k: int) -> np.ndarray:
        """Recommend as if each customer's history were only their ``visible`` items.

        User factors are re-solved ("folded in") from the visible items alone; the item
        factors are untouched. Everything the customer really bought is still excluded
        from the output. Used to simulate cold-start on customers with known behaviour.
        """
        model, confidence, data = self._check_fitted()
        if not len(customer_ids):
            return np.empty((0, k), dtype=np.int64)
        rows, cols, values = [], [], []
        for position, (customer_id, items) in enumerate(zip(customer_ids, visible)):
            full = confidence[data.user_index[int(customer_id)]]
            keep = np.isin(full.indices, items)
            rows.append(np.full(int(keep.sum()), position))
            cols.append(full.indices[keep])
            values.append(full.data[keep])
        partial = csr_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
            shape=(len(customer_ids), data.n_items), dtype=np.float32,
        )
        user_factors = model.recalculate_user(np.arange(len(customer_ids)), partial)
        scores = user_factors @ model.item_factors.T
        return np.vstack([
            top_k_excluding(scores[position], data.seen_items(customer_id), k)
            for position, customer_id in enumerate(customer_ids)
        ])
