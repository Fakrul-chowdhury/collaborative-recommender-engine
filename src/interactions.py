"""Index-encoded interaction data shared by every model, and ground-truth construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


@dataclass
class InteractionData:
    """Training interactions as a sparse user x item matrix of purchase counts.

    Attributes:
        user_ids: Customer id for each matrix row.
        item_ids: Stock code for each matrix column; this is the recommendable catalogue.
        matrix: CSR matrix, ``matrix[u, i]`` = number of purchase occasions.
        last_purchase: For each stored entry of ``matrix`` (same order as ``matrix.data``),
            the timestamp of the most recent purchase.
    """

    user_ids: np.ndarray
    item_ids: np.ndarray
    matrix: csr_matrix
    last_purchase: np.ndarray
    user_index: dict[int, int] = field(init=False, repr=False)
    item_index: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.user_index = {int(u): i for i, u in enumerate(self.user_ids)}
        self.item_index = {str(s): i for i, s in enumerate(self.item_ids)}

    @property
    def n_users(self) -> int:
        return self.matrix.shape[0]

    @property
    def n_items(self) -> int:
        return self.matrix.shape[1]

    def history_size(self, customer_id: int) -> int:
        """Number of distinct items the customer bought in training (0 if unseen)."""
        row = self.user_index.get(int(customer_id))
        return 0 if row is None else int(self.matrix.indptr[row + 1] - self.matrix.indptr[row])

    def seen_items(self, customer_id: int) -> np.ndarray:
        """Column indices of the items the customer bought in training."""
        row = self.user_index.get(int(customer_id))
        if row is None:
            return np.empty(0, dtype=np.int32)
        return self.matrix.indices[self.matrix.indptr[row]:self.matrix.indptr[row + 1]]


def build_interaction_data(frame: pd.DataFrame) -> InteractionData:
    """Encode an aggregated interaction table (one row per customer-item pair).

    Args:
        frame: Columns ``customer_id``, ``stock_code``, ``purchase_count``, ``last_purchase``.
    """
    user_ids = np.sort(frame["customer_id"].unique())
    item_ids = np.sort(frame["stock_code"].unique().astype(str))
    rows = np.searchsorted(user_ids, frame["customer_id"].to_numpy())
    cols = np.searchsorted(item_ids, frame["stock_code"].to_numpy().astype(str))
    # Sort by (row, col) so that last_purchase lines up with the CSR data order.
    order = np.lexsort((cols, rows))
    rows, cols = rows[order], cols[order]
    counts = frame["purchase_count"].to_numpy(dtype=np.float32)[order]
    matrix = csr_matrix((counts, (rows, cols)), shape=(len(user_ids), len(item_ids)))
    matrix.sort_indices()
    return InteractionData(
        user_ids=user_ids,
        item_ids=item_ids,
        matrix=matrix,
        last_purchase=frame["last_purchase"].to_numpy()[order],
    )


def build_ground_truth(holdout: pd.DataFrame, data: InteractionData) -> dict[int, set[str]]:
    """Held-out items each customer should be recommended.

    Recommenders never return items the customer already bought in training, so repeat
    purchases are removed from the targets: a customer's ground truth is the set of items
    bought in the hold-out window that are *new to them*. Customers left with no such item
    are not evaluated. Hold-out items missing from the training catalogue are kept - no
    model can recommend them, and they count as misses for all models alike.
    """
    truth: dict[int, set[str]] = {}
    for customer_id, items in holdout.groupby("customer_id")["stock_code"]:
        seen = set(data.item_ids[data.seen_items(int(customer_id))])
        novel = set(items.astype(str)) - seen
        if novel:
            truth[int(customer_id)] = novel
    return truth


def history_sizes(data: InteractionData, customer_ids: Iterable[int]) -> np.ndarray:
    """Vector of training-history sizes for the given customers."""
    return np.array([data.history_size(c) for c in customer_ids], dtype=np.int64)
