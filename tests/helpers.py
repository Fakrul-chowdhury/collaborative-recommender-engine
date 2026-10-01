"""Small builders shared by the model tests."""

from __future__ import annotations

import pandas as pd

from src.interactions import InteractionData
from src.models.base import PAD


def interaction_frame(rows: list[tuple]) -> pd.DataFrame:
    """Rows of ``(customer_id, stock_code, purchase_count, last_purchase)``."""
    frame = pd.DataFrame(rows, columns=["customer_id", "stock_code", "purchase_count",
                                        "last_purchase"])
    frame["last_purchase"] = pd.to_datetime(frame["last_purchase"])
    return frame


def no_seen_items(data: InteractionData, customers, recommendations) -> bool:
    """True if no recommendation list contains an item its customer bought in training."""
    return all(
        not set(recommendations[row][recommendations[row] != PAD]) & set(data.seen_items(customer))
        for row, customer in enumerate(customers)
    )
