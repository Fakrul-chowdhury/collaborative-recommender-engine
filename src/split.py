"""Chronological train / validation / test split.

Time runs ``train | validation | test``: the last ``test_weeks`` of data form the test
window and the ``val_weeks`` before it the validation window. Each set is aggregated
from ``clean_transactions`` by one windowed SQL query, so an interaction can only ever
land in the set whose time range contains it.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import pandas as pd

from src import config
from src.sql.runner import query_frame

_ISO = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class SplitBounds:
    """Half-open window boundaries: train ``[train_start, val_start)``, validation
    ``[val_start, test_start)``, test ``[test_start, test_end)``."""

    train_start: pd.Timestamp
    val_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def as_dict(self) -> dict[str, str]:
        """ISO-formatted boundaries, for logging and metrics.json."""
        return {name: getattr(self, name).strftime(_ISO)
                for name in ("train_start", "val_start", "test_start", "test_end")}


@dataclass(frozen=True)
class Split:
    """Aggregated user-item interactions for each window.

    ``train_val`` is train and validation re-aggregated together; the final models are
    fitted on it before being scored on ``test``.
    """

    bounds: SplitBounds
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_val: pd.DataFrame


def compute_split_bounds(min_date: pd.Timestamp | str, max_date: pd.Timestamp | str,
                         val_weeks: int = config.VAL_WEEKS,
                         test_weeks: int = config.TEST_WEEKS) -> SplitBounds:
    """Place the split boundaries by counting whole weeks back from the last day of data.

    Raises:
        ValueError: If the data does not span more than the validation and test windows.
    """
    min_date, max_date = pd.Timestamp(min_date), pd.Timestamp(max_date)
    test_end = max_date.normalize() + pd.Timedelta(days=1)
    test_start = test_end - pd.Timedelta(weeks=test_weeks)
    val_start = test_start - pd.Timedelta(weeks=val_weeks)
    if val_start <= min_date:
        raise ValueError("Not enough history for the requested validation and test windows")
    return SplitBounds(train_start=min_date.normalize(), val_start=val_start,
                       test_start=test_start, test_end=test_end)


def load_window(conn: sqlite3.Connection, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Aggregate user-item interactions with ``start <= invoice_date < end``."""
    frame = query_frame(conn, "04_interactions_window.sql",
                        {"start": start.strftime(_ISO), "end": end.strftime(_ISO)})
    for column in ("first_purchase", "last_purchase"):
        frame[column] = pd.to_datetime(frame[column])
    return frame


def chronological_split(conn: sqlite3.Connection, val_weeks: int = config.VAL_WEEKS,
                        test_weeks: int = config.TEST_WEEKS) -> Split:
    """Build the train / validation / test interaction tables from the database."""
    dates = query_frame(conn, "05_date_bounds.sql").iloc[0]
    bounds = compute_split_bounds(dates["min_date"], dates["max_date"], val_weeks, test_weeks)
    return Split(
        bounds=bounds,
        train=load_window(conn, bounds.train_start, bounds.val_start),
        val=load_window(conn, bounds.val_start, bounds.test_start),
        test=load_window(conn, bounds.test_start, bounds.test_end),
        train_val=load_window(conn, bounds.train_start, bounds.test_start),
    )
