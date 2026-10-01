"""The split must be strictly chronological: nothing from a later window leaks backwards."""

from __future__ import annotations

import pandas as pd
import pytest

from src.interactions import build_ground_truth, build_interaction_data
from src.split import compute_split_bounds, load_window


def test_bounds_are_whole_weeks_back_from_the_last_day():
    bounds = compute_split_bounds("2010-12-01 08:26:00", "2011-12-09 12:50:00",
                                  val_weeks=4, test_weeks=4)
    assert bounds.test_end == pd.Timestamp("2011-12-10")
    assert bounds.test_start == pd.Timestamp("2011-11-12")
    assert bounds.val_start == pd.Timestamp("2011-10-15")
    assert bounds.train_start == pd.Timestamp("2010-12-01")


def test_bounds_reject_too_little_history():
    with pytest.raises(ValueError):
        compute_split_bounds("2011-11-01", "2011-12-09", val_weeks=4, test_weeks=4)


def test_windows_do_not_overlap_in_time(split):
    b = split.bounds
    assert b.train_start < b.val_start < b.test_start < b.test_end
    for frame in (split.train, split.val, split.test):
        assert len(frame) > 0

    assert split.train["last_purchase"].max() < b.val_start
    assert split.val["first_purchase"].min() >= b.val_start
    assert split.val["last_purchase"].max() < b.test_start
    assert split.test["first_purchase"].min() >= b.test_start
    assert split.test["last_purchase"].max() < b.test_end
    # The refit set for the final model stops where the test window starts.
    assert split.train_val["last_purchase"].max() < b.test_start


def test_every_transaction_lands_in_exactly_one_window(database, split):
    def rows_between(start, end):
        query = "SELECT COUNT(*) FROM clean_transactions WHERE invoice_date >= ? AND invoice_date < ?"
        return database.execute(query, (str(start), str(end))).fetchone()[0]

    b = split.bounds
    total = database.execute("SELECT COUNT(*) FROM clean_transactions").fetchone()[0]
    parts = [rows_between(b.train_start, b.val_start), rows_between(b.val_start, b.test_start),
             rows_between(b.test_start, b.test_end)]
    assert sum(parts) == total
    assert all(part > 0 for part in parts)


def test_train_counts_ignore_later_purchases(database, split):
    """A pair bought in both train and test keeps only its pre-boundary count in train."""
    full = pd.read_sql_query("SELECT customer_id, stock_code, purchase_count FROM interactions",
                             database)
    merged = split.train.merge(full, on=["customer_id", "stock_code"], suffixes=("_train", "_full"))
    assert (merged["purchase_count_train"] <= merged["purchase_count_full"]).all()
    later = split.train.merge(split.test, on=["customer_id", "stock_code"], suffixes=("_train", "_test"))
    if len(later):
        both = later.merge(full, on=["customer_id", "stock_code"])
        assert (both["purchase_count_train"] < both["purchase_count"]).all()


def test_train_plus_validation_equals_single_window(database, split):
    b = split.bounds
    again = load_window(database, b.train_start, b.test_start)
    pd.testing.assert_frame_equal(again, split.train_val)
    assert split.train_val["purchase_count"].sum() == (
        split.train["purchase_count"].sum() + split.val["purchase_count"].sum())


def test_ground_truth_excludes_items_seen_in_training(split):
    data = build_interaction_data(split.train)
    truth = build_ground_truth(split.val, data)
    assert truth, "expected at least one evaluable validation user"
    train_pairs = set(zip(split.train["customer_id"], split.train["stock_code"]))
    for customer_id, items in truth.items():
        assert items, "users without new items must be dropped"
        assert not any((customer_id, item) in train_pairs for item in items)
    assert set(truth) <= set(split.val["customer_id"])
