"""The SQL layer must drop every kind of non-purchase row and aggregate correctly."""

from __future__ import annotations

import pandas as pd

from src.data.download import RAW_COLUMNS, normalise_raw
from src.sql import build_database, connect


def _raw(rows: list[tuple]) -> pd.DataFrame:
    return normalise_raw(pd.DataFrame(rows, columns=RAW_COLUMNS))


def test_cleaning_rules_drop_bad_rows():
    rows = [
        # (invoice, stock_code, description, quantity, invoice_date, price, customer_id, country)
        ("1001", "85123A", "WHITE HEART HOLDER", 6, "2011-01-03 10:00:00", 2.55, 17850, "UK"),
        ("1001", 22423, "REGENCY CAKESTAND", 1, "2011-01-03 10:00:00", 12.75, 17850, "UK"),
        ("C1002", "85123A", "WHITE HEART HOLDER", -6, "2011-01-04 10:00:00", 2.55, 17850, "UK"),
        ("1003", "85123A", "WHITE HEART HOLDER", -2, "2011-01-05 10:00:00", 2.55, 17850, "UK"),
        ("1004", "85123A", "WHITE HEART HOLDER", 2, "2011-01-06 10:00:00", 0.0, 17850, "UK"),
        ("1005", "85123A", "WHITE HEART HOLDER", 2, "2011-01-07 10:00:00", 2.55, None, "UK"),
        ("1006", "POST", "POSTAGE", 1, "2011-01-08 10:00:00", 18.0, 17850, "UK"),
        ("1006", "M", "Manual", 1, "2011-01-08 10:00:00", 5.0, 17850, "UK"),
        ("1006", "BANK CHARGES", "Bank Charges", 1, "2011-01-08 10:00:00", 15.0, 17850, "UK"),
        ("1006", "gift_0001_20", "Gift voucher", 1, "2011-01-08 10:00:00", 20.0, 17850, "UK"),
        ("1006", "15056BL", "EDWARDIAN PARASOL BLACK", 3, "2011-01-08 10:00:00", 5.95, 17850, "UK"),
    ]
    conn = connect(":memory:")
    report = build_database(conn, _raw(rows))

    kept = pd.read_sql_query("SELECT invoice, stock_code FROM clean_transactions", conn)
    assert sorted(map(tuple, kept.values)) == [
        ("1001", "22423"), ("1001", "85123A"), ("1006", "15056BL"),
    ]
    assert report["raw_rows"] == len(rows)
    assert report["clean_rows"] == 3
    assert report["cancelled_invoice"] == 1
    assert report["null_customer"] == 1
    assert report["non_product_code"] == 4


def test_duplicate_lines_are_removed():
    line = ("1001", "85123A", "WHITE HEART HOLDER", 6, "2011-01-03 10:00:00", 2.55, 17850, "UK")
    conn = connect(":memory:")
    report = build_database(conn, _raw([line, line]))
    assert report["clean_rows"] == 1


def test_interactions_count_purchase_occasions_and_last_purchase():
    rows = [
        ("1001", "85123A", "HOLDER", 6, "2011-01-03 10:00:00", 2.55, 17850, "UK"),
        ("1002", "85123A", "HOLDER", 4, "2011-02-10 09:30:00", 2.55, 17850, "UK"),
        ("1002", "22423", "CAKESTAND", 1, "2011-02-10 09:30:00", 12.75, 17850, "UK"),
        ("1003", "85123A", "HOLDER", 1, "2011-03-01 12:00:00", 2.55, 13047, "UK"),
    ]
    conn = connect(":memory:")
    build_database(conn, _raw(rows))
    table = pd.read_sql_query(
        "SELECT * FROM interactions ORDER BY customer_id, stock_code", conn
    ).set_index(["customer_id", "stock_code"])

    assert len(table) == 3
    repeat = table.loc[(17850, "85123A")]
    assert repeat["purchase_count"] == 2
    assert repeat["total_quantity"] == 10
    assert repeat["last_purchase"] == "2011-02-10 09:30:00"
    assert table.loc[(17850, "22423"), "purchase_count"] == 1


def test_items_table_keeps_most_frequent_description():
    rows = [
        ("1001", "85123A", "WHITE HEART HOLDER", 1, "2011-01-03 10:00:00", 2.55, 1, "UK"),
        ("1002", "85123A", "WHITE HEART HOLDER", 1, "2011-01-04 10:00:00", 2.55, 2, "UK"),
        ("1003", "85123A", "CREAM HEART HOLDER", 1, "2011-01-05 10:00:00", 2.55, 3, "UK"),
    ]
    conn = connect(":memory:")
    build_database(conn, _raw(rows))
    items = pd.read_sql_query("SELECT * FROM items", conn)
    assert items.loc[0, "description"] == "WHITE HEART HOLDER"
    assert items.loc[0, "n_customers"] == 3


def test_synthetic_data_survives_cleaning(database):
    """The synthetic fallback contains dirty rows, and none of them reach the clean table."""
    raw = pd.read_sql_query("SELECT COUNT(*) AS n FROM raw_transactions", database)["n"][0]
    clean = pd.read_sql_query(
        """SELECT COUNT(*) AS n,
                  SUM(invoice LIKE 'C%') AS cancelled,
                  SUM(quantity <= 0) AS bad_quantity,
                  SUM(price <= 0) AS bad_price,
                  SUM(customer_id IS NULL) AS anonymous,
                  SUM(stock_code = 'POST') AS postage
           FROM clean_transactions""", database).iloc[0]
    assert 0 < clean["n"] < raw
    assert clean[["cancelled", "bad_quantity", "bad_price", "anonymous", "postage"]].sum() == 0
