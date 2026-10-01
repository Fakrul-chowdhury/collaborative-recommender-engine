"""Data acquisition: the synthetic generator and the download fallback."""

from __future__ import annotations

import pandas as pd

from src.data import download
from src.data.download import RAW_COLUMNS, load_raw_transactions
from src.data.synthetic import generate_synthetic_transactions


def test_synthetic_generator_is_seeded_and_spans_about_a_year():
    first = generate_synthetic_transactions(n_users=150, n_items=120, seed=3)
    second = generate_synthetic_transactions(n_users=150, n_items=120, seed=3)
    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == RAW_COLUMNS

    span = first["invoice_date"].max() - first["invoice_date"].min()
    assert 300 <= span.days <= 366


def test_synthetic_item_popularity_is_heavy_tailed():
    frame = generate_synthetic_transactions(n_users=600, n_items=300, seed=5)
    products = frame[frame["stock_code"] != "POST"]
    buyers = products.groupby("stock_code")["customer_id"].nunique().sort_values(ascending=False)
    top_decile_share = buyers.head(len(buyers) // 10).sum() / buyers.sum()
    assert top_decile_share > 0.3  # 10% of items draw far more than 10% of the demand


def test_download_failure_falls_back_to_synthetic(monkeypatch):
    def unreachable(*args, **kwargs):
        raise OSError("network unreachable")

    monkeypatch.setattr(download, "download_online_retail", unreachable)
    monkeypatch.setattr(download, "generate_synthetic_transactions",
                        lambda seed: generate_synthetic_transactions(100, 80, seed=seed))
    frame, source = load_raw_transactions()
    assert source == "synthetic"
    assert list(frame.columns) == RAW_COLUMNS
    assert len(frame) > 0
    # Normalised types: text ids, ISO timestamps that sort chronologically as strings.
    assert frame["invoice_date"].str.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}").all()
