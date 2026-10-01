"""Shared fixtures: a small synthetic database built through the real SQL pipeline."""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

import src  # noqa: F401  (sets BLAS thread limits before numpy-heavy imports)
from src.data.download import normalise_raw
from src.data.synthetic import generate_synthetic_transactions
from src.sql import build_database, connect


@pytest.fixture(scope="session")
def raw_frame() -> pd.DataFrame:
    return normalise_raw(generate_synthetic_transactions(n_users=400, n_items=250, seed=7))


@pytest.fixture(scope="session")
def database(raw_frame: pd.DataFrame) -> sqlite3.Connection:
    conn = connect(":memory:")
    build_database(conn, raw_frame)
    yield conn
    conn.close()
