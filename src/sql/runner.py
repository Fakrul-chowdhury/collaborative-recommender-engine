"""Thin helpers that execute the project's .sql files against SQLite."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src import config

#: Scripts that turn ``raw_transactions`` into the modelling tables, in execution order.
BUILD_SCRIPTS: tuple[str, ...] = (
    "01_clean_transactions.sql",
    "02_items.sql",
    "03_interactions.sql",
)


def connect(db_path: Path | str = config.DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection, creating the parent directory if needed."""
    if str(db_path) != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(db_path))


def _read_sql(name: str, sql_dir: Path = config.SQL_DIR) -> str:
    return (sql_dir / name).read_text(encoding="utf-8")


def run_script(conn: sqlite3.Connection, name: str) -> None:
    """Execute a multi-statement .sql file (DDL / table builds)."""
    conn.executescript(_read_sql(name))
    conn.commit()


def query_frame(conn: sqlite3.Connection, name: str,
                params: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Run a single-SELECT .sql file with named parameters and return the result."""
    return pd.read_sql_query(_read_sql(name), conn, params=dict(params or {}))


def load_raw_table(conn: sqlite3.Connection, raw: pd.DataFrame) -> None:
    """Replace the ``raw_transactions`` table with the given frame."""
    raw.to_sql("raw_transactions", conn, if_exists="replace", index=False, chunksize=50_000)
    conn.commit()


def build_database(conn: sqlite3.Connection, raw: pd.DataFrame | None = None) -> dict[str, int]:
    """Load raw data (if given) and run the cleaning and aggregation scripts.

    Returns:
        The cleaning report: raw row count, rows hit by each rule, and the sizes of the
        resulting tables.
    """
    if raw is not None:
        load_raw_table(conn, raw)
    for script in BUILD_SCRIPTS:
        run_script(conn, script)
    report = query_frame(conn, "06_cleaning_report.sql").iloc[0]
    return {key: int(value) for key, value in report.items()}
