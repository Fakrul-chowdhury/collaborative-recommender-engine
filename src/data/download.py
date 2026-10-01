"""Download the UCI "Online Retail II" dataset, falling back to synthetic data on failure."""

from __future__ import annotations

import logging
import shutil
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from src import config
from src.data.synthetic import generate_synthetic_transactions

logger = logging.getLogger(__name__)

#: Column names of the raw table, in the order they are loaded into SQLite.
RAW_COLUMNS: list[str] = [
    "invoice",
    "stock_code",
    "description",
    "quantity",
    "invoice_date",
    "price",
    "customer_id",
    "country",
]

_UCI_RENAMES = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "price",
    "Customer ID": "customer_id",
    "Country": "country",
}


def download_online_retail(raw_dir: Path = config.RAW_DIR, url: str = config.DATASET_URL,
                           timeout: int = 120) -> Path:
    """Download and unzip the dataset, returning the path of the extracted workbook.

    The download is skipped when the workbook is already present in ``raw_dir``.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(raw_dir.glob("*.xlsx"))
    if existing:
        logger.info("Using cached workbook %s", existing[0].name)
        return existing[0]

    archive = raw_dir / "online_retail_ii.zip"
    logger.info("Downloading %s", url)
    request = urllib.request.Request(url, headers={"User-Agent": "collaborative-recommender-engine"})
    with urllib.request.urlopen(request, timeout=timeout) as response, open(archive, "wb") as out:
        shutil.copyfileobj(response, out)

    with zipfile.ZipFile(archive) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".xlsx")]
        if not members:
            raise RuntimeError(f"No .xlsx workbook found in {archive.name}")
        zf.extract(members[0], raw_dir)
    archive.unlink()
    return raw_dir / members[0]


def _read_workbook(path: Path) -> pd.DataFrame:
    """Read every sheet of the workbook into one frame with snake_case columns."""
    logger.info("Reading %s (about a million rows, this takes a few minutes)", path.name)
    sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    frame = pd.concat(sheets.values(), ignore_index=True).rename(columns=_UCI_RENAMES)
    missing = set(RAW_COLUMNS) - set(frame.columns)
    if missing:
        raise RuntimeError(f"Workbook is missing expected columns: {sorted(missing)}")
    return frame[RAW_COLUMNS]


def normalise_raw(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce raw columns to types SQLite stores unambiguously.

    Invoice numbers and stock codes are mixed int/str in the workbook, so both become
    text; timestamps become ISO-8601 strings, which sort chronologically as text.
    """
    out = frame[RAW_COLUMNS].copy()
    for column in ("invoice", "stock_code"):
        out[column] = out[column].astype(str).str.strip()
    out["invoice_date"] = pd.to_datetime(out["invoice_date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    out["customer_id"] = pd.to_numeric(out["customer_id"], errors="coerce").astype("Int64")
    return out


def load_raw_transactions(force_synthetic: bool = False,
                          seed: int = config.SEED) -> tuple[pd.DataFrame, str]:
    """Return ``(raw_transactions, source)`` where source is ``"uci_online_retail_ii"`` or
    ``"synthetic"``.

    The real dataset is always tried first; any failure (network, parsing) falls back to
    the synthetic generator so the pipeline stays runnable offline. The source is recorded
    alongside the metrics so results are never mistaken for the other kind of data.
    """
    if not force_synthetic:
        try:
            workbook = download_online_retail()
            return normalise_raw(_read_workbook(workbook)), "uci_online_retail_ii"
        except Exception as exc:  # noqa: BLE001 - any failure must trigger the fallback
            logger.warning("Dataset download failed (%s); using synthetic transactions", exc)
    return normalise_raw(generate_synthetic_transactions(seed=seed)), "synthetic"
