"""Raw data acquisition: UCI download with a synthetic fallback."""

from src.data.download import RAW_COLUMNS, download_online_retail, load_raw_transactions
from src.data.synthetic import generate_synthetic_transactions

__all__ = [
    "RAW_COLUMNS",
    "download_online_retail",
    "generate_synthetic_transactions",
    "load_raw_transactions",
]
