"""Central configuration: paths, seeds and the constants that define the experiment."""

from __future__ import annotations

from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
DB_PATH: Path = DATA_DIR / "retail.db"
SOURCE_PATH: Path = DATA_DIR / "source.json"
SQL_DIR: Path = ROOT / "sql"
RESULTS_DIR: Path = ROOT / "results"

SEED: int = 42

# Evaluation protocol
K: int = 10
VAL_WEEKS: int = 4
TEST_WEEKS: int = 4

# Users with fewer than this many distinct training items are routed away from ALS.
MIN_HISTORY: int = 3

# Tuning objective: ACCURACY_WEIGHT * relative precision@K + (1 - ACCURACY_WEIGHT) * relative
# intra-list diversity, both expressed as ratios to the popularity baseline on validation.
# Fixed before any tuning run; see README "Design decisions".
ACCURACY_WEIGHT: float = 0.7
N_TRIALS: int = 60

# Content model
EMBEDDING_DIM: int = 64
N_CATEGORIES: int = 40

DATASET_URL: str = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
