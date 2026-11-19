"""Frozen settings for the synthetic demonstration run."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261119
N_USERS = 24
N_SPLITS = 5
TOP_K = 3
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DATA_PATH = ROOT / "data" / "generated" / "synthetic_relevance.csv"
RESULTS_DIR = ROOT / "results"
SCENARIOS = ("editor", "testing", "language", "performance", "documentation")
