"""Frozen settings for the synthetic demonstration run."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261119
N_USERS = 24
N_SPLITS = 5
TOP_K = 3
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DATA_PATH = ROOT / "data" / "generated" / "synthetic_relevance.csv"
RESULTS_DIR = ROOT / "results"
MODEL_CACHE = ROOT / "work" / "model_cache"
SCENARIOS = ("editor", "testing", "language", "performance", "documentation")
