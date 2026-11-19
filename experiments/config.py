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
MODEL_SNAPSHOT = ROOT / "work" / "model_snapshot"
INFERENCE_MODEL = "qwen2.5-coder:0.5b"
INFERENCE_MODEL_BLOB_SHA256 = "20693aeb02c63304e263a72453b6ab89e1c700a87c6948cac523ac1e6f7cade0"
MAX_OUTPUT_TOKENS = 128
DEMO_QUERY = "Write two tests for a Python function add(a, b) that returns their sum."
DEMO_MEMORY = "The user prefers pytest for Python tests."
SCENARIOS = ("editor", "testing", "language", "performance", "documentation")
