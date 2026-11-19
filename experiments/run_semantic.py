"""Compare pretrained semantic retrieval with label-free TF-IDF on synthetic data."""

from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import paired_cosine_distances

from experiments.config import (
    DATA_PATH,
    EMBEDDING_MODEL,
    EMBEDDING_REVISION,
    MODEL_CACHE,
    RESULTS_DIR,
    TOP_K,
)
from memorygraph.metrics import retrieval_metrics


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run python -m experiments.run_baselines first")
    frame = pd.read_csv(DATA_PATH)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = Path(
        snapshot_download(
            repo_id=EMBEDDING_MODEL,
            revision=EMBEDDING_REVISION,
            cache_dir=MODEL_CACHE,
        )
    )
    model = SentenceTransformer(str(snapshot))
    weight_path = snapshot / "model.safetensors"
    texts = pd.unique(pd.concat([frame["query_text"], frame["memory_text"]], ignore_index=True))
    vectors = model.encode(texts.tolist(), normalize_embeddings=True, show_progress_bar=False)
    lookup = dict(zip(texts, vectors, strict=True))
    semantic = np.array(
        [
            float(np.dot(lookup[q], lookup[m]))
            for q, m in zip(frame["query_text"], frame["memory_text"], strict=True)
        ]
    )
    lexical_matrix = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(
        pd.concat([frame["query_text"], frame["memory_text"]])
    )
    n = len(frame)
    lexical = 1 - paired_cosine_distances(lexical_matrix[:n], lexical_matrix[n:])
    output = frame[["query_id", "memory_id", "relevant"]].copy()
    output["semantic_score"] = semantic
    output["tfidf_score"] = lexical
    output.to_csv(RESULTS_DIR / "semantic_predictions.csv", index=False)
    summary = {
        "dataset": "synthetic fictional coding preferences; no real users",
        "embedding_model": EMBEDDING_MODEL,
        "embedding_revision": EMBEDDING_REVISION,
        "embedding_weights_sha256": hashlib.sha256(weight_path.read_bytes()).hexdigest(),
        "sentence_transformers_version": version("sentence-transformers"),
        "python": platform.python_version(),
        "dataset_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        "semantic": retrieval_metrics(frame, semantic, TOP_K),
        "tfidf": retrieval_metrics(frame, lexical, TOP_K),
        "energy": "not measured",
        "note": "Retrieval metrics on templated synthetic data are not real-user performance.",
    }
    (RESULTS_DIR / "semantic_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
