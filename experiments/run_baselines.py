"""Generate the synthetic dataset and run grouped classifier/retrieval baselines."""

from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from experiments.config import DATA_PATH, N_SPLITS, RESULTS_DIR, SEED, TOP_K
from memorygraph.data import make_dataset, paired_text
from memorygraph.metrics import classification_metrics, retrieval_metrics


def make_pipeline(name: str) -> Pipeline:
    classifier = (
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
        if name == "logistic_regression"
        else RandomForestClassifier(n_estimators=120, class_weight="balanced", random_state=SEED)
    )
    return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2))), ("classifier", classifier)])


def sha256(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frame = make_dataset()
    frame.to_csv(DATA_PATH, index=False)
    text = paired_text(frame)
    labels = frame["relevant"].to_numpy(dtype=int)
    groups = frame["user_id"].to_numpy()
    cv = GroupKFold(n_splits=N_SPLITS)
    summary: dict[str, object] = {
        "dataset": "synthetic fictional coding preferences; no real users",
        "rows": len(frame),
        "users": frame["user_id"].nunique(),
        "queries": frame["query_id"].nunique(),
        "seed": SEED,
        "folds": N_SPLITS,
        "top_k": TOP_K,
        "python": platform.python_version(),
        "os": platform.platform(),
        "cpu": platform.processor(),
        "logical_cpu_count": psutil.cpu_count(),
        "versions": {name: version(name) for name in ["numpy", "pandas", "scikit-learn"]},
        "dataset_sha256": sha256(DATA_PATH),
        "energy": "not measured",
        "methods": {},
    }
    predictions = frame[["query_id", "user_id", "memory_id", "relevant"]].copy()
    for name in ["logistic_regression", "random_forest"]:
        scores = np.zeros(len(frame), dtype=float)
        folds = np.full(len(frame), -1, dtype=int)
        for fold_number, (train, valid) in enumerate(cv.split(text, labels, groups)):
            assert not (set(groups[train]) & set(groups[valid]))
            model = make_pipeline(name)
            model.fit(text.iloc[train], labels[train])
            scores[valid] = model.predict_proba(text.iloc[valid])[:, 1]
            folds[valid] = fold_number
        predicted = (scores >= 0.5).astype(int)
        predictions[f"{name}_score"] = scores
        predictions[f"{name}_fold"] = folds
        summary["methods"][name] = {
            **classification_metrics(labels, predicted),
            **retrieval_metrics(frame, scores, TOP_K),
            "validation": "out-of-fold by synthetic user ID",
        }
        final_model = make_pipeline(name).fit(text, labels)
        artifact = RESULTS_DIR / f"{name}.joblib"
        joblib.dump(final_model, artifact)
        summary["methods"][name]["artifact_sha256"] = sha256(artifact)
    # Unsupervised lexical baseline: no relevance labels are used for fitting.
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(pd.concat([frame["query_text"], frame["memory_text"]]))
    count = len(frame)
    lexical_scores = 1 - paired_cosine_distances(matrix[:count], matrix[count:])
    predictions["tfidf_score"] = lexical_scores
    summary["methods"]["tfidf_cosine"] = {
        **retrieval_metrics(frame, lexical_scores, TOP_K),
        "validation": "label-free retrieval on synthetic queries",
    }
    predictions.to_csv(RESULTS_DIR / "predictions.csv", index=False)
    (RESULTS_DIR / "baseline_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary["methods"], indent=2))


if __name__ == "__main__":
    main()
