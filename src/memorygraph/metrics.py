"""Classification and retrieval metrics for grouped synthetic queries."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }


def retrieval_metrics(frame: pd.DataFrame, scores: np.ndarray, k: int = 3) -> dict[str, float]:
    """Compute macro precision@k and reciprocal rank across query groups."""
    if len(frame) != len(scores):
        raise ValueError("Scores must match frame length")
    if k < 1:
        raise ValueError("k must be positive")
    working = frame[["query_id", "relevant"]].copy().reset_index(drop=True)
    working["score"] = scores
    precisions: list[float] = []
    reciprocal_ranks: list[float] = []
    for _, group in working.groupby("query_id", sort=False):
        ranked = group.sort_values("score", ascending=False, kind="stable")
        precisions.append(float(ranked.head(k)["relevant"].sum() / k))
        positions = np.flatnonzero(ranked["relevant"].to_numpy() == 1)
        reciprocal_ranks.append(float(1 / (positions[0] + 1)) if len(positions) else 0.0)
    return {"precision_at_k": float(np.mean(precisions)), "mrr": float(np.mean(reciprocal_ranks))}
