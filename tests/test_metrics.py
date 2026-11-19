import numpy as np
import pandas as pd
import pytest

from memorygraph.metrics import classification_metrics, retrieval_metrics


def test_classification_metrics_known_values():
    result = classification_metrics(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]))
    assert result == {"precision": 0.5, "recall": 0.5, "f1": 0.5}


def test_retrieval_metrics_rank_relevant_memory_first():
    frame = pd.DataFrame({"query_id": ["a", "a", "b", "b"], "relevant": [1, 0, 0, 1]})
    result = retrieval_metrics(frame, np.array([0.9, 0.1, 0.2, 0.8]), k=1)
    assert result == {"precision_at_k": 1.0, "mrr": 1.0}
    with pytest.raises(ValueError):
        retrieval_metrics(frame, np.array([0.5]), k=1)
