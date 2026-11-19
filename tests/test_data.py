import pytest
from sklearn.model_selection import GroupKFold

from memorygraph.data import make_dataset


def test_synthetic_dataset_has_one_relevant_memory_per_query_and_grouped_users():
    frame = make_dataset(n_users=10, seed=7)
    assert len(frame) == 10 * 5 * 5
    assert frame.groupby("query_id")["relevant"].sum().eq(1).all()
    groups = frame["user_id"].to_numpy()
    for train, valid in GroupKFold(n_splits=5).split(frame, frame["relevant"], groups):
        assert set(groups[train]).isdisjoint(set(groups[valid]))


def test_dataset_rejects_too_few_users():
    with pytest.raises(ValueError):
        make_dataset(n_users=4)
