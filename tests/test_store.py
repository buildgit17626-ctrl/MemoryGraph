import pytest

from memorygraph.store import MemoryStore


def test_add_update_delete_round_trip(tmp_path):
    store = MemoryStore(tmp_path / "memories.json")
    item = store.add("  The user prefers pytest.  ")
    assert store.list()[0].text == "The user prefers pytest."
    store.update(item.id, "The user prefers unittest.")
    assert store.list()[0].text == "The user prefers unittest."
    store.delete(item.id)
    assert store.list() == []
    with pytest.raises(KeyError):
        store.delete(item.id)
    with pytest.raises(ValueError):
        store.add(" ")
