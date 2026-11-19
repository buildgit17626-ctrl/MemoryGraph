import numpy as np

from memorygraph.retrieval import GraphIndex, normalize_entities, rank_memories
from memorygraph.store import Memory


class FakeEncoder:
    def encode(self, texts):
        return np.array(
            [[1, 0] if "Python" in text or "python" in text else [0, 1] for text in texts]
        )


def test_entity_normalization_and_graph_update():
    assert normalize_entities("Use VS Code and pytest") == {"vscode", "testing"}
    memories = [Memory("1", "The user uses VS Code."), Memory("2", "The user writes Python.")]
    assert GraphIndex(memories).matching_ids("vscode setup") == {"1"}
    assert (
        GraphIndex([Memory("1", "The user writes TypeScript.")]).matching_ids("vscode setup")
        == set()
    )


def test_retrieval_methods_and_empty_memory_list():
    memories = [
        Memory("1", "The user writes Python."),
        Memory("2", "The user prefers documentation."),
    ]
    assert rank_memories("Python code", memories, method="tfidf", k=1)[0].memory.id == "1"
    assert (
        rank_memories("Python code", memories, method="semantic", encoder=FakeEncoder(), k=1)[
            0
        ].memory.id
        == "1"
    )
    assert rank_memories("python code", memories, method="graph_assisted", k=1)[0].memory.id == "1"
    assert rank_memories("anything", [], k=1) == []
