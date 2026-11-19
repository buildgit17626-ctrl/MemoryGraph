"""Lexical, semantic, and simple graph-assisted memory retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from memorygraph.store import Memory

ALIASES = {
    "py": "python",
    "python3": "python",
    "pytest": "testing",
    "unit tests": "testing",
    "typescript": "typescript",
    "ts": "typescript",
    "vs code": "vscode",
    "vscode": "vscode",
    "docs": "documentation",
    "readme": "documentation",
    "latency": "performance",
    "speed": "performance",
}
ENTITY_WORDS = {"python", "typescript", "testing", "vscode", "documentation", "performance"}


def normalize_entities(text: str) -> set[str]:
    """Map a small documented alias vocabulary to canonical entity nodes."""
    lowered = text.lower()
    entities = {
        canonical
        for alias, canonical in ALIASES.items()
        if re.search(rf"\b{re.escape(alias)}\b", lowered)
    }
    entities.update(word for word in ENTITY_WORDS if re.search(rf"\b{word}\b", lowered))
    return entities


def tfidf_scores(query: str, memories: list[Memory]) -> np.ndarray:
    if not memories:
        return np.array([], dtype=float)
    matrix = TfidfVectorizer(ngram_range=(1, 2)).fit_transform([query] + [m.text for m in memories])
    return cosine_similarity(matrix[0], matrix[1:]).ravel()


def semantic_scores(query: str, memories: list[Memory], encoder) -> np.ndarray:
    """Use a pretrained encoder; callers may inject a fake encoder in tests."""
    if not memories:
        return np.array([], dtype=float)
    vectors = np.asarray(encoder.encode([query] + [m.text for m in memories]), dtype=float)
    query_vector, memory_vectors = vectors[0:1], vectors[1:]
    return cosine_similarity(query_vector, memory_vectors).ravel()


@dataclass
class RankedMemory:
    memory: Memory
    score: float


class GraphIndex:
    def __init__(self, memories: list[Memory]):
        self.graph = nx.Graph()
        for memory in memories:
            node = f"memory:{memory.id}"
            self.graph.add_node(node, kind="memory", text=memory.text)
            for entity in normalize_entities(memory.text):
                entity_node = f"entity:{entity}"
                self.graph.add_node(entity_node, kind="entity", name=entity)
                self.graph.add_edge(node, entity_node)

    def matching_ids(self, query: str) -> set[str]:
        found: set[str] = set()
        for entity in normalize_entities(query):
            node = f"entity:{entity}"
            if node in self.graph:
                found.update(
                    neighbor.removeprefix("memory:") for neighbor in self.graph.neighbors(node)
                )
        return found


def rank_memories(
    query: str, memories: list[Memory], method: str = "tfidf", encoder=None, k: int = 3
) -> list[RankedMemory]:
    if k < 1:
        raise ValueError("k must be positive")
    if method == "tfidf":
        scores = tfidf_scores(query, memories)
    elif method == "semantic":
        if encoder is None:
            raise ValueError("semantic retrieval requires an encoder")
        scores = semantic_scores(query, memories, encoder)
    elif method == "graph_assisted":
        scores = tfidf_scores(query, memories)
        matches = GraphIndex(memories).matching_ids(query)
        scores = scores + np.array([0.25 if m.id in matches else 0 for m in memories])
    else:
        raise ValueError(f"Unknown retrieval method: {method}")
    ranked = sorted(zip(memories, scores, strict=True), key=lambda pair: pair[1], reverse=True)
    return [RankedMemory(memory=m, score=float(score)) for m, score in ranked[:k]]
