"""Flag explicit preference conflicts without changing stored memories."""

from __future__ import annotations

import re
from dataclasses import dataclass

from memorygraph.store import Memory

PREFERENCES = {
    "testing framework": ("pytest", "unittest"),
    "editor": ("vs code", "vscode", "pycharm"),
}


@dataclass(frozen=True)
class Conflict:
    first_id: str
    second_id: str
    category: str
    first_value: str
    second_value: str


def explicit_preference(text: str) -> tuple[str, str] | None:
    """Recognize only positive first-person or user preference statements."""
    match = re.search(r"\b(?:the user|i)\s+prefers?\s+(?:to use\s+)?", text, re.I)
    if not match:
        return None
    remainder = text[match.end() :].lower()
    for category, values in PREFERENCES.items():
        for value in values:
            if re.match(rf"{re.escape(value)}\b", remainder):
                canonical = "vs code" if value == "vscode" else value
                return category, canonical
    return None


def find_conflicts(memories: list[Memory]) -> list[Conflict]:
    """Return conflicting ID pairs; never infer which preference is newer or correct."""
    parsed = [(memory, explicit_preference(memory.text)) for memory in memories]
    conflicts = []
    for index, (first, first_pref) in enumerate(parsed):
        if first_pref is None:
            continue
        for second, second_pref in parsed[index + 1 :]:
            if second_pref and first_pref[0] == second_pref[0] and first_pref[1] != second_pref[1]:
                conflicts.append(
                    Conflict(first.id, second.id, first_pref[0], first_pref[1], second_pref[1])
                )
    return conflicts


def selected_conflicts(memories: list[Memory], selected_ids: set[str]) -> list[Conflict]:
    """Block a prompt only when it would include both sides of a known conflict."""
    return [
        conflict
        for conflict in find_conflicts(memories)
        if conflict.first_id in selected_ids and conflict.second_id in selected_ids
    ]
