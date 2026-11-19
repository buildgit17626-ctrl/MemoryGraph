"""Editable plain-text memories for the local demo."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4


@dataclass
class Memory:
    id: str
    text: str


class MemoryStore:
    def __init__(self, path: Path):
        self.path = path

    def list(self) -> list[Memory]:
        if not self.path.exists():
            return []
        return [Memory(**item) for item in json.loads(self.path.read_text(encoding="utf-8"))]

    def _save(self, items: list[Memory]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps([asdict(item) for item in items], indent=2), encoding="utf-8")
        temp.replace(self.path)

    def add(self, text: str) -> Memory:
        value = text.strip()
        if not value:
            raise ValueError("Memory text cannot be blank")
        item = Memory(id=uuid4().hex, text=value)
        items = self.list()
        items.append(item)
        self._save(items)
        return item

    def update(self, memory_id: str, text: str) -> None:
        value = text.strip()
        if not value:
            raise ValueError("Memory text cannot be blank")
        items = self.list()
        for item in items:
            if item.id == memory_id:
                item.text = value
                self._save(items)
                return
        raise KeyError(memory_id)

    def delete(self, memory_id: str) -> None:
        items = self.list()
        remaining = [item for item in items if item.id != memory_id]
        if len(remaining) == len(items):
            raise KeyError(memory_id)
        self._save(remaining)
