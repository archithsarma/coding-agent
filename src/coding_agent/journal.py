"""Bounded in-memory journal for safe, session-local workspace undo."""

from __future__ import annotations

from dataclasses import dataclass

from coding_agent.content import sha256_text
from coding_agent.domain import OperationRecord, OperationStatus


@dataclass(frozen=True)
class ReversibleFile:
    path: str
    before: str
    final: str

    @property
    def before_hash(self) -> str:
        return sha256_text(self.before)

    @property
    def final_hash(self) -> str:
        return sha256_text(self.final)


@dataclass
class ReversibleOperation:
    operation: OperationRecord
    files: tuple[ReversibleFile, ...]
    available: bool = True

    @property
    def content_bytes(self) -> int:
        return sum(
            len(file.before.encode("utf-8")) + len(file.final.encode("utf-8"))
            for file in self.files
        )


class InMemoryOperationJournal:
    """Small bounded journal; it is deliberately not graph state or persistence."""

    def __init__(self, *, max_operations: int = 32, max_content_bytes: int = 4_194_304):
        if max_operations <= 0 or max_content_bytes <= 0:
            raise ValueError("journal bounds must be greater than zero")
        self._max_operations = max_operations
        self._max_content_bytes = max_content_bytes
        self._entries: list[ReversibleOperation] = []
        self._content_bytes = 0

    def record(self, entry: ReversibleOperation) -> None:
        retained = []
        for item in self._entries:
            if item.operation.operation_id == entry.operation.operation_id:
                self._content_bytes -= item.content_bytes
            else:
                retained.append(item)
        self._entries = retained
        self._content_bytes += entry.content_bytes
        self._entries.append(entry)
        self._evict()

    def latest(self) -> ReversibleOperation | None:
        for entry in reversed(self._entries):
            if entry.available:
                return entry
        return None

    def mark_reverted(self, operation_id: str) -> None:
        for entry in self._entries:
            if entry.operation.operation_id == operation_id:
                entry.available = False
                entry.operation.status = OperationStatus.REVERTED
                return

    def entries(self) -> tuple[ReversibleOperation, ...]:
        return tuple(self._entries)

    def _evict(self) -> None:
        while self._entries and (
            len(self._entries) > self._max_operations
            or self._content_bytes > self._max_content_bytes
        ):
            removed = self._entries.pop(0)
            self._content_bytes -= removed.content_bytes
            removed.available = False
