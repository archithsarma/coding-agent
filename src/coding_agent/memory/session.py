"""Bounded, session-isolated structured history."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol
from uuid import uuid4


@dataclass(frozen=True)
class SessionEvent:
    """Compact trajectory metadata; raw source and model output are excluded."""

    session_id: str
    trajectory: str
    summary: str
    files: tuple[str, ...] = ()
    outcome: str = "unknown"
    operation_id: str | None = None
    verification_status: str | None = None
    commands: tuple[str, ...] = ()
    event_id: str = field(default_factory=lambda: uuid4().hex)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def serialized_size(self) -> int:
        return len(json.dumps(self.__dict__, default=str, sort_keys=True).encode())


class SessionMemory(Protocol):
    """Session event boundary, isolated by the caller-provided session ID."""

    def record(self, event: SessionEvent) -> None: ...

    def list_events(self, session_id: str) -> list[SessionEvent]: ...

    def retrieve(
        self,
        session_id: str,
        *,
        trajectory: str | None = None,
        files: set[str] | None = None,
    ) -> list[SessionEvent]: ...


@dataclass
class InMemorySessionMemory:
    """Bounded oldest-first session storage with deterministic retrieval."""

    max_events: int = 50
    max_serialized_bytes: int = 32_000
    _events: list[SessionEvent] = field(default_factory=list)

    def record(self, event: SessionEvent) -> None:
        if not event.session_id or not event.summary.strip():
            raise ValueError("session events require a session ID and summary")
        if event.serialized_size() > self.max_serialized_bytes:
            raise ValueError("session event exceeds configured serialized bound")
        self._events.append(event)
        self._compact()

    def list_events(self, session_id: str) -> list[SessionEvent]:
        return [event for event in self._events if event.session_id == session_id]

    def retrieve(
        self,
        session_id: str,
        *,
        trajectory: str | None = None,
        files: set[str] | None = None,
    ) -> list[SessionEvent]:
        requested_files = files or set()
        events = self.list_events(session_id)
        return sorted(
            events,
            key=lambda event: (
                int(bool(trajectory and event.trajectory == trajectory)),
                len(requested_files.intersection(event.files)),
                event.timestamp,
            ),
            reverse=True,
        )

    def _compact(self) -> None:
        while (
            len(self._events) > self.max_events
            or self._size() > self.max_serialized_bytes
        ):
            self._events.pop(0)

    def _size(self) -> int:
        return sum(event.serialized_size() for event in self._events)
