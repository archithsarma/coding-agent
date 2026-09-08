"""Small structured tracing boundary with bounded, non-sensitive metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Protocol, cast
from uuid import uuid4

from coding_agent.model.protocol import ModelClient, ModelT, TextGenerationResult


class TraceSink(Protocol):
    def emit(self, event: TraceEvent) -> None: ...


@dataclass(frozen=True)
class TraceEvent:
    trace_id: str
    session_id: str
    event_type: str
    trajectory: str | None = None
    node: str | None = None
    operation_id: str | None = None
    duration_ms: float | None = None
    outcome: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def as_dict(self) -> dict[str, object]:
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "event_type": self.event_type,
            "trajectory": self.trajectory,
            "node": self.node,
            "operation_id": self.operation_id,
            "duration_ms": self.duration_ms,
            "outcome": self.outcome,
            "metadata": _bounded_json(self.metadata),
            "timestamp": self.timestamp,
        }


class InMemoryTraceSink:
    """Testable sink retaining only a bounded number of events."""

    def __init__(self, max_events: int = 1_000) -> None:
        self._max_events = max_events
        self._events: list[TraceEvent] = []

    def emit(self, event: TraceEvent) -> None:
        self._events.append(event)
        if len(self._events) > self._max_events:
            del self._events[: len(self._events) - self._max_events]

    def events(self) -> tuple[TraceEvent, ...]:
        return tuple(self._events)


class JsonlTraceSink:
    """Append-only JSONL trace sink; metadata is bounded before serialization."""

    def __init__(self, path: Path) -> None:
        self._path = path.expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: TraceEvent) -> None:
        with self._path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event.as_dict(), sort_keys=True) + "\n")


@dataclass
class TraceState:
    session_id: str
    sink: TraceSink = field(default_factory=InMemoryTraceSink)
    trace_id: str | None = None
    trajectory: str | None = None

    def begin_turn(self) -> str:
        self.trace_id = uuid4().hex
        self.trajectory = None
        return self.trace_id

    def emit(
        self,
        event_type: str,
        *,
        node: str | None = None,
        operation_id: str | None = None,
        duration_ms: float | None = None,
        outcome: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        if self.trace_id is None:
            self.begin_turn()
        bounded = _bounded_json(metadata or {})
        self.sink.emit(
            TraceEvent(
                trace_id=self.trace_id or "",
                session_id=self.session_id,
                event_type=event_type,
                trajectory=self.trajectory,
                node=node,
                operation_id=operation_id,
                duration_ms=duration_ms,
                outcome=outcome,
                metadata=cast(dict[str, object], bounded),
            )
        )


class TracingModelClient:
    """Provider-neutral model decorator that records timing and outcome only."""

    def __init__(self, delegate: ModelClient, trace: TraceState) -> None:
        self._delegate = delegate
        self._trace = trace

    async def generate_text(
        self,
        *,
        instructions: str,
        input: str,
        max_output_tokens: int | None = None,
    ) -> TextGenerationResult:
        started = perf_counter()
        self._trace.emit("model.started", metadata={"kind": "text"})
        try:
            result = await self._delegate.generate_text(
                instructions=instructions,
                input=input,
                max_output_tokens=max_output_tokens,
            )
        except Exception:
            self._trace.emit(
                "model.failed",
                duration_ms=(perf_counter() - started) * 1000,
                outcome="failed",
                metadata={"kind": "text"},
            )
            raise
        self._trace.emit(
            "model.completed",
            duration_ms=(perf_counter() - started) * 1000,
            outcome="succeeded",
            metadata={"kind": "text"},
        )
        return result

    async def generate_structured(
        self,
        *,
        instructions: str,
        input: str,
        output_type: type[ModelT],
        max_output_tokens: int | None = None,
    ) -> ModelT:
        started = perf_counter()
        self._trace.emit("model.started", metadata={"kind": "structured"})
        try:
            result = await self._delegate.generate_structured(
                instructions=instructions,
                input=input,
                output_type=output_type,
                max_output_tokens=max_output_tokens,
            )
        except Exception:
            self._trace.emit(
                "model.failed",
                duration_ms=(perf_counter() - started) * 1000,
                outcome="failed",
                metadata={"kind": "structured"},
            )
            raise
        self._trace.emit(
            "model.completed",
            duration_ms=(perf_counter() - started) * 1000,
            outcome="succeeded",
            metadata={"kind": "structured"},
        )
        return result


def _bounded_json(value: object, depth: int = 0) -> object:
    if depth > 3:
        return "<truncated>"
    if isinstance(value, str):
        return value[:500]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, dict):
        return {
            str(key)[:80]: (
                "<redacted>"
                if str(key).casefold()
                in {"api_key", "token", "password", "secret", "content", "prompt"}
                else _bounded_json(item, depth + 1)
            )
            for key, item in list(value.items())[:20]
        }
    if isinstance(value, (list, tuple)):
        return [_bounded_json(item, depth + 1) for item in value[:20]]
    return str(value)[:200]
