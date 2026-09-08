import json
from pathlib import Path

import pytest

from coding_agent.observability import (
    InMemoryTraceSink,
    JsonlTraceSink,
    TraceState,
)


def test_trace_state_correlates_turns_and_redacts_sensitive_metadata() -> None:
    sink = InMemoryTraceSink()
    trace = TraceState(session_id="session-1", sink=sink)

    first = trace.begin_turn()
    trace.emit(
        "request.received",
        metadata={"api_key": "secret", "content": "source text", "count": 1},
    )
    second = trace.begin_turn()
    trace.emit("route.selected", metadata={"source": "deterministic"})

    events = sink.events()
    assert first != second
    assert events[0].trace_id == first
    assert events[1].trace_id == second
    assert events[0].metadata["api_key"] == "<redacted>"
    assert events[0].metadata["content"] == "<redacted>"


def test_jsonl_trace_sink_writes_one_bounded_json_object_per_line(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trace.jsonl"
    trace = TraceState(session_id="session-1", sink=JsonlTraceSink(path))
    trace.begin_turn()
    trace.emit("tool.completed", metadata={"capability": "filesystem.read"})

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["event_type"] == "tool.completed"
    assert payload["session_id"] == "session-1"


@pytest.mark.anyio
async def test_trace_state_can_record_model_failure() -> None:
    from coding_agent.model import ModelNotConfiguredError
    from coding_agent.observability import TracingModelClient

    class MissingModel:
        async def generate_text(self, **_kwargs):
            raise ModelNotConfiguredError("model_not_configured")

        async def generate_structured(self, **_kwargs):
            raise ModelNotConfiguredError("model_not_configured")

    sink = InMemoryTraceSink()
    trace = TraceState(session_id="session-1", sink=sink)
    trace.begin_turn()

    with pytest.raises(ModelNotConfiguredError):
        await TracingModelClient(MissingModel(), trace).generate_text(
            instructions="x", input="y"
        )

    assert [event.event_type for event in sink.events()] == [
        "model.started",
        "model.failed",
    ]
