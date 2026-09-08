"""Evaluation of memory, dry-run, trace hygiene, and conflict safety."""

from __future__ import annotations

import asyncio
from pathlib import Path

from coding_agent.domain import ExecutionBudget
from coding_agent.memory import SQLitePreferenceStore
from coding_agent.observability import InMemoryTraceSink

from .conftest import EvaluationModel, invoke, mcp_runtime


def test_dry_run_returns_candidate_without_write_or_verification(
    demo_workspace: Path,
) -> None:
    target = demo_workspace / "src/task_app/validation.py"
    original = target.read_bytes()

    async def scenario() -> dict[str, object]:
        async with mcp_runtime(demo_workspace, EvaluationModel()) as context:
            return await invoke(context, "Add validation", dry_run=True)

    result = asyncio.run(scenario())
    assert result["current_node"] == "edit_dry_run_complete"
    assert result["edit"]["file_changes"]
    assert result["counters"]["repair_attempts"] == 0
    assert result["edit"].get("operation_record") is None
    assert target.read_bytes() == original


def test_preference_survives_context_recreation_and_is_prompt_only(
    demo_workspace: Path, tmp_path: Path
) -> None:
    store_path = tmp_path / "preferences.sqlite3"
    first_store = SQLitePreferenceStore(store_path)
    first_store.remember_preference(
        "documentation", "Always add docstrings when editing functions."
    )

    async def scenario() -> EvaluationModel:
        model = EvaluationModel()
        async with mcp_runtime(demo_workspace, model) as context:
            object.__setattr__(
                context, "preference_store", SQLitePreferenceStore(store_path)
            )
            await invoke(context, "Add validation")
        return model

    model = asyncio.run(scenario())
    assert any("Always add docstrings" in item for item in model.inputs)
    assert all(
        "filesystem" not in item.lower()
        for item in model.inputs
        if "Always add" in item
    )


def test_trace_correlates_turns_and_redacts_sensitive_payloads(
    demo_workspace: Path,
) -> None:
    async def scenario() -> tuple[InMemoryTraceSink, str]:
        model = EvaluationModel()
        sink = InMemoryTraceSink()
        async with mcp_runtime(demo_workspace, model) as context:
            context.trace.sink = sink
            await invoke(context, "How are tasks created?")
            await invoke(context, "run tests")
        return sink, context.session_id

    sink, session_id = asyncio.run(scenario())
    events = sink.events()
    assert len({event.trace_id for event in events}) == 2
    assert {event.session_id for event in events} == {session_id}
    event_types = {event.event_type for event in events}
    assert {"route.selected", "trajectory.completed"} <= event_types
    serialized = str([event.as_dict() for event in events])
    assert "api_key" not in serialized
    assert "content" not in serialized
    assert "Tasks are created" not in serialized


def test_undo_conflict_preserves_external_change(demo_workspace: Path) -> None:
    target = demo_workspace / "src/task_app/validation.py"
    external = "# human change\n" + target.read_text()

    async def scenario() -> dict[str, object]:
        async with mcp_runtime(demo_workspace, EvaluationModel()) as context:
            await invoke(context, "Add validation")
            target.write_text(external, encoding="utf-8")
            return await invoke(context, "Undo that")

    result = asyncio.run(scenario())
    assert result["failure"]["code"] == "undo_conflict"
    assert target.read_text() == external


def test_repair_budget_stops_without_unbounded_loop(demo_workspace: Path) -> None:
    target = demo_workspace / "src/task_app/validation.py"
    original = target.read_text()

    async def scenario() -> dict[str, object]:
        target.write_text(
            original.replace("return normalized", "return title"), encoding="utf-8"
        )
        async with mcp_runtime(
            demo_workspace, EvaluationModel(repair=False)
        ) as context:
            return await invoke(
                context,
                "Add validation while preserving normalization",
                ExecutionBudget(
                    max_llm_calls=4,
                    max_tool_calls=64,
                    max_repair_attempts=1,
                    max_shell_execution_seconds=0,
                ),
            )

    result = asyncio.run(scenario())
    assert result["counters"]["repair_attempts"] <= 1
    assert result["current_node"] == "edit_failed"
