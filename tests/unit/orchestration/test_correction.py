from datetime import UTC, datetime
from pathlib import Path

import pytest

from coding_agent.domain import (
    ExecutionBudget,
    OperationRecord,
    OperationStatus,
    ToolRequest,
    ToolResult,
    Trajectory,
)
from coding_agent.journal import (
    InMemoryOperationJournal,
    ReversibleFile,
    ReversibleOperation,
)
from coding_agent.model import TextGenerationResult
from coding_agent.orchestration.context import OrchestrationContext
from coding_agent.orchestration.graph import build_graph
from coding_agent.policies import PolicyChain, WorkspacePathPolicy
from coding_agent.tools import ToolDescriptor, ToolRegistry, ToolRuntime


class NoModel:
    def __init__(self) -> None:
        self.calls = 0

    async def generate_structured(self, **kwargs):
        self.calls += 1
        raise AssertionError("Correction must not call the model")

    async def generate_text(self, **kwargs) -> TextGenerationResult:
        self.calls += 1
        raise AssertionError("Correction must not call the model")


class Tool:
    def __init__(self, descriptor, handler) -> None:
        self.descriptor = descriptor
        self.handler = handler

    async def execute(self, request: ToolRequest) -> ToolResult:
        return await self.handler(request)


class Filesystem:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[ToolRequest] = []

    async def read(self, request: ToolRequest) -> ToolResult:
        self.calls.append(request)
        return ToolResult(
            call_id=request.call_id,
            success=True,
            data={"path": "src/task.py", "content": self.content},
        )

    async def write(self, request: ToolRequest) -> ToolResult:
        self.calls.append(request)
        self.content = request.arguments["content"]
        return ToolResult(
            call_id=request.call_id,
            success=True,
            data={"path": "src/task.py", "bytes_written": len(self.content)},
        )


def make_runtime(filesystem: Filesystem, tmp_path: Path) -> ToolRuntime:
    registry = ToolRegistry()
    registry.register(
        Tool(
            ToolDescriptor(
                tool_name="fake-read",
                capability="filesystem.read",
                description="read",
                mutating=False,
            ),
            filesystem.read,
        )
    )
    registry.register(
        Tool(
            ToolDescriptor(
                tool_name="fake-write",
                capability="filesystem.write",
                description="write",
                mutating=True,
            ),
            filesystem.write,
        )
    )
    return ToolRuntime(
        registry,
        policies=PolicyChain([WorkspacePathPolicy(tmp_path)]),
    )


def make_journal(before: str, final: str) -> InMemoryOperationJournal:
    now = datetime.now(UTC)
    operation = OperationRecord(
        operation_id="edit-1",
        trajectory=Trajectory.EDIT,
        user_request="change task",
        status=OperationStatus.SUCCEEDED,
        created_at=now,
        completed_at=now,
    )
    journal = InMemoryOperationJournal()
    journal.record(
        ReversibleOperation(
            operation,
            (ReversibleFile("src/task.py", before, final),),
        )
    )
    return journal


async def run_undo(
    tmp_path: Path,
    filesystem: Filesystem,
    journal: InMemoryOperationJournal,
):
    model = NoModel()
    result = await build_graph(
        ExecutionBudget(
            max_llm_calls=4,
            max_tool_calls=10,
            max_repair_attempts=0,
            max_shell_execution_seconds=0,
        )
    ).ainvoke(
        {"task_id": "undo-task", "user_request": "undo that"},
        context=OrchestrationContext(
            model=model,
            tools=make_runtime(filesystem, tmp_path),
            path_policy=WorkspacePathPolicy(tmp_path),
            journal=journal,
        ),
    )
    return result, model


@pytest.mark.anyio
async def test_undo_restores_original_and_marks_operation_reverted(tmp_path: Path):
    filesystem = Filesystem("B")
    journal = make_journal("A", "B")

    result, model = await run_undo(tmp_path, filesystem, journal)

    assert filesystem.content == "A"
    assert result["current_node"] == "correction_complete"
    assert result["correction"]["operation_record"]["status"] == "succeeded"
    assert result["correction"]["operation_record"]["parent_operation_id"] == "edit-1"
    change = result["correction"]["operation_record"]["file_changes"][0]
    assert change["before_hash"] != change["after_hash"]
    assert change["patch"].startswith("--- a/src/task.py")
    assert journal.entries()[0].operation.status == OperationStatus.REVERTED
    assert journal.latest() is None
    assert model.calls == 0
    assert not [call for call in filesystem.calls if call.capability == "shell.execute"]


@pytest.mark.anyio
async def test_undo_conflict_writes_nothing(tmp_path: Path):
    filesystem = Filesystem("D")
    journal = make_journal("A", "B")

    result, _ = await run_undo(tmp_path, filesystem, journal)

    assert result["failure"]["code"] == "undo_conflict"
    assert not [
        call for call in filesystem.calls if call.capability == "filesystem.write"
    ]
    assert journal.latest() is not None


@pytest.mark.anyio
async def test_undo_supports_empty_original_content(tmp_path: Path):
    filesystem = Filesystem("content")
    journal = make_journal("", "content")

    result, _ = await run_undo(tmp_path, filesystem, journal)

    assert result["current_node"] == "correction_complete"
    assert filesystem.content == ""


@pytest.mark.anyio
async def test_second_undo_does_not_redo(tmp_path: Path):
    filesystem = Filesystem("B")
    journal = make_journal("A", "B")
    first, _ = await run_undo(tmp_path, filesystem, journal)
    second, _ = await run_undo(tmp_path, filesystem, journal)

    assert first["current_node"] == "correction_complete"
    assert second["failure"]["code"] == "no_reversible_operation"
    assert filesystem.content == "A"


def test_journal_evicts_oldest_content_by_bounds():
    journal = InMemoryOperationJournal(max_operations=2, max_content_bytes=4)
    for index, value in enumerate(("A", "B", "C")):
        operation = OperationRecord(
            operation_id=f"edit-{index}",
            trajectory=Trajectory.EDIT,
            user_request="change task",
            status=OperationStatus.SUCCEEDED,
            created_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )
        journal.record(
            ReversibleOperation(
                operation, (ReversibleFile(f"src/{index}.py", "", value),)
            )
        )

    assert [entry.operation.operation_id for entry in journal.entries()] == [
        "edit-1",
        "edit-2",
    ]
    assert journal.latest().operation.operation_id == "edit-2"
