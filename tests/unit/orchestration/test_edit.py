from __future__ import annotations

import json
from pathlib import Path

import pytest

from coding_agent.domain import (
    ExecutionBudget,
    PolicyViolationError,
    ToolErrorInfo,
    ToolRequest,
    ToolResult,
)
from coding_agent.memory import (
    InMemorySessionMemory,
    PreferencePersistenceError,
    SQLitePreferenceStore,
)
from coding_agent.model import ModelError, ModelTimeoutError
from coding_agent.orchestration.context import OrchestrationContext
from coding_agent.orchestration.edit import (
    EditPlanOutput,
    EditSelection,
    RepairPlanOutput,
)
from coding_agent.orchestration.edit_config import EditConfig
from coding_agent.orchestration.graph import build_graph
from coding_agent.policies import PolicyChain, WorkspacePathPolicy
from coding_agent.tools import ToolDescriptor, ToolRegistry, ToolRuntime


class EditModel:
    def __init__(
        self, *, selected=None, planned_path="routes/tasks.py", planned_paths=None
    ) -> None:
        self.selected = selected or ["routes/tasks.py"]
        self.planned_path = planned_path
        self.planned_paths = planned_paths or [planned_path]
        self.fail_selection = False
        self.fail_plan = False
        self.replacement_old = "return []"
        self.replacement_new = (
            "if not title.strip():\n        raise ValueError('title')\n    return []"
        )
        self.structured_inputs: list[str] = []
        self.structured_instructions: list[str] = []
        self.repair_old = ""
        self.repair_new = ""

    async def generate_structured(self, *, instructions, input, output_type, **kwargs):
        self.structured_inputs.append(input)
        self.structured_instructions.append(instructions)
        if output_type is EditSelection:
            if self.fail_selection:
                raise ModelTimeoutError("selection timed out")
            return output_type(paths=self.selected)
        if output_type is RepairPlanOutput:
            return RepairPlanOutput(
                can_repair=True,
                summary="repair",
                files=[
                    {
                        "path": self.planned_path,
                        "replacements": [
                            {
                                "old_text": self.repair_old,
                                "new_text": self.repair_new,
                                "expected_occurrences": 1,
                            }
                        ],
                    }
                ],
            )
        if self.fail_plan:
            raise ModelError("planning failed")
        return EditPlanOutput(
            files=[
                {
                    "path": path,
                    "replacements": [
                        {
                            "old_text": self.replacement_old,
                            "new_text": self.replacement_new,
                            "expected_occurrences": 1,
                        }
                    ],
                }
                for path in self.planned_paths
            ]
        )

    async def generate_text(self, **kwargs):
        raise AssertionError("Edit must not call text generation")


class FakeFilesystem:
    def __init__(self, contents: dict[str, str]) -> None:
        self.contents = contents
        self.calls: list[ToolRequest] = []
        self.fail_write_path: str | None = None
        self.fail_rollback = False
        self.read_counts: dict[str, int] = {}
        self.stale_after_first_read = False

    async def list_tool(self, request: ToolRequest) -> ToolResult:
        self.calls.append(request)
        path = request.arguments["path"]
        entries = (
            [{"name": "routes", "type": "directory"}]
            if path == "."
            else [
                {"name": name.rsplit("/", 1)[-1], "type": "file"}
                for name in self.contents
                if name.startswith(f"{path}/") and "/" not in name[len(path) + 1 :]
            ]
        )
        return ToolResult(
            call_id=request.call_id,
            success=True,
            data={"path": path, "entries": entries},
        )

    async def read_tool(self, request: ToolRequest) -> ToolResult:
        self.calls.append(request)
        path = request.arguments["path"]
        assert isinstance(path, str)
        self.read_counts[path] = self.read_counts.get(path, 0) + 1
        if self.stale_after_first_read and self.read_counts[path] == 2:
            self.contents[path] = "stale\n"
        return ToolResult(
            call_id=request.call_id,
            success=True,
            data={"path": path, "content": self.contents[path]},
        )

    async def write_tool(self, request: ToolRequest) -> ToolResult:
        self.calls.append(request)
        path = request.arguments["path"]
        content = request.arguments["content"]
        assert isinstance(path, str) and isinstance(content, str)
        if path == self.fail_write_path or (
            self.fail_rollback and "rollback" in request.call_id
        ):
            return ToolResult(
                call_id=request.call_id,
                success=False,
                error=ToolErrorInfo(code="mcp_tool_error", message="write failed"),
            )
        self.contents[path] = content
        return ToolResult(
            call_id=request.call_id,
            success=True,
            data={"path": path, "bytes_written": len(content.encode("utf-8"))},
        )


class Tool:
    def __init__(self, descriptor: ToolDescriptor, handler) -> None:
        self.descriptor = descriptor
        self.handler = handler

    async def execute(self, request: ToolRequest) -> ToolResult:
        return await self.handler(request)


class FakeShell:
    def __init__(self, exit_codes: list[int] | None = None) -> None:
        self.exit_codes = exit_codes or [0, 0, 0]

    async def execute(self, request: ToolRequest) -> ToolResult:
        argv = request.arguments["argv"]
        exit_code = self.exit_codes.pop(0) if self.exit_codes else 0
        return ToolResult(
            call_id=request.call_id,
            success=True,
            data={
                "argv": argv,
                "cwd": ".",
                "exit_code": exit_code,
                "stdout": "",
                "stderr": "",
                "duration_ms": 1,
                "timed_out": False,
                "stdout_truncated": False,
                "stderr_truncated": False,
            },
        )


def runtime(
    filesystem: FakeFilesystem,
    tmp_path: Path,
    shell_exit_codes: list[int] | None = None,
) -> ToolRuntime:
    registry = ToolRegistry()
    registry.register(
        Tool(
            ToolDescriptor(
                tool_name="fake-list",
                capability="filesystem.list",
                description="list",
                mutating=False,
            ),
            filesystem.list_tool,
        )
    )
    registry.register(
        Tool(
            ToolDescriptor(
                tool_name="fake-shell",
                capability="shell.execute",
                description="shell",
                mutating=True,
            ),
            FakeShell(shell_exit_codes).execute,
        )
    )
    registry.register(
        Tool(
            ToolDescriptor(
                tool_name="fake-read",
                capability="filesystem.read",
                description="read",
                mutating=False,
            ),
            filesystem.read_tool,
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
            filesystem.write_tool,
        )
    )
    return ToolRuntime(
        registry,
        policies=PolicyChain([WorkspacePathPolicy(tmp_path)]),
    )


def budget(
    *, max_tool_calls: int = 64, max_llm_calls: int = 2, max_repair_attempts: int = 0
) -> ExecutionBudget:
    return ExecutionBudget(
        max_llm_calls=max_llm_calls,
        max_tool_calls=max_tool_calls,
        max_repair_attempts=max_repair_attempts,
        max_shell_execution_seconds=0,
    )


async def run_edit(
    tmp_path: Path,
    filesystem: FakeFilesystem,
    model: EditModel,
    *,
    max_tool_calls: int = 64,
    max_llm_calls: int = 2,
    max_repair_attempts: int = 0,
    shell_exit_codes: list[int] | None = None,
    session_memory=None,
    preference_store=None,
    session_id="test-session",
    user_request="add title validation",
    dry_run=False,
):
    return await build_graph(
        budget(
            max_tool_calls=max_tool_calls,
            max_llm_calls=max_llm_calls,
            max_repair_attempts=max_repair_attempts,
        ),
        dry_run=dry_run,
    ).ainvoke(
        {"task_id": "task-1", "user_request": user_request},
        context=OrchestrationContext(
            model=model,
            tools=runtime(filesystem, tmp_path, shell_exit_codes),
            path_policy=WorkspacePathPolicy(tmp_path),
            edit_config=EditConfig(max_write_bytes=1000),
            session_memory=session_memory or InMemorySessionMemory(),
            preference_store=preference_store
            or SQLitePreferenceStore(tmp_path / "preferences.sqlite3"),
            session_id=session_id,
        ),
    )


@pytest.mark.anyio
async def test_edit_happy_path_has_two_model_calls_and_operation_record(tmp_path: Path):
    filesystem = FakeFilesystem({"routes/tasks.py": "def task():\n    return []\n"})
    model = EditModel()
    result = await run_edit(tmp_path, filesystem, model, max_tool_calls=9)

    assert result["current_node"] == "edit_complete"
    assert result["failure"] is None
    assert filesystem.contents["routes/tasks.py"].startswith("def task")
    assert "raise ValueError" in filesystem.contents["routes/tasks.py"]
    assert result["edit"]["source_files"] == []
    assert result["edit"]["snapshots"] == []
    assert result["edit"]["file_changes"][0]["path"] == "routes/tasks.py"
    assert result["edit"]["operation_record"]["trajectory"] == "edit"
    assert result["edit"]["operation_record"]["status"] == "succeeded"
    assert len(result["edit"]["operation_record"]["verifications"]) == 3
    assert result["counters"] == {
        "llm_calls": 2,
        "tool_calls": 9,
        "repair_attempts": 0,
    }
    assert len(model.structured_inputs) == 2
    assert "def task" in model.structured_inputs[1]


@pytest.mark.anyio
async def test_edit_prompt_receives_persisted_preferences_with_precedence(
    tmp_path: Path,
):
    store = SQLitePreferenceStore(tmp_path / "preferences.sqlite3")
    store.remember_preference(
        "documentation", "Always add docstrings when editing functions."
    )
    store.close()
    store = SQLitePreferenceStore(tmp_path / "preferences.sqlite3")
    model = EditModel()
    await run_edit(
        tmp_path,
        FakeFilesystem({"routes/tasks.py": "def task():\n    return []\n"}),
        model,
        max_tool_calls=9,
        preference_store=store,
        user_request="change this function but do not add a docstring",
    )

    planner_input = json.loads(model.structured_inputs[1])
    assert planner_input["active_preferences"] == [
        "Always add docstrings when editing functions."
    ]
    assert planner_input["preference_precedence"] == (
        "current request overrides preferences"
    )


class FailingPreferenceStore:
    def retrieve_preferences(self, categories):
        raise PreferencePersistenceError("database unavailable")


@pytest.mark.anyio
async def test_preference_retrieval_failure_fails_open_for_edit(tmp_path: Path):
    model = EditModel()
    result = await run_edit(
        tmp_path,
        FakeFilesystem({"routes/tasks.py": "def task():\n    return []\n"}),
        model,
        max_tool_calls=9,
        preference_store=FailingPreferenceStore(),
    )

    assert result["current_node"] == "edit_complete"
    assert json.loads(model.structured_inputs[1])["active_preferences"] == []


@pytest.mark.anyio
async def test_edit_records_compact_session_event(tmp_path: Path):
    session_memory = InMemorySessionMemory()
    model = EditModel()
    result = await run_edit(
        tmp_path,
        FakeFilesystem({"routes/tasks.py": "def task():\n    return []\n"}),
        model,
        max_tool_calls=9,
        session_memory=session_memory,
    )

    events = session_memory.list_events("test-session")
    assert result["current_node"] == "edit_complete"
    assert len(events) == 1
    assert events[0].files == ("routes/tasks.py",)
    assert "def task" not in events[0].summary


@pytest.mark.anyio
async def test_edit_dry_run_does_not_write_verify_or_journal(tmp_path: Path):
    filesystem = FakeFilesystem({"routes/tasks.py": "def task():\n    return []\n"})
    model = EditModel()
    session_memory = InMemorySessionMemory()
    result = await run_edit(
        tmp_path,
        filesystem,
        model,
        max_tool_calls=9,
        session_memory=session_memory,
        dry_run=True,
    )

    assert result["current_node"] == "edit_dry_run_complete"
    assert filesystem.contents["routes/tasks.py"] == "def task():\n    return []\n"
    assert not [
        call for call in filesystem.calls if call.capability == "filesystem.write"
    ]
    assert not [call for call in filesystem.calls if call.capability == "shell.execute"]
    assert result["counters"] == {
        "llm_calls": 2,
        "tool_calls": 3,
        "repair_attempts": 0,
    }
    assert "Dry run only" in result["edit"]["answer"]
    assert "operation_record" not in result["edit"]


@pytest.mark.anyio
async def test_edit_restarts_verification_after_bounded_repair(tmp_path: Path):
    filesystem = FakeFilesystem(
        {"routes/tasks.py": "def task(title):\n    return []\n"}
    )
    model = EditModel()
    model.repair_old = "return []"
    model.repair_new = "return [title]"
    result = await run_edit(
        tmp_path,
        filesystem,
        model,
        max_tool_calls=15,
        max_llm_calls=3,
        max_repair_attempts=2,
        shell_exit_codes=[1, 0, 0, 0],
    )

    assert result["current_node"] == "edit_complete"
    assert result["counters"] == {
        "llm_calls": 3,
        "tool_calls": 13,
        "repair_attempts": 1,
    }
    assert filesystem.contents["routes/tasks.py"].endswith("return [title]\n")
    assert len(result["edit"]["operation_record"]["verifications"]) == 4
    assert (
        result["edit"]["file_changes"][0]["before_hash"]
        != result["edit"]["file_changes"][0]["after_hash"]
    )


@pytest.mark.parametrize("planned_path", ["users.py", "../outside.py"])
@pytest.mark.anyio
async def test_planner_cannot_widen_selected_scope(tmp_path: Path, planned_path: str):
    filesystem = FakeFilesystem({"routes/tasks.py": "return []"})
    result = await run_edit(tmp_path, filesystem, EditModel(planned_path=planned_path))

    assert result["failure"]["code"] == "invalid_edit_plan"
    assert not [
        call for call in filesystem.calls if call.capability == "filesystem.write"
    ]


@pytest.mark.anyio
async def test_missing_old_text_is_rejected_before_transaction(tmp_path: Path):
    filesystem = FakeFilesystem({"routes/tasks.py": "return []"})
    model = EditModel()
    model.replacement_old = "missing"
    result = await run_edit(tmp_path, filesystem, model)

    assert result["failure"]["code"] == "invalid_edit_plan"
    assert not [
        call for call in filesystem.calls if call.capability == "filesystem.write"
    ]


@pytest.mark.anyio
async def test_ambiguous_replacement_is_rejected_before_transaction(tmp_path: Path):
    filesystem = FakeFilesystem({"routes/tasks.py": "return []\nreturn []\n"})
    model = EditModel()
    result = await run_edit(tmp_path, filesystem, model)

    assert result["failure"]["code"] == "invalid_edit_plan"
    assert not [
        call for call in filesystem.calls if call.capability == "filesystem.write"
    ]


@pytest.mark.anyio
async def test_model_selection_failure_has_distinct_code_and_no_reads(tmp_path: Path):
    filesystem = FakeFilesystem({"routes/tasks.py": "return []"})
    model = EditModel()
    model.fail_selection = True
    result = await run_edit(tmp_path, filesystem, model)

    assert result["failure"]["code"] == "model_timeout"
    assert result["counters"]["llm_calls"] == 1
    assert not [
        call for call in filesystem.calls if call.capability == "filesystem.read"
    ]


@pytest.mark.anyio
async def test_planning_failure_clears_source_and_makes_no_write(tmp_path: Path):
    filesystem = FakeFilesystem({"routes/tasks.py": "return []"})
    model = EditModel()
    model.fail_plan = True
    result = await run_edit(tmp_path, filesystem, model)

    assert result["failure"]["code"] == "model_failure"
    assert result["counters"]["llm_calls"] == 2
    assert result["edit"]["source_files"] == []
    assert not [
        call for call in filesystem.calls if call.capability == "filesystem.write"
    ]


@pytest.mark.anyio
async def test_stale_transaction_conflict_does_not_write(tmp_path: Path):
    filesystem = FakeFilesystem({"routes/tasks.py": "return []"})
    filesystem.stale_after_first_read = True
    result = await run_edit(tmp_path, filesystem, EditModel(), max_tool_calls=8)

    assert result["failure"]["code"] == "edit_conflict"
    assert not [
        call for call in filesystem.calls if call.capability == "filesystem.write"
    ]


@pytest.mark.anyio
async def test_budget_reserve_blocks_mutation_and_minimum_safe_budget_allows_it(
    tmp_path: Path,
):
    blocked_fs = FakeFilesystem({"routes/tasks.py": "return []"})
    blocked = await run_edit(tmp_path, blocked_fs, EditModel(), max_tool_calls=7)
    assert blocked["failure"]["code"] == "tool_budget_exhausted"
    assert not [
        call for call in blocked_fs.calls if call.capability == "filesystem.write"
    ]

    allowed_fs = FakeFilesystem({"routes/tasks.py": "return []"})
    allowed = await run_edit(tmp_path, allowed_fs, EditModel(), max_tool_calls=9)
    assert allowed["current_node"] == "edit_complete"


@pytest.mark.anyio
async def test_rollback_complete_failure_is_reported_as_restored(tmp_path: Path):
    filesystem = FakeFilesystem(
        {"routes/a.py": "return []", "routes/b.py": "return []"}
    )
    filesystem.fail_write_path = "routes/b.py"
    model = EditModel(
        selected=["routes/a.py", "routes/b.py"],
        planned_paths=["routes/a.py", "routes/b.py"],
    )
    result = await run_edit(tmp_path, filesystem, model, max_tool_calls=14)

    assert result["failure"]["code"] == "mcp_tool_error"
    assert result["failure"]["details"]["rollback_complete"] is True
    assert "restored" in result["edit"]["answer"]


@pytest.mark.anyio
async def test_partial_rollback_is_high_severity_and_has_no_success_record(
    tmp_path: Path,
):
    filesystem = FakeFilesystem(
        {"routes/a.py": "return []", "routes/b.py": "return []"}
    )
    filesystem.fail_write_path = "routes/b.py"
    filesystem.fail_rollback = True
    model = EditModel(
        selected=["routes/a.py", "routes/b.py"],
        planned_paths=["routes/a.py", "routes/b.py"],
    )
    result = await run_edit(tmp_path, filesystem, model, max_tool_calls=14)

    assert result["failure"]["code"] == "partial_mutation"
    assert result["failure"]["details"]["partial_mutation_risk"] is True
    assert "partial changes" in result["edit"]["answer"]
    assert result["edit"]["operation_record"]["status"] == "failed"


@pytest.mark.anyio
async def test_policy_violation_propagates(tmp_path: Path):
    filesystem = FakeFilesystem({"routes/tasks.py": "return []"})
    registry = ToolRegistry()
    registry.register(
        Tool(
            ToolDescriptor(
                tool_name="fake-list",
                capability="filesystem.list",
                description="list",
                mutating=False,
            ),
            filesystem.list_tool,
        )
    )
    with pytest.raises(PolicyViolationError):
        await build_graph().ainvoke(
            {"task_id": "task-1", "user_request": "add title validation"},
            context=OrchestrationContext(
                model=EditModel(),
                tools=ToolRuntime(
                    registry,
                    policies=PolicyChain(
                        [WorkspacePathPolicy(tmp_path), _RejectingPolicy()]
                    ),
                ),
                path_policy=WorkspacePathPolicy(tmp_path),
            ),
        )


class _RejectingPolicy:
    async def validate(self, descriptor, request):
        raise PolicyViolationError("edit policy rejected request")
