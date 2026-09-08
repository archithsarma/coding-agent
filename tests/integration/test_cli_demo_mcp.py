import asyncio
import shutil
from pathlib import Path

import pytest

from coding_agent.config import FilesystemMcpSettings, ShellMcpSettings
from coding_agent.domain import ExecutionBudget, Trajectory
from coding_agent.model import TextGenerationResult
from coding_agent.orchestration.context import OrchestrationContext
from coding_agent.orchestration.edit import EditPlanOutput, EditSelection
from coding_agent.orchestration.edit_config import EditConfig
from coding_agent.orchestration.explore import FileSelection
from coding_agent.orchestration.graph import build_graph
from coding_agent.policies import PolicyChain, ShellCommandPolicy, WorkspacePathPolicy
from coding_agent.tools import ToolRegistry, ToolRuntime
from coding_agent.tools.mcp import (
    FilesystemMcpAdapter,
    ShellMcpAdapter,
    StdioMcpClient,
)

pytestmark = pytest.mark.integration


class DemoModel:
    async def generate_structured(self, *, output_type, **_kwargs):
        if output_type in {FileSelection, EditSelection}:
            return output_type(paths=["routes/tasks.py"])
        return EditPlanOutput(
            files=[
                {
                    "path": "routes/tasks.py",
                    "replacements": [
                        {
                            "old_text": "    return title\n",
                            "new_text": (
                                "    if not title.strip():\n"
                                "        raise ValueError('title')\n"
                                "    return title.strip()\n"
                            ),
                            "expected_occurrences": 1,
                        }
                    ],
                }
            ]
        )

    async def generate_text(self, **_kwargs) -> TextGenerationResult:
        return TextGenerationResult(
            text="Tasks are created by routes/tasks.py.", model="fake"
        )


def test_full_multi_turn_demo_reuses_one_session_runtime(tmp_path: Path) -> None:
    if shutil.which("npx") is None:
        pytest.skip("npx is unavailable; install Node.js to run the MCP test")
    routes = tmp_path / "routes"
    routes.mkdir()
    (routes / "tasks.py").write_text(
        "def task(title):\n    return title\n", encoding="utf-8"
    )
    (tmp_path / "test_tasks.py").write_text(
        "from routes.tasks import task\n\n\n"
        "def test_title_is_returned():\n"
        "    assert task('Task') == 'Task'\n",
        encoding="utf-8",
    )
    root = tmp_path.resolve()
    filesystem_settings = FilesystemMcpSettings(workspace_root=root)
    shell_settings = ShellMcpSettings(
        workspace_root=root, operation_timeout_seconds=130
    )

    async def scenario() -> tuple[list[dict[str, object]], OrchestrationContext]:
        workspace_policy = WorkspacePathPolicy(root)
        filesystem = FilesystemMcpAdapter(
            StdioMcpClient(filesystem_settings, workspace_root=root),
            path_policy=workspace_policy,
            max_read_bytes=filesystem_settings.max_read_bytes,
            max_write_bytes=filesystem_settings.max_write_bytes,
            operation_timeout_seconds=filesystem_settings.operation_timeout_seconds,
        )
        shell = ShellMcpAdapter(
            StdioMcpClient(shell_settings, workspace_root=root),
            path_policy=workspace_policy,
            settings=shell_settings,
        )
        async with filesystem, shell:
            registry = ToolRegistry()
            filesystem.register_tools(registry)
            shell.register_tools(registry)
            runtime = ToolRuntime(
                registry,
                policies=PolicyChain(
                    [
                        workspace_policy,
                        ShellCommandPolicy(
                            max_timeout_seconds=shell_settings.max_timeout_seconds,
                            path_policy=workspace_policy,
                        ),
                    ]
                ),
            )
            context = OrchestrationContext(
                model=DemoModel(),
                tools=runtime,
                path_policy=workspace_policy,
                edit_config=EditConfig(verification_suite=(("pytest",),)),
            )
            budget = ExecutionBudget(
                max_llm_calls=4,
                max_tool_calls=40,
                max_repair_attempts=0,
                max_shell_execution_seconds=0,
            )
            graph = build_graph(budget)
            results = []
            for request in (
                "where is task creation implemented?",
                "run tests",
                "add empty-title validation",
                "undo that",
            ):
                results.append(
                    await graph.ainvoke(
                        {"task_id": "demo", "user_request": request},
                        context=context,
                    )
                )
            return results, context

    results, context = asyncio.run(scenario())
    assert results[0]["trajectory"] == Trajectory.EXPLORE.value
    assert results[1]["current_node"] == "run_complete"
    assert results[2]["current_node"] == "edit_complete"
    assert results[3]["current_node"] == "correction_complete"
    assert (routes / "tasks.py").read_text(encoding="utf-8") == (
        "def task(title):\n    return title\n"
    )
    events = context.session_memory.list_events(context.session_id)
    assert {event.trajectory for event in events} == {
        "explore",
        "run",
        "edit",
        "correction",
    }
    trace_events = context.trace.sink.events()
    assert len({event.trace_id for event in trace_events}) == 4
    assert all("return title" not in str(event.as_dict()) for event in trace_events)
