import asyncio
import shutil
from pathlib import Path

import pytest

from coding_agent.config import FilesystemMcpSettings
from coding_agent.domain import ExecutionBudget
from coding_agent.model import TextGenerationResult
from coding_agent.orchestration.context import OrchestrationContext
from coding_agent.orchestration.edit import EditPlanOutput, EditSelection
from coding_agent.orchestration.graph import build_graph
from coding_agent.policies import PolicyChain, WorkspacePathPolicy
from coding_agent.tools import ToolRegistry, ToolRuntime
from coding_agent.tools.mcp import FilesystemMcpAdapter, StdioMcpClient

pytestmark = pytest.mark.integration


class EditIntegrationModel:
    def __init__(self) -> None:
        self.structured_calls = 0

    async def generate_structured(self, *, output_type, **kwargs):
        self.structured_calls += 1
        if output_type is EditSelection:
            return output_type(paths=["routes/tasks.py"])
        return EditPlanOutput(
            files=[
                {
                    "path": "routes/tasks.py",
                    "replacements": [
                        {
                            "old_text": "return []",
                            "new_text": (
                                "if not title.strip():\n"
                                "        raise ValueError('title')\n"
                                "    return []"
                            ),
                            "expected_occurrences": 1,
                        }
                    ],
                }
            ]
        )

    async def generate_text(self, **kwargs) -> TextGenerationResult:
        raise AssertionError("Edit must not call text generation")


def test_real_edit_trajectory_uses_mcp_transaction(tmp_path: Path) -> None:
    if shutil.which("npx") is None:
        pytest.skip("npx is unavailable; install Node.js to run the MCP test")

    routes = tmp_path / "routes"
    routes.mkdir()
    source = "def task(title):\n    return []\n"
    (routes / "tasks.py").write_text(source, encoding="utf-8")
    settings = FilesystemMcpSettings(
        workspace_root=tmp_path, operation_timeout_seconds=30
    )
    workspace_root = settings.resolved_workspace_root()
    model = EditIntegrationModel()

    async def scenario() -> dict[str, object]:
        adapter = FilesystemMcpAdapter(
            StdioMcpClient(settings, workspace_root=workspace_root),
            path_policy=WorkspacePathPolicy(workspace_root),
            max_read_bytes=settings.max_read_bytes,
            max_write_bytes=settings.max_write_bytes,
            operation_timeout_seconds=settings.operation_timeout_seconds,
        )
        async with adapter:
            registry = ToolRegistry()
            adapter.register_tools(registry)
            runtime = ToolRuntime(
                registry,
                policies=PolicyChain([WorkspacePathPolicy(workspace_root)]),
            )
            return await build_graph(
                ExecutionBudget(
                    max_llm_calls=2,
                    max_tool_calls=8,
                    max_repair_attempts=0,
                    max_shell_execution_seconds=0,
                )
            ).ainvoke(
                {"task_id": "edit-integration", "user_request": "add title validation"},
                context=OrchestrationContext(
                    model=model,
                    tools=runtime,
                    path_policy=WorkspacePathPolicy(workspace_root),
                ),
            )

    result = asyncio.run(scenario())
    updated = (routes / "tasks.py").read_text(encoding="utf-8")
    assert result["current_node"] == "edit_complete"
    assert result["failure"] is None
    assert updated == (
        "def task(title):\n"
        "    if not title.strip():\n"
        "        raise ValueError('title')\n"
        "    return []\n"
    )
    assert result["edit"]["file_changes"][0]["path"] == "routes/tasks.py"
    assert result["edit"]["file_changes"][0]["before_hash"]
    assert result["edit"]["file_changes"][0]["after_hash"]
    assert result["edit"]["file_changes"][0]["patch"].startswith(
        "--- a/routes/tasks.py"
    )
    assert result["edit"]["operation_record"]["status"] == "succeeded"
    assert result["edit"]["operation_record"]["verifications"] == []
    assert result["counters"] == {
        "llm_calls": 2,
        "tool_calls": 6,
        "repair_attempts": 0,
    }
    assert model.structured_calls == 2
