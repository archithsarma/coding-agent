import asyncio
import shutil
from pathlib import Path

import pytest

from coding_agent.config import FilesystemMcpSettings
from coding_agent.domain import ExecutionBudget
from coding_agent.model import TextGenerationResult
from coding_agent.orchestration.context import OrchestrationContext
from coding_agent.orchestration.graph import build_graph
from coding_agent.policies import PolicyChain, WorkspacePathPolicy
from coding_agent.tools import ToolRegistry, ToolRuntime
from coding_agent.tools.mcp import FilesystemMcpAdapter, StdioMcpClient

pytestmark = pytest.mark.integration


class IntegrationModel:
    def __init__(self) -> None:
        self.text_inputs: list[str] = []

    async def generate_structured(self, *, output_type, **kwargs):
        return output_type(paths=["routes/tasks.py"])

    async def generate_text(self, *, input, **kwargs):
        self.text_inputs.append(input)
        return TextGenerationResult(
            text="Tasks are listed in routes/tasks.py.", model="fake"
        )


def test_explore_uses_real_filesystem_mcp(tmp_path: Path) -> None:
    if shutil.which("npx") is None:
        pytest.skip(
            "npx is unavailable; install Node.js to run the MCP integration test"
        )

    routes = tmp_path / "routes"
    routes.mkdir()
    (routes / "tasks.py").write_text(
        "def list_tasks():\n    return []\n", encoding="utf-8"
    )
    settings = FilesystemMcpSettings(
        workspace_root=tmp_path,
        operation_timeout_seconds=30,
    )
    workspace_root = settings.resolved_workspace_root()

    async def scenario() -> dict[str, object]:
        adapter = FilesystemMcpAdapter(
            StdioMcpClient(settings, workspace_root=workspace_root),
            path_policy=WorkspacePathPolicy(workspace_root),
            max_read_bytes=settings.max_read_bytes,
            operation_timeout_seconds=settings.operation_timeout_seconds,
        )
        model = IntegrationModel()
        async with adapter:
            registry = ToolRegistry()
            adapter.register_tools(registry)
            runtime = ToolRuntime(
                registry,
                policies=PolicyChain([WorkspacePathPolicy(workspace_root)]),
            )
            result = await build_graph(
                ExecutionBudget(
                    max_llm_calls=2,
                    max_tool_calls=10,
                    max_repair_attempts=0,
                    max_shell_execution_seconds=0,
                )
            ).ainvoke(
                {
                    "task_id": "integration",
                    "user_request": "where is task listing implemented?",
                },
                context=OrchestrationContext(model=model, tools=runtime),
            )
            assert model.text_inputs
            assert "def list_tasks" in model.text_inputs[0]
            return result

    result = asyncio.run(scenario())
    assert result["current_node"] == "explore_explain"
    assert result["explore"]["answer"] == "Tasks are listed in routes/tasks.py."
    assert result["explore"]["file_contents"] == []
