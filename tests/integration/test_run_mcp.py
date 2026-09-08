import asyncio
from pathlib import Path

import pytest

from coding_agent.config import ShellMcpSettings
from coding_agent.domain import ExecutionBudget
from coding_agent.model import TextGenerationResult
from coding_agent.orchestration.context import OrchestrationContext
from coding_agent.orchestration.graph import build_graph
from coding_agent.policies import PolicyChain, ShellCommandPolicy, WorkspacePathPolicy
from coding_agent.tools import ToolRegistry, ToolRuntime
from coding_agent.tools.mcp import ShellMcpAdapter, StdioMcpClient

pytestmark = pytest.mark.integration


class NoModel:
    async def generate_text(self, **kwargs) -> TextGenerationResult:
        raise AssertionError("Run must not call the model")

    async def generate_structured(self, **kwargs):
        raise AssertionError("Run must not call the model")


def test_run_uses_real_shell_mcp_for_passing_and_failing_tests(tmp_path: Path) -> None:
    (tmp_path / "test_smoke.py").write_text(
        "def test_pass():\n    assert 1 + 1 == 2\n",
        encoding="utf-8",
    )
    settings = ShellMcpSettings(
        workspace_root=tmp_path,
        operation_timeout_seconds=130,
    )
    workspace_root = settings.resolved_workspace_root()

    async def invoke(request: str) -> dict[str, object]:
        adapter = ShellMcpAdapter(
            StdioMcpClient(settings, workspace_root=workspace_root),
            path_policy=WorkspacePathPolicy(workspace_root),
            settings=settings,
        )
        async with adapter:
            registry = ToolRegistry()
            adapter.register_tools(registry)
            runtime = ToolRuntime(
                registry,
                policies=PolicyChain(
                    [
                        WorkspacePathPolicy(workspace_root),
                        ShellCommandPolicy(
                            max_timeout_seconds=settings.max_timeout_seconds,
                            path_policy=WorkspacePathPolicy(workspace_root),
                        ),
                    ]
                ),
            )
            return await build_graph(
                ExecutionBudget(
                    max_llm_calls=0,
                    max_tool_calls=1,
                    max_repair_attempts=0,
                    max_shell_execution_seconds=0,
                )
            ).ainvoke(
                {"task_id": "run-integration", "user_request": request},
                context=OrchestrationContext(model=NoModel(), tools=runtime),
            )

    passed = asyncio.run(invoke("run tests"))
    (tmp_path / "test_failure.py").write_text(
        "def test_fail():\n    assert 1 + 1 == 3\n",
        encoding="utf-8",
    )
    failed = asyncio.run(invoke("run pytest"))

    assert passed["current_node"] == "run_complete"
    assert passed["run"]["verification"]["passed"] is True
    assert passed["run"]["verification"]["exit_code"] == 0
    assert passed["counters"] == {
        "llm_calls": 0,
        "tool_calls": 1,
        "repair_attempts": 0,
    }
    assert failed["current_node"] == "run_complete"
    assert failed["run"]["verification"]["passed"] is False
