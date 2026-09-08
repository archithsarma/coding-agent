import asyncio
from pathlib import Path

import pytest

from coding_agent.config import ShellMcpSettings
from coding_agent.domain import ToolRequest
from coding_agent.policies import PolicyChain, ShellCommandPolicy, WorkspacePathPolicy
from coding_agent.tools import ToolRegistry, ToolRuntime
from coding_agent.tools.mcp import ShellMcpAdapter, StdioMcpClient

pytestmark = pytest.mark.integration


def test_local_shell_mcp_runs_approved_commands(tmp_path: Path) -> None:
    (tmp_path / "test_smoke.py").write_text(
        "def test_smoke():\n    assert 1 + 1 == 2\n", encoding="utf-8"
    )
    settings = ShellMcpSettings(
        workspace_root=tmp_path,
        operation_timeout_seconds=11,
        default_timeout_seconds=5,
        max_timeout_seconds=10,
    )

    async def scenario() -> tuple[object, object]:
        root = settings.resolved_workspace_root()
        adapter = ShellMcpAdapter(
            StdioMcpClient(settings, workspace_root=root),
            path_policy=WorkspacePathPolicy(root),
            settings=settings,
        )
        async with adapter:
            registry = ToolRegistry()
            adapter.register_tools(registry)
            runtime = ToolRuntime(
                registry,
                policies=PolicyChain(
                    [
                        WorkspacePathPolicy(root),
                        ShellCommandPolicy(
                            default_timeout_seconds=settings.default_timeout_seconds,
                            max_timeout_seconds=settings.max_timeout_seconds,
                            path_policy=WorkspacePathPolicy(root),
                        ),
                    ]
                ),
            )
            passed = await runtime.invoke(
                ToolRequest(
                    call_id="pass",
                    capability="shell.execute",
                    arguments={"argv": ["pytest", "test_smoke.py"], "cwd": "."},
                )
            )
            failed = await runtime.invoke(
                ToolRequest(
                    call_id="fail",
                    capability="shell.execute",
                    arguments={"argv": ["pytest", "missing_test.py"], "cwd": "."},
                )
            )
            return passed, failed

    passed, failed = asyncio.run(scenario())
    assert passed.success is True
    assert isinstance(passed.data, dict)
    assert passed.data["exit_code"] == 0
    assert failed.success is True
    assert isinstance(failed.data, dict)
    assert failed.data["exit_code"] != 0
