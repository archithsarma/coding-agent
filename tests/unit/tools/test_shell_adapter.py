import asyncio
from pathlib import Path

import pytest

from coding_agent.config import ShellMcpSettings
from coding_agent.domain import PolicyViolationError, ToolRequest, ToolResult
from coding_agent.policies import PolicyChain, ShellCommandPolicy, WorkspacePathPolicy
from coding_agent.tools import ToolRegistry, ToolRuntime
from coding_agent.tools.mcp import McpCallResult, McpToolDescription, ShellMcpAdapter
from coding_agent.tools.mcp.errors import McpTimeoutError, McpToolUnavailableError


def payload(*, exit_code: int = 0, cwd: str = ".") -> dict[str, object]:
    return {
        "argv": ["pytest", "tests/unit"],
        "cwd": cwd,
        "exit_code": exit_code,
        "stdout": "output",
        "stderr": "",
        "duration_ms": 3,
        "timed_out": False,
        "stdout_truncated": False,
        "stderr_truncated": False,
    }


class FakeShellConnection:
    def __init__(
        self,
        *,
        result: McpCallResult,
        tool_names: tuple[str, ...] = ("run_command",),
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.tool_names = tool_names
        self.error = error
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def __aenter__(self) -> "FakeShellConnection":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def list_tools(self) -> tuple[McpToolDescription, ...]:
        return tuple(
            McpToolDescription(name=name, description=name) for name in self.tool_names
        )

    async def call_tool(self, name: str, arguments: dict[str, object]) -> McpCallResult:
        self.calls.append((name, arguments))
        if self.error is not None:
            raise self.error
        return self.result


def make_adapter(tmp_path: Path, connection: FakeShellConnection) -> ShellMcpAdapter:
    settings = ShellMcpSettings(
        workspace_root=tmp_path,
        operation_timeout_seconds=3,
        default_timeout_seconds=1,
        max_timeout_seconds=2,
    )
    return ShellMcpAdapter(
        connection,
        path_policy=WorkspacePathPolicy(tmp_path),
        settings=settings,
    )


def invoke(
    adapter: ShellMcpAdapter,
    *,
    arguments: dict[str, object] | None = None,
    policies: PolicyChain | None = None,
) -> ToolResult:
    async def scenario() -> ToolResult:
        async with adapter:
            registry = ToolRegistry()
            adapter.register_tools(registry)
            runtime = ToolRuntime(registry, policies=policies)
            return await runtime.invoke(
                ToolRequest(
                    call_id="1",
                    capability="shell.execute",
                    arguments=arguments or {"argv": ["pytest", "tests/unit"]},
                )
            )

    return asyncio.run(scenario())


def test_shell_adapter_registers_only_shell_execute_and_normalizes_result(
    tmp_path: Path,
) -> None:
    connection = FakeShellConnection(
        result=McpCallResult(False, (), payload(exit_code=1))
    )
    adapter = make_adapter(tmp_path, connection)

    async def scenario() -> tuple[tuple[str, ...], ToolResult]:
        async with adapter:
            registry = ToolRegistry()
            adapter.register_tools(registry)
            result = await ToolRuntime(registry).invoke(
                ToolRequest(
                    call_id="1",
                    capability="shell.execute",
                    arguments={"argv": ["pytest", "tests/unit"]},
                )
            )
            return tuple(d.capability for d in registry.list_descriptors()), result

    capabilities, result = asyncio.run(scenario())
    assert capabilities == ("shell.execute",)
    assert result.success is True
    assert result.data == payload(exit_code=1)


def test_policy_rejection_prevents_mcp_invocation(tmp_path: Path) -> None:
    connection = FakeShellConnection(result=McpCallResult(False, (), payload()))
    policy_chain = PolicyChain(
        [
            WorkspacePathPolicy(tmp_path),
            ShellCommandPolicy(max_timeout_seconds=2),
        ]
    )

    with pytest.raises(PolicyViolationError):
        invoke(
            make_adapter(tmp_path, connection),
            arguments={"argv": ["bash", "-c", "echo unsafe"]},
            policies=policy_chain,
        )
    assert connection.calls == []


def test_shell_adapter_normalizes_malformed_and_mcp_timeout_results(
    tmp_path: Path,
) -> None:
    malformed = invoke(
        make_adapter(
            tmp_path,
            FakeShellConnection(result=McpCallResult(False, (), {"unexpected": True})),
        )
    )
    timed_out = invoke(
        make_adapter(
            tmp_path,
            FakeShellConnection(
                result=McpCallResult(False, (), payload()),
                error=McpTimeoutError("late"),
            ),
        ),
    )
    assert malformed.error is not None and malformed.error.code == "malformed_response"
    assert timed_out.error is not None and timed_out.error.code == "mcp_timeout"


def test_shell_adapter_normalizes_explicit_timeout_payload(tmp_path: Path) -> None:
    result = invoke(
        make_adapter(
            tmp_path,
            FakeShellConnection(
                result=McpCallResult(False, (), {**payload(), "timed_out": True})
            ),
        )
    )
    assert result.error is not None and result.error.code == "shell_timeout"


def test_shell_adapter_requires_run_command(tmp_path: Path) -> None:
    adapter = make_adapter(
        tmp_path,
        FakeShellConnection(result=McpCallResult(False, (), payload()), tool_names=()),
    )
    with pytest.raises(McpToolUnavailableError):
        asyncio.run(adapter.__aenter__())
