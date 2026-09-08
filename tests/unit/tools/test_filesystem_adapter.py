import asyncio
from collections.abc import Awaitable
from pathlib import Path
from typing import TypeVar

import pytest

from coding_agent.domain import ToolRequest, ToolResult
from coding_agent.policies import WorkspacePathPolicy
from coding_agent.tools import ToolRegistry, ToolRuntime
from coding_agent.tools.mcp import (
    FilesystemMcpAdapter,
    McpCallResult,
    McpToolDescription,
)

ResultT = TypeVar("ResultT")


class FakeMcpConnection:
    def __init__(self, *, call_result: McpCallResult | None = None) -> None:
        self.call_result = call_result
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def __aenter__(self) -> "FakeMcpConnection":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def list_tools(self) -> tuple[McpToolDescription, ...]:
        return (
            McpToolDescription(name="read_text_file", description="Read text."),
            McpToolDescription(name="list_directory", description="List entries."),
        )

    async def call_tool(self, name: str, arguments: dict[str, object]) -> McpCallResult:
        self.calls.append((name, arguments))
        assert self.call_result is not None
        return self.call_result


class ExplodingConnection(FakeMcpConnection):
    async def call_tool(self, name: str, arguments: dict[str, object]) -> McpCallResult:
        raise RuntimeError("adapter programming defect")


def run(coroutine: Awaitable[ResultT]) -> ResultT:
    return asyncio.run(coroutine)


def make_adapter(
    tmp_path: Path, connection: FakeMcpConnection, *, max_read_bytes: int = 100
) -> FilesystemMcpAdapter:
    return FilesystemMcpAdapter(
        connection,
        path_policy=WorkspacePathPolicy(tmp_path),
        max_read_bytes=max_read_bytes,
        operation_timeout_seconds=0.1,
    )


def test_adapter_registers_only_read_and_list_capabilities(tmp_path: Path) -> None:
    connection = FakeMcpConnection(
        call_result=McpCallResult(
            is_error=False,
            content=(),
            structured_content={"content": "[FILE] main.py"},
        )
    )
    adapter = make_adapter(tmp_path, connection)

    async def scenario() -> tuple[str, ...]:
        async with adapter:
            registry = ToolRegistry()
            adapter.register_tools(registry)
            return tuple(d.capability for d in registry.list_descriptors())

    assert run(scenario()) == ("filesystem.list", "filesystem.read")


def test_adapter_normalizes_list_and_read_results(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("print('ok')", encoding="utf-8")
    connection = FakeMcpConnection(
        call_result=McpCallResult(
            is_error=False,
            content=({"type": "text", "text": "[FILE] main.py\n[DIR] src"},),
            structured_content={"content": "[FILE] main.py\n[DIR] src"},
        )
    )
    adapter = make_adapter(tmp_path, connection)

    async def scenario() -> tuple[ToolResult, ToolResult]:
        async with adapter:
            registry = ToolRegistry()
            adapter.register_tools(registry)
            runtime = ToolRuntime(registry)
            listing = await runtime.invoke(
                ToolRequest(
                    call_id="1", capability="filesystem.list", arguments={"path": "."}
                )
            )
            reading = await runtime.invoke(
                ToolRequest(
                    call_id="2",
                    capability="filesystem.read",
                    arguments={"path": "main.py"},
                )
            )
            return listing, reading

    listing, reading = run(scenario())
    assert listing.success is True
    assert listing.data == {
        "path": ".",
        "entries": [
            {"name": "main.py", "type": "file"},
            {"name": "src", "type": "directory"},
        ],
    }
    assert reading.success is True
    assert reading.data == {"path": "main.py", "content": "[FILE] main.py\n[DIR] src"}


def test_adapter_normalizes_mcp_errors_malformed_responses_and_timeouts(
    tmp_path: Path,
) -> None:
    (tmp_path / "main.py").write_text("print('ok')", encoding="utf-8")

    async def scenario() -> tuple[ToolResult, ToolResult, ToolResult]:
        error_adapter = make_adapter(
            tmp_path,
            FakeMcpConnection(
                call_result=McpCallResult(
                    is_error=True,
                    content=({"type": "text", "text": "access denied"},),
                    structured_content=None,
                )
            ),
        )
        malformed_adapter = make_adapter(
            tmp_path,
            FakeMcpConnection(
                call_result=McpCallResult(
                    is_error=False, content=(), structured_content={"unexpected": True}
                )
            ),
        )
        timeout_adapter = make_adapter(tmp_path, SlowConnection())
        results: list[ToolResult] = []
        for adapter in (error_adapter, malformed_adapter, timeout_adapter):
            async with adapter:
                registry = ToolRegistry()
                adapter.register_tools(registry)
                results.append(
                    await ToolRuntime(registry).invoke(
                        ToolRequest(
                            call_id="1",
                            capability="filesystem.read",
                            arguments={"path": "main.py"},
                        )
                    )
                )
        return tuple(results)  # type: ignore[return-value]

    class SlowConnection(FakeMcpConnection):
        async def call_tool(
            self, name: str, arguments: dict[str, object]
        ) -> McpCallResult:
            await asyncio.sleep(0.2)
            return await super().call_tool(name, arguments)

    error, malformed, timeout = run(scenario())
    assert error.error is not None and error.error.code == "mcp_tool_error"
    assert malformed.error is not None and malformed.error.code == "malformed_response"
    assert timeout.error is not None and timeout.error.code == "mcp_timeout"


def test_unexpected_adapter_errors_propagate(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("print('ok')", encoding="utf-8")
    adapter = make_adapter(tmp_path, ExplodingConnection())

    async def scenario() -> None:
        async with adapter:
            registry = ToolRegistry()
            adapter.register_tools(registry)
            await ToolRuntime(registry).invoke(
                ToolRequest(
                    call_id="1",
                    capability="filesystem.read",
                    arguments={"path": "main.py"},
                )
            )

    with pytest.raises(RuntimeError, match="programming defect"):
        run(scenario())


def test_adapter_rejects_files_over_read_limit_before_mcp_call(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("012345", encoding="utf-8")
    connection = FakeMcpConnection(
        call_result=McpCallResult(
            is_error=False, content=(), structured_content={"content": "unused"}
        )
    )
    adapter = make_adapter(tmp_path, connection, max_read_bytes=3)

    async def scenario() -> ToolResult:
        async with adapter:
            registry = ToolRegistry()
            adapter.register_tools(registry)
            return await ToolRuntime(registry).invoke(
                ToolRequest(
                    call_id="1",
                    capability="filesystem.read",
                    arguments={"path": "main.py"},
                )
            )

    result = run(scenario())
    assert result.error is not None and result.error.code == "file_too_large"
    assert connection.calls == []
