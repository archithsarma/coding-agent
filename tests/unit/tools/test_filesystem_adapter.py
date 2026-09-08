import asyncio
from collections.abc import Awaitable
from pathlib import Path
from typing import TypeVar

import pytest

from coding_agent.content import sha256_text
from coding_agent.domain import (
    InvalidRequestError,
    PolicyViolationError,
    ToolRequest,
    ToolResult,
)
from coding_agent.policies import PolicyChain, WorkspacePathPolicy
from coding_agent.tools import ToolRegistry, ToolRuntime
from coding_agent.tools.mcp import (
    FilesystemMcpAdapter,
    McpCallResult,
    McpToolDescription,
)
from coding_agent.tools.mcp.errors import McpToolUnavailableError

ResultT = TypeVar("ResultT")


class FakeMcpConnection:
    def __init__(
        self,
        *,
        call_result: McpCallResult | None = None,
        call_results: dict[str, McpCallResult] | None = None,
        tool_names: tuple[str, ...] = (
            "read_text_file",
            "list_directory",
            "get_file_info",
            "write_file",
        ),
    ) -> None:
        self.call_result = call_result
        self.call_results = call_results
        self.tool_names = tool_names
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def __aenter__(self) -> "FakeMcpConnection":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def list_tools(self) -> tuple[McpToolDescription, ...]:
        return tuple(
            McpToolDescription(name=name, description=name) for name in self.tool_names
        )

    async def call_tool(self, name: str, arguments: dict[str, object]) -> McpCallResult:
        self.calls.append((name, arguments))
        if self.call_results is not None:
            return self.call_results[name]
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
        max_write_bytes=100,
        operation_timeout_seconds=0.1,
    )


def result_map(
    *,
    listing: McpCallResult | None = None,
    metadata: McpCallResult | None = None,
    reading: McpCallResult | None = None,
) -> dict[str, McpCallResult]:
    return {
        "list_directory": listing
        or McpCallResult(False, (), {"content": "[FILE] main.py"}),
        "get_file_info": metadata or McpCallResult(False, (), {"content": "size: 1"}),
        "read_text_file": reading or McpCallResult(False, (), {"content": "ok"}),
        "write_file": McpCallResult(False, (), {"content": "written"}),
    }


def invoke_read(adapter: FilesystemMcpAdapter, path: str = "main.py") -> ToolResult:
    async def scenario() -> ToolResult:
        async with adapter:
            registry = ToolRegistry()
            adapter.register_tools(registry)
            return await ToolRuntime(registry).invoke(
                ToolRequest(
                    call_id="1", capability="filesystem.read", arguments={"path": path}
                )
            )

    return run(scenario())


def test_adapter_registers_read_list_and_write_capabilities(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path, FakeMcpConnection(call_results=result_map()))

    async def scenario() -> tuple[str, ...]:
        async with adapter:
            registry = ToolRegistry()
            adapter.register_tools(registry)
            return tuple(d.capability for d in registry.list_descriptors())

    assert run(scenario()) == (
        "filesystem.list",
        "filesystem.read",
        "filesystem.write",
    )


def test_adapter_normalizes_list_and_read_results(tmp_path: Path) -> None:
    connection = FakeMcpConnection(
        call_results=result_map(
            listing=McpCallResult(
                False,
                ({"type": "text", "text": "[FILE] main.py\n[DIR] src"},),
                {"content": "[FILE] main.py\n[DIR] src"},
            ),
            metadata=McpCallResult(False, (), {"content": "size: 42"}),
            reading=McpCallResult(
                False,
                ({"type": "text", "text": "print('ok')"},),
                {"content": "print('ok')"},
            ),
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
    assert reading.data == {"path": "main.py", "content": "print('ok')"}


def test_adapter_normalizes_write_and_enforces_expected_hash(tmp_path: Path) -> None:
    connection = FakeMcpConnection(
        call_results=result_map(
            reading=McpCallResult(False, (), {"content": "before\n"}),
        )
    )
    adapter = make_adapter(tmp_path, connection)

    async def scenario() -> ToolResult:
        async with adapter:
            registry = ToolRegistry()
            adapter.register_tools(registry)
            return await ToolRuntime(registry).invoke(
                ToolRequest(
                    call_id="write",
                    capability="filesystem.write",
                    arguments={
                        "path": "main.py",
                        "content": "after\n",
                        "expected_sha256": sha256_text("before\n"),
                    },
                )
            )

    result = run(scenario())
    assert result.success
    assert result.data == {"path": "main.py", "bytes_written": 6}
    assert [name for name, _ in connection.calls] == [
        "get_file_info",
        "read_text_file",
        "write_file",
    ]


def test_adapter_write_rejects_hash_conflict_without_writing(tmp_path: Path) -> None:
    connection = FakeMcpConnection(
        call_results=result_map(
            reading=McpCallResult(False, (), {"content": "current\n"}),
        )
    )
    adapter = make_adapter(tmp_path, connection)

    async def scenario() -> ToolResult:
        async with adapter:
            registry = ToolRegistry()
            adapter.register_tools(registry)
            return await ToolRuntime(registry).invoke(
                ToolRequest(
                    call_id="write",
                    capability="filesystem.write",
                    arguments={
                        "path": "main.py",
                        "content": "after\n",
                        "expected_sha256": sha256_text("before\n"),
                    },
                )
            )

    result = run(scenario())
    assert result.error is not None and result.error.code == "operation_conflict"
    assert [name for name, _ in connection.calls] == [
        "get_file_info",
        "read_text_file",
    ]


def test_adapter_normalizes_mcp_errors_malformed_responses_and_timeouts(
    tmp_path: Path,
) -> None:
    class SlowConnection(FakeMcpConnection):
        async def call_tool(
            self, name: str, arguments: dict[str, object]
        ) -> McpCallResult:
            await asyncio.sleep(0.2)
            return await super().call_tool(name, arguments)

    adapters = (
        make_adapter(
            tmp_path,
            FakeMcpConnection(
                call_results=result_map(
                    metadata=McpCallResult(
                        True, ({"type": "text", "text": "access denied"},), None
                    )
                )
            ),
        ),
        make_adapter(
            tmp_path,
            FakeMcpConnection(
                call_results=result_map(
                    metadata=McpCallResult(False, (), {"unexpected": True})
                )
            ),
        ),
        make_adapter(tmp_path, SlowConnection(call_results=result_map())),
    )
    results = tuple(invoke_read(adapter) for adapter in adapters)
    error, malformed, timeout = results
    assert error.error is not None and error.error.code == "mcp_tool_error"
    assert malformed.error is not None and malformed.error.code == "malformed_response"
    assert timeout.error is not None and timeout.error.code == "mcp_timeout"


def test_unexpected_adapter_errors_propagate(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="programming defect"):
        invoke_read(make_adapter(tmp_path, ExplodingConnection()))


def test_adapter_rejects_oversized_metadata_without_read(tmp_path: Path) -> None:
    connection = FakeMcpConnection(
        call_results=result_map(metadata=McpCallResult(False, (), {"size": 4}))
    )
    result = invoke_read(make_adapter(tmp_path, connection, max_read_bytes=3))
    assert result.error is not None and result.error.code == "file_too_large"
    assert [name for name, _ in connection.calls] == ["get_file_info"]


def test_active_entered_connection_is_used(tmp_path: Path) -> None:
    active = FakeMcpConnection(call_results=result_map())

    class ContextManager:
        async def __aenter__(self) -> FakeMcpConnection:
            return active

        async def __aexit__(self, *args: object) -> None:
            return None

        async def list_tools(self) -> tuple[McpToolDescription, ...]:
            raise AssertionError("discovery used the context manager")

        async def call_tool(
            self, name: str, arguments: dict[str, object]
        ) -> McpCallResult:
            raise AssertionError("invocation used the context manager")

    result = invoke_read(make_adapter(tmp_path, ContextManager()))
    assert result.success is True
    assert [name for name, _ in active.calls] == ["get_file_info", "read_text_file"]


def test_disconnected_invocation_is_rejected_and_double_enter_is_guarded(
    tmp_path: Path,
) -> None:
    adapter = make_adapter(tmp_path, FakeMcpConnection(call_results=result_map()))

    async def scenario() -> None:
        async with adapter:
            registry = ToolRegistry()
            adapter.register_tools(registry)
            tool = registry.resolve_by_capability("filesystem.read")
            with pytest.raises(RuntimeError, match="already connected"):
                await adapter.__aenter__()
        with pytest.raises(RuntimeError, match="disconnected"):
            await tool.execute(
                ToolRequest(
                    call_id="1", capability="filesystem.read", arguments={"path": "x"}
                )
            )

    run(scenario())


def test_policy_violation_propagates(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path, FakeMcpConnection(call_results=result_map()))

    async def scenario() -> None:
        async with adapter:
            registry = ToolRegistry()
            adapter.register_tools(registry)
            runtime = ToolRuntime(
                registry, policies=PolicyChain([WorkspacePathPolicy(tmp_path)])
            )
            with pytest.raises(PolicyViolationError):
                await runtime.invoke(
                    ToolRequest(
                        call_id="1",
                        capability="filesystem.read",
                        arguments={"path": "../secret"},
                    )
                )

    run(scenario())


def test_malformed_path_argument_uses_invalid_request_semantics(tmp_path: Path) -> None:
    adapter = make_adapter(tmp_path, FakeMcpConnection(call_results=result_map()))

    async def scenario() -> None:
        async with adapter:
            registry = ToolRegistry()
            adapter.register_tools(registry)
            with pytest.raises(InvalidRequestError):
                await registry.resolve_by_capability("filesystem.read").execute(
                    ToolRequest(
                        call_id="1", capability="filesystem.read", arguments={"path": 3}
                    )
                )

    run(scenario())


def test_metadata_lookup_failure_does_not_read(tmp_path: Path) -> None:
    connection = FakeMcpConnection(
        call_results=result_map(
            metadata=McpCallResult(True, ({"type": "text", "text": "failed"},), None)
        )
    )
    result = invoke_read(make_adapter(tmp_path, connection))
    assert result.error is not None and result.error.code == "mcp_tool_error"
    assert [name for name, _ in connection.calls] == ["get_file_info"]


def test_missing_metadata_tool_fails_initialization(tmp_path: Path) -> None:
    adapter = make_adapter(
        tmp_path, FakeMcpConnection(tool_names=("read_text_file", "list_directory"))
    )
    with pytest.raises(McpToolUnavailableError, match="get_file_info"):
        run(adapter.__aenter__())


def test_missing_write_tool_fails_initialization(tmp_path: Path) -> None:
    adapter = make_adapter(
        tmp_path,
        FakeMcpConnection(
            tool_names=("read_text_file", "list_directory", "get_file_info")
        ),
    )
    with pytest.raises(McpToolUnavailableError, match="write_file"):
        run(adapter.__aenter__())
