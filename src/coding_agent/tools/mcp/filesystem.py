"""Read-only filesystem tools backed by the official MCP filesystem server."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING

from coding_agent.domain import (
    PolicyViolationError,
    ToolErrorInfo,
    ToolRequest,
    ToolResult,
)
from coding_agent.policies import WorkspacePathPolicy
from coding_agent.tools.descriptor import ToolDescriptor
from coding_agent.tools.mcp.client import (
    McpCallResult,
    McpConnection,
    McpToolDescription,
)
from coding_agent.tools.mcp.errors import (
    McpIntegrationError,
    McpResponseError,
    McpTimeoutError,
    McpToolUnavailableError,
)

if TYPE_CHECKING:
    from pydantic import JsonValue

    from coding_agent.tools.protocol import InternalTool
    from coding_agent.tools.registry import ToolRegistry


class FilesystemMcpAdapter:
    def __init__(
        self,
        connection: McpConnection,
        *,
        path_policy: WorkspacePathPolicy,
        max_read_bytes: int,
        operation_timeout_seconds: float,
    ) -> None:
        self._connection = connection
        self._path_policy = path_policy
        self._max_read_bytes = max_read_bytes
        self._timeout = operation_timeout_seconds
        self._exit_stack: AsyncExitStack | None = None
        self._tools: tuple[InternalTool, ...] = ()

    async def __aenter__(self) -> FilesystemMcpAdapter:
        stack = AsyncExitStack()
        await stack.__aenter__()
        try:
            connection = await stack.enter_async_context(self._connection)
            available = {tool.name for tool in await self._discover(connection)}
            required = {"read_text_file", "list_directory"}
            missing = required - available
            if missing:
                raise McpToolUnavailableError(
                    "MCP filesystem server is missing tools: "
                    f"{', '.join(sorted(missing))}"
                )
            self._tools = (
                _FilesystemListTool(self),
                _FilesystemReadTool(self),
            )
            self._exit_stack = stack
            return self
        except BaseException:
            await stack.__aexit__(None, None, None)
            raise

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._exit_stack is not None:
            await self._exit_stack.__aexit__(exc_type, exc_value, traceback)
            self._exit_stack = None
            self._tools = ()

    def register_tools(self, registry: ToolRegistry) -> None:
        if self._exit_stack is None:
            raise RuntimeError(
                "filesystem adapter must be connected before registration"
            )
        for tool in self._tools:
            registry.register(tool)

    async def _discover(
        self, connection: McpConnection
    ) -> tuple[McpToolDescription, ...]:
        try:
            async with asyncio.timeout(self._timeout):
                return await connection.list_tools()
        except TimeoutError as error:
            raise McpTimeoutError("MCP tool discovery timed out") from error

    async def _invoke(
        self,
        request: ToolRequest,
        external_name: str,
        path: Path,
    ) -> McpCallResult | ToolResult:
        try:
            async with asyncio.timeout(self._timeout):
                result = await self._connection.call_tool(
                    external_name, {"path": str(path)}
                )
                if result.is_error:
                    return _failure(
                        request,
                        "mcp_tool_error",
                        _error_text(result) or f"MCP tool '{external_name}' failed",
                    )
                return result
        except TimeoutError:
            return _failure(
                request, "mcp_timeout", f"MCP tool '{external_name}' timed out"
            )
        except McpTimeoutError as error:
            return _failure(request, "mcp_timeout", str(error))
        except McpResponseError as error:
            return _failure(request, "malformed_response", str(error))
        except McpIntegrationError as error:
            return _failure(request, "mcp_tool_error", str(error))

    def resolve_path(self, request: ToolRequest) -> tuple[str, Path] | ToolResult:
        raw_path = request.arguments.get("path")
        if not isinstance(raw_path, str):
            return _failure(
                request, "invalid_path", "filesystem requests require a string path"
            )
        try:
            return raw_path, self._path_policy.resolver.resolve(raw_path)
        except PolicyViolationError as error:
            return _failure(request, "policy_violation", str(error))


class _FilesystemListTool:
    descriptor = ToolDescriptor(
        tool_name="filesystem-list",
        capability="filesystem.list",
        description="List entries in the configured workspace.",
        mutating=False,
    )

    def __init__(self, adapter: FilesystemMcpAdapter) -> None:
        self._adapter = adapter

    async def execute(self, request: ToolRequest) -> ToolResult:
        resolved = self._adapter.resolve_path(request)
        if isinstance(resolved, ToolResult):
            return resolved
        raw_path, path = resolved
        result = await self._adapter._invoke(request, "list_directory", path)
        if isinstance(result, ToolResult):
            return result
        text = _result_text(result)
        if text is None:
            return _failure(
                request, "malformed_response", "filesystem listing lacked text content"
            )
        entries: list[JsonValue] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            if line.startswith("[FILE] "):
                entries.append({"name": line[7:], "type": "file"})
            elif line.startswith("[DIR] "):
                entries.append({"name": line[6:], "type": "directory"})
            else:
                return _failure(
                    request,
                    "malformed_response",
                    "filesystem listing had an invalid entry",
                )
        entries.sort(key=_entry_name)
        return ToolResult(
            call_id=request.call_id,
            success=True,
            data={"path": raw_path, "entries": entries},
        )


class _FilesystemReadTool:
    descriptor = ToolDescriptor(
        tool_name="filesystem-read",
        capability="filesystem.read",
        description="Read a UTF-8 text file in the configured workspace.",
        mutating=False,
    )

    def __init__(self, adapter: FilesystemMcpAdapter) -> None:
        self._adapter = adapter

    async def execute(self, request: ToolRequest) -> ToolResult:
        resolved = self._adapter.resolve_path(request)
        if isinstance(resolved, ToolResult):
            return resolved
        raw_path, path = resolved
        try:
            if path.stat().st_size > self._adapter._max_read_bytes:
                return _failure(
                    request, "file_too_large", "file exceeds configured read limit"
                )
        except OSError:
            pass
        result = await self._adapter._invoke(request, "read_text_file", path)
        if isinstance(result, ToolResult):
            return result
        text = _result_text(result)
        if text is None:
            return _failure(
                request, "malformed_response", "filesystem read lacked text content"
            )
        return ToolResult(
            call_id=request.call_id,
            success=True,
            data={"path": raw_path, "content": text},
        )


def _result_text(result: McpCallResult) -> str | None:
    if isinstance(result.structured_content, dict):
        content = result.structured_content.get("content")
        if isinstance(content, str):
            return content
    for item in result.content:
        if isinstance(item, dict) and item.get("type") == "text":
            text = item.get("text")
            if isinstance(text, str):
                return text
    return None


def _error_text(result: McpCallResult) -> str | None:
    if isinstance(result.structured_content, dict):
        content = result.structured_content.get("content")
        if isinstance(content, str):
            return content
    for item in result.content:
        if isinstance(item, dict) and item.get("type") == "text":
            text = item.get("text")
            if isinstance(text, str):
                return text
    return None


def _entry_name(entry: JsonValue) -> str:
    if isinstance(entry, dict):
        name = entry.get("name")
        if isinstance(name, str):
            return name
    return ""


def _failure(request: ToolRequest, code: str, message: str) -> ToolResult:
    return ToolResult(
        call_id=request.call_id,
        success=False,
        error=ToolErrorInfo(code=code, message=message),
    )
