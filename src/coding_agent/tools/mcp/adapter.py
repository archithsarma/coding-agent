"""Shared lifecycle and invocation boundary for MCP-backed tool adapters."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from contextlib import AsyncExitStack
from types import TracebackType
from typing import TYPE_CHECKING, ClassVar

from coding_agent.domain import ToolErrorInfo, ToolRequest, ToolResult
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


class McpAdapter(ABC):
    required_external_tools: ClassVar[frozenset[str]]

    def __init__(
        self, connection: McpConnection, *, operation_timeout_seconds: float
    ) -> None:
        self._connection = connection
        self._timeout = operation_timeout_seconds
        self._exit_stack: AsyncExitStack | None = None
        self._active_connection: McpConnection | None = None
        self._tools: tuple[InternalTool, ...] = ()

    async def __aenter__(self) -> McpAdapter:
        if self._exit_stack is not None or self._active_connection is not None:
            raise RuntimeError("MCP adapter is already connected")
        stack = AsyncExitStack()
        await stack.__aenter__()
        try:
            connection = await stack.enter_async_context(self._connection)
            self._active_connection = connection
            available = {tool.name for tool in await self._discover(connection)}
            missing = self.required_external_tools - available
            if missing:
                raise McpToolUnavailableError(
                    "MCP server is missing required tools: "
                    f"{', '.join(sorted(missing))}"
                )
            self._tools = self._build_tools()
            self._exit_stack = stack
            return self
        except BaseException as error:
            self._active_connection = None
            self._exit_stack = None
            self._tools = ()
            try:
                await stack.__aexit__(None, None, None)
            except BaseException as cleanup_error:
                error.add_note(f"MCP adapter startup cleanup failed: {cleanup_error}")
            raise

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        stack = self._exit_stack
        self._exit_stack = None
        self._active_connection = None
        self._tools = ()
        if stack is not None:
            await stack.__aexit__(exc_type, exc_value, traceback)

    def register_tools(self, registry: ToolRegistry) -> None:
        if self._exit_stack is None:
            raise RuntimeError("MCP adapter must be connected before registration")
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
        arguments: dict[str, JsonValue],
    ) -> McpCallResult | ToolResult:
        connection = self._active_connection
        if connection is None:
            raise RuntimeError("MCP adapter is disconnected")
        try:
            async with asyncio.timeout(self._timeout):
                result = await connection.call_tool(external_name, arguments)
                if result.is_error:
                    return _failure(
                        request,
                        "mcp_tool_error",
                        _extract_text(result) or f"MCP tool '{external_name}' failed",
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

    @abstractmethod
    def _build_tools(self) -> tuple[InternalTool, ...]: ...


def _extract_text(result: McpCallResult) -> str | None:
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


def _failure(request: ToolRequest, code: str, message: str) -> ToolResult:
    return ToolResult(
        call_id=request.call_id,
        success=False,
        error=ToolErrorInfo(code=code, message=message),
    )
