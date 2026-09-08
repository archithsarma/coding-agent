"""MCP SDK boundary for a single stdio server connection."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Protocol

from mcp import Client, MCPError, StdioServerParameters
from pydantic import JsonValue, TypeAdapter, ValidationError

from coding_agent.tools.mcp.errors import (
    McpConnectionError,
    McpResponseError,
    McpTimeoutError,
)

_JSON_VALUE: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)


@dataclass(frozen=True)
class McpToolDescription:
    name: str
    description: str


@dataclass(frozen=True)
class McpCallResult:
    is_error: bool
    content: tuple[JsonValue, ...]
    structured_content: JsonValue | None


class McpConnection(Protocol):
    """Small internal protocol implemented by the real and fake MCP clients."""

    async def __aenter__(self) -> McpConnection: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    async def list_tools(self) -> tuple[McpToolDescription, ...]: ...

    async def call_tool(
        self, name: str, arguments: dict[str, JsonValue]
    ) -> McpCallResult: ...


class McpServerSettings(Protocol):
    mcp_command: str
    operation_timeout_seconds: float

    def resolved_workspace_root(self) -> Path: ...

    def server_arguments(self, workspace_root: Path | None = None) -> list[str]: ...


class StdioMcpClient(McpConnection):
    def __init__(
        self, settings: McpServerSettings, *, workspace_root: Path | None = None
    ) -> None:
        self._settings = settings
        self._workspace_root = workspace_root or settings.resolved_workspace_root()
        self._client: Client | None = None

    async def __aenter__(self) -> StdioMcpClient:
        if self._client is not None:
            raise RuntimeError("MCP client is already connected")
        parameters = StdioServerParameters(
            command=self._settings.mcp_command,
            args=self._settings.server_arguments(self._workspace_root),
        )
        client = Client(
            parameters,
            read_timeout_seconds=self._settings.operation_timeout_seconds,
        )
        try:
            async with asyncio.timeout(self._settings.operation_timeout_seconds):
                await client.__aenter__()
        except BaseException as error:
            try:
                await client.__aexit__(None, None, None)
            except BaseException as cleanup_error:
                error.add_note(f"MCP startup cleanup failed: {cleanup_error}")
            if isinstance(error, TimeoutError):
                raise McpTimeoutError("MCP server startup timed out") from error
            if isinstance(error, (MCPError, OSError, ValueError)):
                raise McpConnectionError("MCP server connection failed") from error
            raise
        self._client = client
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        client = self._client
        self._client = None
        if client is not None:
            await client.__aexit__(exc_type, exc_value, traceback)

    async def list_tools(self) -> tuple[McpToolDescription, ...]:
        client = self._require_client()
        try:
            async with asyncio.timeout(self._settings.operation_timeout_seconds):
                response = await client.list_tools()
        except TimeoutError as error:
            raise McpTimeoutError("MCP tool discovery timed out") from error
        except MCPError as error:
            raise McpConnectionError("MCP tool discovery failed") from error
        return tuple(
            McpToolDescription(name=tool.name, description=tool.description or "")
            for tool in response.tools
        )

    async def call_tool(
        self, name: str, arguments: dict[str, JsonValue]
    ) -> McpCallResult:
        client = self._require_client()
        try:
            async with asyncio.timeout(self._settings.operation_timeout_seconds):
                response = await client.call_tool(name, arguments)
        except TimeoutError as error:
            raise McpTimeoutError(f"MCP tool '{name}' timed out") from error
        except MCPError as error:
            raise McpConnectionError(f"MCP tool '{name}' failed") from error

        try:
            content = tuple(
                _JSON_VALUE.validate_python(
                    item.model_dump(mode="json", by_alias=True, exclude_none=True)
                )
                for item in response.content
            )
            structured = (
                None
                if response.structured_content is None
                else _JSON_VALUE.validate_python(response.structured_content)
            )
        except (ValidationError, TypeError, ValueError) as error:
            raise McpResponseError(
                "MCP tool response was not JSON-compatible"
            ) from error
        return McpCallResult(
            is_error=response.is_error,
            content=content,
            structured_content=structured,
        )

    def _require_client(self) -> Client:
        if self._client is None:
            raise McpConnectionError("MCP client is not connected")
        return self._client
