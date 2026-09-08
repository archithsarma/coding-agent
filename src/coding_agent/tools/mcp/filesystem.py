"""Filesystem tools backed by the official MCP filesystem server."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from coding_agent.content import sha256_text
from coding_agent.domain import (
    InvalidRequestError,
    ToolRequest,
    ToolResult,
)
from coding_agent.policies import WorkspacePathPolicy
from coding_agent.tools.descriptor import ToolDescriptor
from coding_agent.tools.mcp.adapter import McpAdapter, _extract_text, _failure
from coding_agent.tools.mcp.client import (
    McpCallResult,
    McpConnection,
)

if TYPE_CHECKING:
    from pydantic import JsonValue

    from coding_agent.tools.protocol import InternalTool


class FilesystemMcpAdapter(McpAdapter):
    required_external_tools = frozenset(
        {"read_text_file", "list_directory", "get_file_info", "write_file"}
    )

    def __init__(
        self,
        connection: McpConnection,
        *,
        path_policy: WorkspacePathPolicy,
        max_read_bytes: int,
        max_write_bytes: int,
        operation_timeout_seconds: float,
    ) -> None:
        super().__init__(
            connection, operation_timeout_seconds=operation_timeout_seconds
        )
        self._path_policy = path_policy
        self._max_read_bytes = max_read_bytes
        if max_write_bytes <= 0:
            raise ValueError("max_write_bytes must be greater than zero")
        self._max_write_bytes = max_write_bytes

    def _build_tools(self) -> tuple[InternalTool, ...]:
        return (
            _FilesystemListTool(self),
            _FilesystemReadTool(self),
            _FilesystemWriteTool(self),
        )

    def resolve_path(self, request: ToolRequest) -> tuple[str, Path]:
        raw_path = request.arguments.get("path")
        if not isinstance(raw_path, str):
            raise InvalidRequestError("filesystem requests require a string path")
        return raw_path, self._path_policy.resolve_path(raw_path)


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
        raw_path, path = resolved
        result = await self._adapter._invoke(
            request, "list_directory", {"path": str(path)}
        )
        if isinstance(result, ToolResult):
            return result
        text = _extract_text(result)
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
        raw_path, path = resolved
        metadata = await self._adapter._invoke(
            request, "get_file_info", {"path": str(path)}
        )
        if isinstance(metadata, ToolResult):
            return metadata
        size = _extract_file_size(metadata)
        if size is None:
            return _failure(
                request,
                "malformed_response",
                "filesystem metadata lacked a reliable file size",
            )
        if size > self._adapter._max_read_bytes:
            return _failure(
                request, "file_too_large", "file exceeds configured read limit"
            )
        result = await self._adapter._invoke(
            request, "read_text_file", {"path": str(path)}
        )
        if isinstance(result, ToolResult):
            return result
        text = _extract_text(result)
        if text is None:
            return _failure(
                request, "malformed_response", "filesystem read lacked text content"
            )
        return ToolResult(
            call_id=request.call_id,
            success=True,
            data={"path": raw_path, "content": text},
        )


class _FilesystemWriteTool:
    descriptor = ToolDescriptor(
        tool_name="filesystem-write",
        capability="filesystem.write",
        description="Write bounded UTF-8 text to a file in the configured workspace.",
        mutating=True,
    )

    def __init__(self, adapter: FilesystemMcpAdapter) -> None:
        self._adapter = adapter

    async def execute(self, request: ToolRequest) -> ToolResult:
        raw_path, path = self._adapter.resolve_path(request)
        content = request.arguments.get("content")
        expected_sha256 = request.arguments.get("expected_sha256")
        if not isinstance(content, str) or not isinstance(expected_sha256, str):
            raise InvalidRequestError(
                "filesystem writes require string content and expected_sha256"
            )
        if "\x00" in content:
            raise InvalidRequestError(
                "filesystem write content must not contain null bytes"
            )
        content_bytes = content.encode("utf-8")
        if len(content_bytes) > self._adapter._max_write_bytes:
            return _failure(
                request, "file_too_large", "content exceeds configured write limit"
            )

        metadata = await self._adapter._invoke(
            request, "get_file_info", {"path": str(path)}
        )
        if isinstance(metadata, ToolResult):
            return metadata
        size = _extract_file_size(metadata)
        if size is None:
            return _failure(
                request,
                "malformed_response",
                "filesystem write precondition metadata lacked a reliable file size",
            )
        if size > self._adapter._max_read_bytes:
            return _failure(
                request,
                "file_too_large",
                "file exceeds configured read limit for hash verification",
            )
        current = await self._adapter._invoke(
            request, "read_text_file", {"path": str(path)}
        )
        if isinstance(current, ToolResult):
            return current
        current_text = _extract_text(current)
        if current_text is None:
            return _failure(
                request,
                "malformed_response",
                "filesystem write precondition read lacked text content",
            )
        if sha256_text(current_text) != expected_sha256:
            return _failure(
                request,
                "operation_conflict",
                "file changed before filesystem write",
            )

        result = await self._adapter._invoke(
            request,
            "write_file",
            {"path": str(path), "content": content},
        )
        if isinstance(result, ToolResult):
            return result
        return ToolResult(
            call_id=request.call_id,
            success=True,
            data={"path": raw_path, "bytes_written": len(content_bytes)},
        )


def _extract_file_size(result: McpCallResult) -> int | None:
    structured = result.structured_content
    if isinstance(structured, dict):
        size = structured.get("size")
        if isinstance(size, int) and not isinstance(size, bool) and size >= 0:
            return size

    text = _extract_text(result)
    if text is None:
        return None
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip().lower() == "size":
            try:
                size = int(value.strip())
            except ValueError:
                return None
            return size if size >= 0 else None
    return None


def _entry_name(entry: JsonValue) -> str:
    if isinstance(entry, dict):
        name = entry.get("name")
        if isinstance(name, str):
            return name
    return ""
