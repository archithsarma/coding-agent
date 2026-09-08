"""Agent-facing shell.execute adapter backed by the local Shell MCP server."""

from __future__ import annotations

from typing import TYPE_CHECKING

from coding_agent.config import ShellMcpSettings
from coding_agent.domain import (
    PolicyViolationError,
    ToolErrorInfo,
    ToolRequest,
    ToolResult,
)
from coding_agent.policies import WorkspacePathPolicy
from coding_agent.shell import ShellValidationError, validate_invocation
from coding_agent.tools.descriptor import ToolDescriptor
from coding_agent.tools.mcp.adapter import McpAdapter, _failure
from coding_agent.tools.mcp.client import McpCallResult, McpConnection

if TYPE_CHECKING:
    from pydantic import JsonValue

    from coding_agent.tools.protocol import InternalTool


class ShellMcpAdapter(McpAdapter):
    required_external_tools = frozenset({"run_command"})

    def __init__(
        self,
        connection: McpConnection,
        *,
        path_policy: WorkspacePathPolicy,
        settings: ShellMcpSettings,
    ) -> None:
        super().__init__(
            connection,
            operation_timeout_seconds=settings.operation_timeout_seconds,
        )
        self._path_policy = path_policy
        self._settings = settings

    def _build_tools(self) -> tuple[InternalTool, ...]:
        return (_ShellExecuteTool(self),)

    def _validate_request(self, request: ToolRequest) -> tuple[list[str], str, float]:
        try:
            invocation = validate_invocation(
                request.arguments,
                default_timeout_seconds=self._settings.default_timeout_seconds,
                max_timeout_seconds=self._settings.max_timeout_seconds,
                max_arguments=self._settings.max_arguments,
                max_argument_length=self._settings.max_argument_length,
                allowed_executables=self._settings.allowed_executables,
            )
        except ShellValidationError as error:
            raise PolicyViolationError(str(error)) from error
        self._path_policy.resolve_path(invocation.cwd)
        return list(invocation.argv), invocation.cwd, invocation.timeout_seconds


class _ShellExecuteTool:
    descriptor = ToolDescriptor(
        tool_name="shell-execute",
        capability="shell.execute",
        description="Run an approved development command in the workspace.",
        mutating=True,
    )

    def __init__(self, adapter: ShellMcpAdapter) -> None:
        self._adapter = adapter

    async def execute(self, request: ToolRequest) -> ToolResult:
        argv, cwd, timeout_seconds = self._adapter._validate_request(request)
        argv_payload: list[JsonValue] = list(argv)
        result = await self._adapter._invoke(
            request,
            "run_command",
            {
                "argv": argv_payload,
                "cwd": cwd,
                "timeout_seconds": timeout_seconds,
            },
        )
        if isinstance(result, ToolResult):
            return result
        payload = _normalize_result(result, request, argv, cwd)
        if isinstance(payload, ToolResult):
            return payload
        if payload.get("timed_out") is True:
            return ToolResult(
                call_id=request.call_id,
                success=False,
                error=ToolErrorInfo(
                    code="shell_timeout",
                    message="shell command timed out",
                    details=payload,
                ),
            )
        return ToolResult(call_id=request.call_id, success=True, data=payload)


def _normalize_result(
    result: McpCallResult,
    request: ToolRequest,
    argv: list[str],
    cwd: str,
) -> dict[str, JsonValue] | ToolResult:
    structured = result.structured_content
    if not isinstance(structured, dict):
        return _failure(
            request, "malformed_response", "shell response lacked structured data"
        )
    expected_keys = {
        "argv",
        "cwd",
        "exit_code",
        "stdout",
        "stderr",
        "duration_ms",
        "timed_out",
        "stdout_truncated",
        "stderr_truncated",
    }
    if set(structured) != expected_keys:
        return _failure(
            request, "malformed_response", "shell response had an invalid schema"
        )
    exit_code = structured.get("exit_code")
    argv_payload: list[JsonValue] = list(argv)
    stdout = structured.get("stdout")
    stderr = structured.get("stderr")
    duration_ms = structured.get("duration_ms")
    timed_out = structured.get("timed_out")
    stdout_truncated = structured.get("stdout_truncated")
    stderr_truncated = structured.get("stderr_truncated")
    if (
        structured.get("argv") != argv
        or structured.get("cwd") != cwd
        or not isinstance(exit_code, int)
        or isinstance(exit_code, bool)
        or not isinstance(stdout, str)
        or not isinstance(stderr, str)
        or not isinstance(duration_ms, int)
        or isinstance(duration_ms, bool)
        or duration_ms < 0
        or not isinstance(timed_out, bool)
        or not isinstance(stdout_truncated, bool)
        or not isinstance(stderr_truncated, bool)
    ):
        return _failure(
            request, "malformed_response", "shell response contained invalid fields"
        )
    return {
        "argv": argv_payload,
        "cwd": cwd,
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "duration_ms": duration_ms,
        "timed_out": timed_out,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
    }
