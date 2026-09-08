"""Constrained local Shell MCP server using structured argv execution."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path

from mcp.server.lowlevel.server import Server
from mcp.server.stdio import stdio_server
from mcp_types import (
    CallToolRequestParams,
    CallToolResult,
    ListToolsResult,
    TextContent,
    Tool,
)

from coding_agent.domain import PolicyViolationError
from coding_agent.policies.workspace import WorkspacePathResolver
from coding_agent.shell import (
    DEFAULT_MAX_ARGUMENT_LENGTH,
    DEFAULT_MAX_ARGUMENTS,
    DEFAULT_MAX_STDERR_BYTES,
    DEFAULT_MAX_STDOUT_BYTES,
    DEFAULT_MAX_TIMEOUT_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    ShellInvocation,
    ShellValidationError,
    validate_invocation,
)

HARD_MAX_TIMEOUT_SECONDS = DEFAULT_MAX_TIMEOUT_SECONDS
HARD_MAX_OUTPUT_BYTES = 8 * 1_048_576
PROCESS_GRACE_SECONDS = 1.0


@dataclass(frozen=True)
class ShellExecutionResult:
    argv: tuple[str, ...]
    cwd: str
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool
    stdout_truncated: bool
    stderr_truncated: bool


class ShellProcessError(RuntimeError):
    """The approved command could not be spawned or reaped."""


class ShellCommandExecutor:
    def __init__(
        self,
        *,
        max_stdout_bytes: int = DEFAULT_MAX_STDOUT_BYTES,
        max_stderr_bytes: int = DEFAULT_MAX_STDERR_BYTES,
        termination_grace_seconds: float = PROCESS_GRACE_SECONDS,
    ) -> None:
        self._max_stdout_bytes = max_stdout_bytes
        self._max_stderr_bytes = max_stderr_bytes
        self._termination_grace_seconds = termination_grace_seconds

    async def execute(
        self, invocation: ShellInvocation, *, cwd: Path
    ) -> ShellExecutionResult:
        started = time.monotonic()
        try:
            process = await asyncio.create_subprocess_exec(
                *invocation.argv,
                cwd=cwd,
                env=_execution_environment(),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=os.name == "posix",
            )
        except OSError as error:
            raise ShellProcessError("approved command could not be started") from error

        if process.stdout is None or process.stderr is None:  # pragma: no cover
            raise ShellProcessError("shell process did not expose output streams")
        stdout_task = asyncio.create_task(
            _read_limited(process.stdout, self._max_stdout_bytes)
        )
        stderr_task = asyncio.create_task(
            _read_limited(process.stderr, self._max_stderr_bytes)
        )
        timed_out = False
        try:
            try:
                await asyncio.wait_for(
                    process.wait(), timeout=invocation.timeout_seconds
                )
            except TimeoutError:
                timed_out = True
                await self._terminate_and_reap(process)
            except asyncio.CancelledError:
                await asyncio.shield(self._terminate_and_reap(process))
                await asyncio.shield(_await_output(stdout_task, stderr_task))
                raise
            stdout, stderr = await _await_output(stdout_task, stderr_task)
        finally:
            if not stdout_task.done() or not stderr_task.done():
                await _await_output(stdout_task, stderr_task)

        duration_ms = max(0, round((time.monotonic() - started) * 1000))
        return ShellExecutionResult(
            argv=invocation.argv,
            cwd=invocation.cwd,
            exit_code=process.returncode if process.returncode is not None else -1,
            stdout=stdout[0].decode("utf-8", errors="replace"),
            stderr=stderr[0].decode("utf-8", errors="replace"),
            duration_ms=duration_ms,
            timed_out=timed_out,
            stdout_truncated=stdout[1],
            stderr_truncated=stderr[1],
        )

    async def _terminate_and_reap(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:  # pragma: no cover - target environment is Unix
            process.terminate()
        try:
            await asyncio.wait_for(
                process.wait(), timeout=self._termination_grace_seconds
            )
            return
        except TimeoutError:
            pass
        if process.returncode is None:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:  # pragma: no cover - target environment is Unix
                process.kill()
            await process.wait()


async def _read_limited(stream: asyncio.StreamReader, limit: int) -> tuple[bytes, bool]:
    captured = bytearray()
    truncated = False
    while True:
        chunk = await stream.read(65_536)
        if not chunk:
            break
        remaining = limit - len(captured)
        if remaining > 0:
            captured.extend(chunk[:remaining])
        if len(chunk) > max(remaining, 0):
            truncated = True
    return bytes(captured), truncated


async def _await_output(
    stdout_task: asyncio.Task[tuple[bytes, bool]],
    stderr_task: asyncio.Task[tuple[bytes, bool]],
) -> tuple[tuple[bytes, bool], tuple[bytes, bool]]:
    return await asyncio.gather(stdout_task, stderr_task)


def _execution_environment() -> dict[str, str]:
    allowed = {
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TMPDIR",
        "TEMP",
        "TMP",
        "VIRTUAL_ENV",
    }
    return {key: value for key, value in os.environ.items() if key in allowed}


def _tool_definition() -> Tool:
    return Tool(
        name="run_command",
        description="Run one approved development command with structured argv.",
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "argv": {"type": "array", "items": {"type": "string"}},
                "cwd": {"type": "string", "default": "."},
                "timeout_seconds": {
                    "type": "number",
                    "default": DEFAULT_TIMEOUT_SECONDS,
                },
            },
            "required": ["argv"],
        },
    )


def create_server(
    *,
    workspace_root: Path,
    max_timeout_seconds: float,
    max_stdout_bytes: int,
    max_stderr_bytes: int,
) -> Server[dict[str, object]]:
    resolver = WorkspacePathResolver(workspace_root)
    max_timeout = max(0.001, min(max_timeout_seconds, HARD_MAX_TIMEOUT_SECONDS))
    stdout_limit = max(1, min(max_stdout_bytes, HARD_MAX_OUTPUT_BYTES))
    stderr_limit = max(1, min(max_stderr_bytes, HARD_MAX_OUTPUT_BYTES))
    executor = ShellCommandExecutor(
        max_stdout_bytes=stdout_limit,
        max_stderr_bytes=stderr_limit,
    )

    async def list_tools(_ctx: object, _params: object) -> ListToolsResult:
        return ListToolsResult(tools=[_tool_definition()])

    async def call_tool(_ctx: object, params: CallToolRequestParams) -> CallToolResult:
        arguments: dict[str, object] = dict(params.arguments or {})
        try:
            invocation = validate_invocation(
                arguments,
                default_timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
                max_timeout_seconds=max_timeout,
                max_arguments=DEFAULT_MAX_ARGUMENTS,
                max_argument_length=DEFAULT_MAX_ARGUMENT_LENGTH,
            )
            cwd = resolver.resolve(invocation.cwd)
            result = await executor.execute(invocation, cwd=cwd)
        except (ShellValidationError, PolicyViolationError, ValueError) as error:
            return _error_result(str(error))
        except ShellProcessError as error:
            return _error_result(str(error))

        payload = {
            "argv": list(result.argv),
            "cwd": result.cwd,
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration_ms": result.duration_ms,
            "timed_out": result.timed_out,
            "stdout_truncated": result.stdout_truncated,
            "stderr_truncated": result.stderr_truncated,
        }
        return CallToolResult(
            content=[TextContent(text=json.dumps(payload, separators=(",", ":")))],
            structured_content=payload,
            is_error=False,
        )

    return Server(
        "coding-agent-shell",
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )


def _error_result(message: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(text=message)], is_error=True)


async def _run_server(args: argparse.Namespace) -> None:
    server = create_server(
        workspace_root=Path(args.workspace_root),
        max_timeout_seconds=args.max_timeout_seconds,
        max_stdout_bytes=args.max_stdout_bytes,
        max_stderr_bytes=args.max_stderr_bytes,
    )
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument(
        "--max-timeout-seconds", type=float, default=HARD_MAX_TIMEOUT_SECONDS
    )
    parser.add_argument(
        "--max-stdout-bytes", type=int, default=DEFAULT_MAX_STDOUT_BYTES
    )
    parser.add_argument(
        "--max-stderr-bytes", type=int, default=DEFAULT_MAX_STDERR_BYTES
    )
    asyncio.run(_run_server(parser.parse_args()))


if __name__ == "__main__":
    main()
