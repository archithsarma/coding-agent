"""Policy for the controlled shell.execute capability."""

from __future__ import annotations

from typing import TYPE_CHECKING

from coding_agent.domain import PolicyViolationError, ToolRequest
from coding_agent.policies.workspace import WorkspacePathPolicy
from coding_agent.shell import (
    ALLOWED_EXECUTABLES,
    ShellValidationError,
    validate_invocation,
)

if TYPE_CHECKING:
    from coding_agent.tools.descriptor import ToolDescriptor


class ShellCommandPolicy:
    def __init__(
        self,
        *,
        default_timeout_seconds: float = 30.0,
        max_timeout_seconds: float = 120.0,
        max_arguments: int = 64,
        max_argument_length: int = 4096,
        path_policy: WorkspacePathPolicy | None = None,
        allowed_executables: tuple[str, ...] = ("pytest", "ruff", "mypy", "git"),
    ) -> None:
        self._default_timeout_seconds = default_timeout_seconds
        self._max_timeout_seconds = max_timeout_seconds
        self._max_arguments = max_arguments
        self._max_argument_length = max_argument_length
        self._path_policy = path_policy
        if (
            not allowed_executables
            or not set(allowed_executables) <= ALLOWED_EXECUTABLES
        ):
            raise ValueError(
                "allowed_executables may only narrow the built-in allowlist"
            )
        self._allowed_executables = allowed_executables

    async def validate(self, descriptor: ToolDescriptor, request: ToolRequest) -> None:
        if descriptor.capability != "shell.execute":
            return
        try:
            validate_invocation(
                request.arguments,
                default_timeout_seconds=self._default_timeout_seconds,
                max_timeout_seconds=self._max_timeout_seconds,
                max_arguments=self._max_arguments,
                max_argument_length=self._max_argument_length,
                allowed_executables=self._allowed_executables,
            )
        except ShellValidationError as error:
            raise PolicyViolationError(str(error)) from error
        if self._path_policy is not None:
            cwd = request.arguments.get("cwd", ".")
            if not isinstance(cwd, str):
                raise PolicyViolationError("cwd must be a string")
            self._path_policy.resolve_path(cwd)
