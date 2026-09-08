"""Validated configuration for the controlled shell integration."""

import sys
from pathlib import Path

from pydantic import Field, field_validator, model_validator

from coding_agent.domain.models import DomainModel
from coding_agent.shell import (
    ALLOWED_EXECUTABLES,
    DEFAULT_MAX_ARGUMENT_LENGTH,
    DEFAULT_MAX_ARGUMENTS,
    DEFAULT_MAX_STDERR_BYTES,
    DEFAULT_MAX_STDOUT_BYTES,
    DEFAULT_MAX_TIMEOUT_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
)


def _non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("value must not be blank")
    return value


class ShellMcpSettings(DomainModel):
    workspace_root: Path
    mcp_command: str = sys.executable
    server_args: tuple[str, ...] = ("-m", "coding_agent.mcp_servers.shell")
    operation_timeout_seconds: float = Field(
        default=DEFAULT_MAX_TIMEOUT_SECONDS + 5, gt=0
    )
    default_timeout_seconds: float = Field(default=DEFAULT_TIMEOUT_SECONDS, gt=0)
    max_timeout_seconds: float = Field(default=DEFAULT_MAX_TIMEOUT_SECONDS, gt=0)
    max_stdout_bytes: int = Field(default=DEFAULT_MAX_STDOUT_BYTES, gt=0)
    max_stderr_bytes: int = Field(default=DEFAULT_MAX_STDERR_BYTES, gt=0)
    max_arguments: int = Field(default=DEFAULT_MAX_ARGUMENTS, gt=0)
    max_argument_length: int = Field(default=DEFAULT_MAX_ARGUMENT_LENGTH, gt=0)
    allowed_executables: tuple[str, ...] = tuple(sorted(ALLOWED_EXECUTABLES))

    _validate_command = field_validator("mcp_command")(_non_blank)

    @field_validator("allowed_executables")
    @classmethod
    def validate_allowed_executables(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(item not in ALLOWED_EXECUTABLES for item in value):
            raise ValueError(
                "allowed_executables may only narrow the built-in allowlist"
            )
        return value

    @model_validator(mode="after")
    def validate_timeout_relationship(self) -> "ShellMcpSettings":
        if self.default_timeout_seconds > self.max_timeout_seconds:
            raise ValueError(
                "default_timeout_seconds cannot exceed max_timeout_seconds"
            )
        if self.max_timeout_seconds >= self.operation_timeout_seconds:
            raise ValueError(
                "max_timeout_seconds must be below operation_timeout_seconds"
            )
        return self

    def resolved_workspace_root(self) -> Path:
        try:
            root = self.workspace_root.expanduser().resolve(strict=True)
        except OSError as error:
            raise ValueError("workspace_root must exist") from error
        if not root.is_dir():
            raise ValueError("workspace_root must be a directory")
        return root

    def server_arguments(self, workspace_root: Path | None = None) -> list[str]:
        root = workspace_root or self.resolved_workspace_root()
        return [
            *self.server_args,
            "--workspace-root",
            str(root),
            "--max-timeout-seconds",
            str(self.max_timeout_seconds),
            "--max-stdout-bytes",
            str(self.max_stdout_bytes),
            "--max-stderr-bytes",
            str(self.max_stderr_bytes),
        ]
