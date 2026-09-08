"""Shared validation rules for the controlled shell execution boundary."""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass

DEFAULT_MAX_ARGUMENTS = 64
DEFAULT_MAX_ARGUMENT_LENGTH = 4096
DEFAULT_MAX_TIMEOUT_SECONDS = 120.0
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_STDOUT_BYTES = 1_048_576
DEFAULT_MAX_STDERR_BYTES = 1_048_576

ALLOWED_EXECUTABLES = frozenset({"pytest", "ruff", "mypy", "git"})


class ShellValidationError(ValueError):
    """A structured shell invocation is not permitted."""


@dataclass(frozen=True)
class ShellInvocation:
    argv: tuple[str, ...]
    cwd: str
    timeout_seconds: float


def validate_invocation(
    arguments: Mapping[str, object],
    *,
    default_timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_timeout_seconds: float = DEFAULT_MAX_TIMEOUT_SECONDS,
    max_arguments: int = DEFAULT_MAX_ARGUMENTS,
    max_argument_length: int = DEFAULT_MAX_ARGUMENT_LENGTH,
    allowed_executables: Collection[str] = ALLOWED_EXECUTABLES,
) -> ShellInvocation:
    raw_argv = arguments.get("argv")
    if not isinstance(raw_argv, list) or not raw_argv:
        raise ShellValidationError("argv must be a non-empty list")
    if len(raw_argv) > max_arguments:
        raise ShellValidationError("argv contains too many arguments")
    if any(not isinstance(item, str) for item in raw_argv):
        raise ShellValidationError("argv items must be strings")
    argv = tuple(raw_argv)
    for item in argv:
        if not item.strip() or "\x00" in item:
            raise ShellValidationError(
                "argv items must be non-blank and contain no null bytes"
            )
        if len(item) > max_argument_length:
            raise ShellValidationError("argv item exceeds the maximum length")
    _validate_command_shape(argv, allowed_executables)

    raw_cwd = arguments.get("cwd", ".")
    if not isinstance(raw_cwd, str) or not raw_cwd.strip() or "\x00" in raw_cwd:
        raise ShellValidationError("cwd must be a non-blank string without null bytes")

    raw_timeout = arguments.get("timeout_seconds", default_timeout_seconds)
    if isinstance(raw_timeout, bool) or not isinstance(raw_timeout, (int, float)):
        raise ShellValidationError("timeout_seconds must be a number")
    timeout_seconds = float(raw_timeout)
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ShellValidationError("timeout_seconds must be positive and finite")
    if timeout_seconds > max_timeout_seconds:
        raise ShellValidationError("timeout_seconds exceeds the configured maximum")

    return ShellInvocation(argv=argv, cwd=raw_cwd, timeout_seconds=timeout_seconds)


def _validate_command_shape(
    argv: tuple[str, ...], allowed_executables: Collection[str]
) -> None:
    executable = argv[0]
    if executable not in allowed_executables:
        raise ShellValidationError(f"executable '{executable}' is not allowed")
    if executable == "git" and argv[1:] not in {
        ("status",),
        ("diff",),
        ("diff", "--check"),
    }:
        raise ShellValidationError(
            "only read-only git status and diff commands are allowed"
        )
    if executable == "ruff":
        if len(argv) < 2 or argv[1] not in {"check", "format"}:
            raise ShellValidationError("ruff is limited to check and format --check")
        if argv[1] == "format" and "--check" not in argv[2:]:
            raise ShellValidationError("ruff format requires --check")
        if any(item in {"--fix", "--unsafe-fixes", "--diff"} for item in argv[2:]):
            raise ShellValidationError("ruff fixes and diffs are not allowed")
