"""Deterministic verification command planning and shell-result interpretation."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import JsonValue

from coding_agent.domain import ToolResult, VerificationKind, VerificationResult
from coding_agent.orchestration.routing import normalize_request

MAX_DIAGNOSTIC_SUMMARY_CHARS = 1_000


@dataclass(frozen=True)
class VerificationCommand:
    kind: VerificationKind
    argv: tuple[str, ...]
    cwd: str = "."


class VerificationPayloadError(ValueError):
    """A successful shell result did not match the normalized contract."""


def plan_verification(request: str) -> VerificationCommand | None:
    normalized = normalize_request(request)
    commands = {
        "run tests": VerificationCommand(VerificationKind.TEST, ("pytest",)),
        "run pytest": VerificationCommand(VerificationKind.TEST, ("pytest",)),
        "check lint": VerificationCommand(
            VerificationKind.LINT, ("ruff", "check", ".")
        ),
        "run lint": VerificationCommand(VerificationKind.LINT, ("ruff", "check", ".")),
        "run ruff": VerificationCommand(VerificationKind.LINT, ("ruff", "check", ".")),
        "run mypy": VerificationCommand(VerificationKind.TYPE_CHECK, ("mypy", "src")),
        "type check": VerificationCommand(VerificationKind.TYPE_CHECK, ("mypy", "src")),
    }
    return commands.get(normalized)


def build_verification_result(
    command: VerificationCommand, result: ToolResult
) -> VerificationResult:
    if not result.success:
        raise VerificationPayloadError("cannot interpret an unsuccessful tool result")
    data = result.data
    if not isinstance(data, dict):
        raise VerificationPayloadError("shell result lacked structured data")
    expected = {
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
    if set(data) != expected:
        raise VerificationPayloadError("shell result had an invalid schema")
    argv = data["argv"]
    cwd = data["cwd"]
    exit_code = data["exit_code"]
    stdout = data["stdout"]
    stderr = data["stderr"]
    duration_ms = data["duration_ms"]
    timed_out = data["timed_out"]
    stdout_truncated = data["stdout_truncated"]
    stderr_truncated = data["stderr_truncated"]
    if (
        argv != list(command.argv)
        or cwd != command.cwd
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
        raise VerificationPayloadError("shell result contained invalid fields")
    summary = _diagnostic_summary(stdout, stderr)
    metadata: dict[str, JsonValue] = {
        "timed_out": timed_out,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
    }
    if summary:
        metadata["diagnostic_summary"] = summary
    return VerificationResult(
        kind=command.kind,
        passed=exit_code == 0,
        command=" ".join(command.argv),
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        duration_seconds=duration_ms / 1000,
        metadata=metadata,
    )


def user_facing_summary(result: VerificationResult) -> str:
    label = {
        VerificationKind.TEST: "Tests",
        VerificationKind.LINT: "Ruff",
        VerificationKind.TYPE_CHECK: "Mypy",
    }[result.kind]
    status = "passed" if result.passed else "failed"
    answer = f"{label} {status} (exit code {result.exit_code})."
    summary = result.metadata.get("diagnostic_summary")
    if not result.passed and isinstance(summary, str) and summary:
        answer += f"\nRelevant output:\n{summary}"
    return answer


def _diagnostic_summary(stdout: str, stderr: str) -> str:
    lines = [line.strip() for line in (*stderr.splitlines(), *stdout.splitlines())]
    lines = [line for line in lines if line]
    if not lines:
        return ""
    summary = "\n".join(lines[-8:])
    if len(summary) > MAX_DIAGNOSTIC_SUMMARY_CHARS:
        summary = summary[-MAX_DIAGNOSTIC_SUMMARY_CHARS:]
    return summary
