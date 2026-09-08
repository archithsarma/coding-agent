import asyncio
import os
import sys
from pathlib import Path

import pytest

from coding_agent.mcp_servers.shell import (
    ShellCommandExecutor,
    ShellProcessError,
    _resolve_executable,
)
from coding_agent.shell import ShellInvocation


def run(coroutine: object) -> object:
    return asyncio.run(coroutine)  # type: ignore[arg-type]


def test_executor_captures_success_and_failure_without_shell_parsing(
    tmp_path: Path,
) -> None:
    executor = ShellCommandExecutor()
    result = run(
        executor.execute(
            ShellInvocation(
                argv=(
                    sys.executable,
                    "-c",
                    "import sys; print(sys.argv[1]); "
                    "sys.stderr.write('err'); sys.exit(3)",
                    "; echo SHOULD_NOT_RUN",
                ),
                cwd=".",
                timeout_seconds=2,
            ),
            cwd=tmp_path,
        )
    )

    assert result.exit_code == 3
    assert result.stdout == "; echo SHOULD_NOT_RUN\n"
    assert result.stderr == "err"
    assert result.timed_out is False
    assert result.duration_ms >= 0


def test_executor_enforces_timeout_and_reaps_process(tmp_path: Path) -> None:
    executor = ShellCommandExecutor(termination_grace_seconds=0.05)
    result = run(
        executor.execute(
            ShellInvocation(
                argv=(sys.executable, "-c", "import time; time.sleep(10)"),
                cwd=".",
                timeout_seconds=0.05,
            ),
            cwd=tmp_path,
        )
    )

    assert result.timed_out is True
    assert result.exit_code != 0


def test_executor_bounds_each_output_stream(tmp_path: Path) -> None:
    executor = ShellCommandExecutor(max_stdout_bytes=10, max_stderr_bytes=8)
    result = run(
        executor.execute(
            ShellInvocation(
                argv=(
                    sys.executable,
                    "-c",
                    "import sys; print('o' * 100, end=''); "
                    "print('e' * 100, file=sys.stderr, end='')",
                ),
                cwd=".",
                timeout_seconds=2,
            ),
            cwd=tmp_path,
        )
    )

    assert len(result.stdout.encode()) == 10
    assert len(result.stderr.encode()) == 8
    assert result.stdout_truncated is True
    assert result.stderr_truncated is True


def test_executor_keeps_multibyte_output_within_byte_limit(tmp_path: Path) -> None:
    executor = ShellCommandExecutor(max_stdout_bytes=4)
    result = run(
        executor.execute(
            ShellInvocation(
                argv=(sys.executable, "-c", "print('ééé', end='')"),
                cwd=".",
                timeout_seconds=2,
            ),
            cwd=tmp_path,
        )
    )

    assert len(result.stdout.encode()) <= 4
    assert result.stdout_truncated is True


def test_executor_cancellation_reaps_the_process(tmp_path: Path) -> None:
    async def scenario() -> None:
        executor = ShellCommandExecutor(termination_grace_seconds=0.05)
        task = asyncio.create_task(
            executor.execute(
                ShellInvocation(
                    argv=(sys.executable, "-c", "import time; time.sleep(10)"),
                    cwd=".",
                    timeout_seconds=30,
                ),
                cwd=tmp_path,
            )
        )
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=2)
        except asyncio.CancelledError:
            pass

    run(scenario())


def test_execution_environment_excludes_unapproved_variables(monkeypatch) -> None:
    monkeypatch.setenv("SECRET_SHOULD_NOT_CROSS_BOUNDARY", "hidden")
    from coding_agent.mcp_servers.shell import _execution_environment

    assert "SECRET_SHOULD_NOT_CROSS_BOUNDARY" not in _execution_environment()


def test_approved_logical_executable_resolves_from_server_interpreter_bin(
    tmp_path: Path, monkeypatch
) -> None:
    interpreter = tmp_path / "python"
    interpreter.touch()
    executable = tmp_path / "pytest"
    executable.touch()
    executable.chmod(0o700)
    monkeypatch.setattr(sys, "executable", str(interpreter))
    monkeypatch.setenv("PATH", os.defpath)

    assert _resolve_executable("pytest") == str(executable)


def test_executor_runs_approved_logical_command_without_venv_on_child_path(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("PATH", os.defpath)

    result = run(
        ShellCommandExecutor().execute(
            ShellInvocation(argv=("pytest", "--version"), cwd=".", timeout_seconds=2),
            cwd=tmp_path,
        )
    )

    assert result.exit_code == 0
    assert result.stdout.startswith("pytest ")


def test_missing_approved_executable_has_explicit_resolution_failure(
    tmp_path: Path, monkeypatch
) -> None:
    interpreter = tmp_path / "python"
    interpreter.touch()
    monkeypatch.setattr(sys, "executable", str(interpreter))
    monkeypatch.setenv("PATH", os.defpath)

    with pytest.raises(ShellProcessError, match="approved executable 'pytest'"):
        _resolve_executable("pytest")


def test_absolute_and_arbitrary_executables_are_not_resolved_by_allowlist() -> None:
    assert _resolve_executable("/trusted/tool") == "/trusted/tool"
    assert _resolve_executable("arbitrary-tool") == "arbitrary-tool"
