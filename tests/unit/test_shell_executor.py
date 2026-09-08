import asyncio
import sys
from pathlib import Path

from coding_agent.mcp_servers.shell import ShellCommandExecutor
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
