import pytest

from coding_agent.domain import (
    ExecutionBudget,
    PolicyViolationError,
    ToolErrorInfo,
    ToolRequest,
    ToolResult,
    VerificationKind,
)
from coding_agent.model import TextGenerationResult
from coding_agent.orchestration.context import OrchestrationContext
from coding_agent.orchestration.graph import build_graph
from coding_agent.orchestration.verification import plan_verification


class NoModel:
    async def generate_text(self, **kwargs) -> TextGenerationResult:
        raise AssertionError("Run must not call the model")

    async def generate_structured(self, **kwargs):
        raise AssertionError("Run must not call the model")


class FakeShellRuntime:
    def __init__(
        self, result: ToolResult | None = None, error: Exception | None = None
    ):
        self.result = result
        self.error = error
        self.calls: list[ToolRequest] = []

    async def invoke(self, request: ToolRequest) -> ToolResult:
        self.calls.append(request)
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result.model_copy(update={"call_id": request.call_id})


def budget(*, max_tool_calls: int = 1) -> ExecutionBudget:
    return ExecutionBudget(
        max_llm_calls=2,
        max_tool_calls=max_tool_calls,
        max_repair_attempts=0,
        max_shell_execution_seconds=0,
    )


def shell_payload(
    *,
    argv: list[str] | None = None,
    exit_code: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> dict:
    return {
        "argv": argv or ["pytest"],
        "cwd": ".",
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "duration_ms": 12,
        "timed_out": False,
        "stdout_truncated": False,
        "stderr_truncated": False,
    }


async def run(request: str, runtime, *, execution_budget=None):
    return await build_graph(execution_budget or budget()).ainvoke(
        {"task_id": "task-1", "user_request": request},
        context=OrchestrationContext(model=NoModel(), tools=runtime),
    )


@pytest.mark.parametrize(
    ("intent", "kind", "argv"),
    [
        ("run tests", VerificationKind.TEST, ("pytest",)),
        ("run pytest", VerificationKind.TEST, ("pytest",)),
        ("check lint", VerificationKind.LINT, ("ruff", "check", ".")),
        ("run lint", VerificationKind.LINT, ("ruff", "check", ".")),
        ("run ruff", VerificationKind.LINT, ("ruff", "check", ".")),
        ("run mypy", VerificationKind.TYPE_CHECK, ("mypy", "src")),
        ("type check", VerificationKind.TYPE_CHECK, ("mypy", "src")),
    ],
)
def test_verification_planning_is_deterministic(intent, kind, argv):
    command = plan_verification(f"  {intent.upper()}  ")
    assert command is not None
    assert command.kind == kind
    assert command.argv == argv


@pytest.mark.anyio
async def test_passing_command_completes_without_model_call():
    runtime = FakeShellRuntime(
        ToolResult(call_id="result", success=True, data=shell_payload())
    )
    result = await run("run tests", runtime)

    assert result["current_node"] == "run_complete"
    assert result["run"]["verification"]["passed"] is True
    assert result["run"]["verification"]["kind"] == "test"
    assert result["run"]["answer"] == "Tests passed (exit code 0)."
    assert result["counters"] == {
        "llm_calls": 0,
        "tool_calls": 1,
        "repair_attempts": 0,
    }
    assert len(runtime.calls) == 1
    assert runtime.calls[0].arguments == {"argv": ["pytest"], "cwd": "."}


@pytest.mark.anyio
async def test_failing_command_is_completed_verification_not_tool_failure():
    runtime = FakeShellRuntime(
        ToolResult(
            call_id="result",
            success=True,
            data=shell_payload(
                exit_code=1,
                stdout="1 failed, 2 passed",
                stderr="FAILED tests/test_app.py::test_bad",
            ),
        )
    )
    result = await run("run pytest", runtime)

    assert result["current_node"] == "run_complete"
    assert result["failure"] is None
    assert result["run"]["verification"]["passed"] is False
    assert result["run"]["verification"]["exit_code"] == 1
    assert "FAILED tests/test_app.py::test_bad" in result["run"]["answer"]
    assert result["counters"]["llm_calls"] == 0


@pytest.mark.parametrize("code", ["mcp_tool_error", "shell_timeout"])
@pytest.mark.anyio
async def test_tool_failures_become_structured_run_failures(code):
    runtime = FakeShellRuntime(
        ToolResult(
            call_id="result",
            success=False,
            error=ToolErrorInfo(code=code, message="shell failed"),
        )
    )
    result = await run("run tests", runtime)

    assert result["current_node"] == "run_failed"
    assert result["failure"]["code"] == code
    assert "verification" not in result["run"]
    assert result["counters"]["tool_calls"] == 1
    assert result["counters"]["llm_calls"] == 0


@pytest.mark.anyio
async def test_unsupported_or_injected_run_request_fails_without_tool_call():
    runtime = FakeShellRuntime(
        ToolResult(call_id="result", success=True, data=shell_payload())
    )
    result = await run("run tests && echo hacked", runtime)

    assert result["failure"]["code"] == "unsupported_run_request"
    assert result["counters"]["tool_calls"] == 0
    assert runtime.calls == []


@pytest.mark.anyio
async def test_zero_budget_prevents_invocation_and_one_budget_allows_exactly_one():
    zero_runtime = FakeShellRuntime(
        ToolResult(call_id="result", success=True, data=shell_payload())
    )
    zero = await run(
        "run tests", zero_runtime, execution_budget=budget(max_tool_calls=0)
    )
    assert zero["failure"]["code"] == "tool_budget_exhausted"
    assert zero_runtime.calls == []

    one_runtime = FakeShellRuntime(
        ToolResult(call_id="result", success=True, data=shell_payload())
    )
    one = await run("run tests", one_runtime, execution_budget=budget(max_tool_calls=1))
    assert one["current_node"] == "run_complete"
    assert len(one_runtime.calls) == 1


@pytest.mark.anyio
async def test_policy_and_programming_errors_propagate():
    policy_runtime = FakeShellRuntime(error=PolicyViolationError("not allowed"))
    with pytest.raises(PolicyViolationError):
        await run("run tests", policy_runtime)

    programming_runtime = FakeShellRuntime(error=RuntimeError("bug"))
    with pytest.raises(RuntimeError, match="bug"):
        await run("run tests", programming_runtime)


@pytest.mark.anyio
async def test_unfamiliar_output_still_uses_exit_code_and_bounded_summary():
    runtime = FakeShellRuntime(
        ToolResult(
            call_id="result",
            success=True,
            data=shell_payload(
                argv=["ruff", "check", "."],
                exit_code=2,
                stdout="completely unfamiliar output",
            ),
        )
    )
    result = await run("check lint", runtime)

    assert result["run"]["verification"]["passed"] is False
    assert result["run"]["verification"]["exit_code"] == 2
    assert "completely unfamiliar output" in result["run"]["answer"]
