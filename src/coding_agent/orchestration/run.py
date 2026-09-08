"""Deterministic Run trajectory nodes."""

from __future__ import annotations

from typing import Literal, cast
from uuid import uuid4

from langgraph.runtime import Runtime
from pydantic import JsonValue

from coding_agent.domain import ToolRequest, ToolResult, VerificationKind
from coding_agent.memory import SessionEvent
from coding_agent.orchestration.context import OrchestrationContext
from coding_agent.orchestration.state import OrchestrationState, RunState
from coding_agent.orchestration.verification import (
    VerificationCommand,
    VerificationPayloadError,
    build_verification_result,
    plan_verification,
    user_facing_summary,
)

RUN_PLAN = "run_plan"
RUN_EXECUTE = "run_execute"
RUN_INTERPRET = "run_interpret"
RUN_COMPLETE = "run_complete"
RUN_FAILED = "run_failed"


def _run(state: OrchestrationState) -> RunState:
    return state.get("run", {})


def _failure(
    state: OrchestrationState,
    *,
    code: str,
    message: str,
    node: str,
) -> dict[str, object]:
    return {
        "failure": {
            "code": code,
            "message": message,
            "node": node,
            "retryable": False,
        },
        "run": {**_run(state), "tool_result": None},
        "current_node": node,
    }


def _command(state: OrchestrationState) -> VerificationCommand:
    raw = _run(state).get("command")
    if not isinstance(raw, dict):
        raise RuntimeError("Run state is missing a planned verification command")
    kind = raw.get("kind")
    argv = raw.get("argv")
    cwd = raw.get("cwd")
    if (
        not isinstance(kind, str)
        or not isinstance(argv, list)
        or not isinstance(cwd, str)
    ):
        raise RuntimeError("Run state contains an invalid verification command")
    try:
        verification_kind = VerificationKind(kind)
    except ValueError as error:
        raise RuntimeError("Run state contains an unknown verification kind") from error
    if not all(isinstance(item, str) for item in argv):
        raise RuntimeError("Run state contains an invalid verification argv")
    return VerificationCommand(verification_kind, tuple(cast(list[str], argv)), cwd)


def plan(state: OrchestrationState) -> dict[str, object]:
    command = plan_verification(state["user_request"])
    if command is None:
        return _failure(
            state,
            code="unsupported_run_request",
            message="the requested verification action is not supported",
            node=RUN_PLAN,
        )
    return {
        "run": {
            **_run(state),
            "command": {
                "kind": command.kind.value,
                "argv": list(command.argv),
                "cwd": command.cwd,
            },
        },
        "current_node": RUN_PLAN,
    }


def plan_next(state: OrchestrationState) -> Literal["run_failed", "run_execute"]:
    return cast(
        Literal["run_failed", "run_execute"],
        RUN_FAILED if state.get("failure") else RUN_EXECUTE,
    )


async def execute(
    state: OrchestrationState, runtime: Runtime[OrchestrationContext]
) -> dict[str, object]:
    context = runtime.context
    if context is None:
        raise RuntimeError("Run requires tool runtime context")
    counters = state["counters"]
    budget = state["execution_budget"]
    if counters["tool_calls"] >= budget["max_tool_calls"]:
        return _failure(
            state,
            code="tool_budget_exhausted",
            message="shell tool-call budget exhausted",
            node=RUN_EXECUTE,
        )
    command = _command(state)
    updated_counters = {**counters, "tool_calls": counters["tool_calls"] + 1}
    result = await context.tools.invoke(
        ToolRequest(
            call_id=f"run-shell-{uuid4().hex}",
            capability="shell.execute",
            arguments={
                "argv": list(command.argv),
                "cwd": command.cwd,
            },
        )
    )
    if not result.success:
        error = result.error
        return _failure(
            state,
            code=error.code if error else "tool_failure",
            message=error.message if error else "shell tool failed",
            node=RUN_EXECUTE,
        ) | {"counters": updated_counters}
    return {
        "counters": updated_counters,
        "run": {**_run(state), "tool_result": result.data},
        "current_node": RUN_EXECUTE,
    }


def execute_next(state: OrchestrationState) -> Literal["run_failed", "run_interpret"]:
    return cast(
        Literal["run_failed", "run_interpret"],
        RUN_FAILED if state.get("failure") else RUN_INTERPRET,
    )


def interpret_next(
    state: OrchestrationState,
) -> Literal["run_failed", "run_complete"]:
    return cast(
        Literal["run_failed", "run_complete"],
        RUN_FAILED if state.get("failure") else RUN_COMPLETE,
    )


def interpret(state: OrchestrationState) -> dict[str, object]:
    result_data = _run(state).get("tool_result")
    result = _tool_result_from_state(result_data)
    try:
        verification = build_verification_result(_command(state), result)
    except VerificationPayloadError as error:
        return _failure(
            state,
            code="malformed_tool_result",
            message=str(error),
            node=RUN_INTERPRET,
        )
    return {
        "run": {
            **_run(state),
            "verification": verification.model_dump(mode="json"),
            "tool_result": None,
        },
        "current_node": RUN_INTERPRET,
    }


def _tool_result_from_state(data: object) -> ToolResult:
    return ToolResult(
        call_id="run-result",
        success=True,
        data=cast(JsonValue, data),
    )


def complete(
    state: OrchestrationState, runtime: Runtime[OrchestrationContext]
) -> dict[str, object]:
    from coding_agent.domain import VerificationResult

    raw = _run(state).get("verification")
    if not isinstance(raw, dict):
        raise RuntimeError("Run state is missing a verification result")
    verification = VerificationResult.model_validate(raw)
    _record_session_event(
        runtime,
        state,
        outcome="succeeded" if verification.passed else "failed",
        summary=f"ran {verification.kind.value}",
        verification_status="passed" if verification.passed else "failed",
    )
    return {
        "run": {**_run(state), "answer": user_facing_summary(verification)},
        "current_node": RUN_COMPLETE,
    }


def failed(
    state: OrchestrationState, runtime: Runtime[OrchestrationContext]
) -> dict[str, object]:
    failure = state.get("failure")
    message = failure.get("message") if isinstance(failure, dict) else "run failed"
    failure_code = (
        failure.get("code", "unknown") if isinstance(failure, dict) else "unknown"
    )
    _record_session_event(
        runtime,
        state,
        outcome="failed",
        summary=f"run failed: {failure_code}",
    )
    return {
        "run": {**_run(state), "answer": f"Run failed: {message}", "tool_result": None},
        "current_node": RUN_FAILED,
    }


def _record_session_event(
    runtime: Runtime[OrchestrationContext],
    state: OrchestrationState,
    *,
    outcome: str,
    summary: str,
    verification_status: str | None = None,
) -> None:
    if runtime.context is None:
        return
    command = _run(state).get("command")
    argv = command.get("argv", []) if isinstance(command, dict) else []
    command_text = (
        " ".join(item for item in argv if isinstance(item, str))
        if isinstance(argv, list)
        else ""
    )
    runtime.context.session_memory.record(
        SessionEvent(
            session_id=runtime.context.session_id,
            trajectory="run",
            summary=summary[:500],
            commands=(command_text,),
            outcome=outcome,
            verification_status=verification_status,
        )
    )
