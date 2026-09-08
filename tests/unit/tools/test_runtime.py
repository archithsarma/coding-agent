import asyncio
from collections.abc import Awaitable
from typing import TypeVar

import pytest

from coding_agent.domain import (
    InvalidRequestError,
    OperationConflictError,
    PolicyViolationError,
    ToolErrorInfo,
    ToolExecutionError,
    ToolRequest,
    ToolResult,
)
from coding_agent.tools import (
    PolicyChain,
    ToolDescriptor,
    ToolRegistry,
    ToolRuntime,
)

ResultT = TypeVar("ResultT")


class EchoTool:
    descriptor = ToolDescriptor(
        tool_name="echo",
        capability="testing.echo",
        description="Returns the supplied payload.",
        mutating=False,
    )

    def __init__(self) -> None:
        self.executions = 0

    async def execute(self, request: ToolRequest) -> ToolResult:
        self.executions += 1
        return ToolResult(call_id=request.call_id, success=True, data=request.arguments)


class SecondEchoTool(EchoTool):
    descriptor = ToolDescriptor(
        tool_name="second-echo",
        capability="testing.echo",
        description="A second echo implementation.",
        mutating=False,
    )


class FailingTool(EchoTool):
    descriptor = ToolDescriptor(
        tool_name="failing",
        capability="testing.failing",
        description="Raises an expected operational failure.",
        mutating=False,
    )

    async def execute(self, request: ToolRequest) -> ToolResult:
        raise ToolExecutionError("temporary tool failure")


class BuggyTool(EchoTool):
    descriptor = ToolDescriptor(
        tool_name="buggy",
        capability="testing.buggy",
        description="Raises an unexpected programming error.",
        mutating=False,
    )

    async def execute(self, request: ToolRequest) -> ToolResult:
        raise RuntimeError("programming defect")


class RecordingPolicy:
    def __init__(self, name: str, calls: list[str]) -> None:
        self.name = name
        self.calls = calls

    async def validate(self, descriptor: ToolDescriptor, request: ToolRequest) -> None:
        self.calls.append(self.name)


class RejectingPolicy:
    async def validate(self, descriptor: ToolDescriptor, request: ToolRequest) -> None:
        raise PolicyViolationError("tool is not allowed")


def run(coroutine: Awaitable[ResultT]) -> ResultT:
    return asyncio.run(coroutine)


def test_registry_rejects_duplicates_and_lists_descriptors_deterministically() -> None:
    registry = ToolRegistry()
    echo = EchoTool()
    registry.register(echo)
    registry.register(FailingTool())

    with pytest.raises(OperationConflictError):
        registry.register(EchoTool())

    assert registry.resolve_by_name("echo") is echo
    assert registry.resolve_by_capability("testing.echo") is echo
    assert [descriptor.tool_name for descriptor in registry.list_descriptors()] == [
        "echo",
        "failing",
    ]

    with pytest.raises(InvalidRequestError):
        registry.resolve_by_name("missing")


def test_registry_rejects_two_implementations_for_one_capability() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())

    with pytest.raises(OperationConflictError):
        registry.register(SecondEchoTool())


def test_runtime_invokes_by_capability_and_records_duration() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())
    runtime = ToolRuntime(registry)

    result = run(
        runtime.invoke(
            ToolRequest(
                call_id="call-1", capability="testing.echo", arguments={"value": 3}
            )
        )
    )

    assert isinstance(result, ToolResult)
    assert result.success is True
    assert result.data == {"value": 3}
    assert result.duration_seconds is not None
    assert result.duration_seconds >= 0


def test_unknown_capability_returns_structured_failure() -> None:
    result = run(
        ToolRuntime(ToolRegistry()).invoke(
            ToolRequest(call_id="call-1", capability="testing.missing")
        )
    )

    assert isinstance(result, ToolResult)
    assert result.success is False
    assert result.error == ToolErrorInfo(
        code="unknown_capability",
        message="No tool provides capability 'testing.missing'",
    )


def test_policy_chain_runs_in_order_and_rejection_prevents_execution() -> None:
    calls: list[str] = []
    registry = ToolRegistry()
    echo = EchoTool()
    registry.register(echo)
    runtime = ToolRuntime(
        registry,
        policies=PolicyChain(
            [RecordingPolicy("first", calls), RecordingPolicy("second", calls)]
        ),
    )

    run(runtime.invoke(ToolRequest(call_id="call-1", capability="testing.echo")))
    assert calls == ["first", "second"]

    rejecting_runtime = ToolRuntime(registry, policies=PolicyChain([RejectingPolicy()]))
    with pytest.raises(PolicyViolationError):
        run(
            rejecting_runtime.invoke(
                ToolRequest(call_id="call-2", capability="testing.echo")
            )
        )
    assert echo.executions == 1


def test_expected_tool_failure_is_normalized_but_unexpected_error_propagates() -> None:
    registry = ToolRegistry()
    registry.register(FailingTool())
    registry.register(BuggyTool())
    runtime = ToolRuntime(registry)

    expected = run(
        runtime.invoke(ToolRequest(call_id="call-1", capability="testing.failing"))
    )
    assert isinstance(expected, ToolResult)
    assert expected.success is False
    assert expected.error is not None
    assert expected.error.code == "tool_execution_failed"

    with pytest.raises(RuntimeError, match="programming defect"):
        run(runtime.invoke(ToolRequest(call_id="call-2", capability="testing.buggy")))
