"""Async runtime for policy-checked internal tool invocation."""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING

from coding_agent.domain import (
    InvalidRequestError,
    ToolErrorInfo,
    ToolExecutionError,
    ToolRequest,
    ToolResult,
)
from coding_agent.policies import PolicyChain
from coding_agent.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from coding_agent.observability import TraceState


class ToolRuntime:
    def __init__(
        self, registry: ToolRegistry, *, policies: PolicyChain | None = None
    ) -> None:
        self._registry = registry
        self._policies = policies or PolicyChain()
        self._trace: TraceState | None = None

    def bind_trace(self, trace: TraceState) -> None:
        self._trace = trace

    async def invoke(self, request: ToolRequest) -> ToolResult:
        try:
            tool = self._registry.resolve_by_capability(request.capability)
        except InvalidRequestError as error:
            return ToolResult(
                call_id=request.call_id,
                success=False,
                error=ToolErrorInfo(code="unknown_capability", message=str(error)),
            )

        await self._policies.validate(tool.descriptor, request)
        started = perf_counter()
        if self._trace is not None:
            self._trace.emit(
                "tool.started",
                metadata={
                    "capability": request.capability,
                    "path": request.arguments.get("path", request.arguments.get("cwd")),
                },
            )
        try:
            result = await tool.execute(request)
        except ToolExecutionError as error:
            result = ToolResult(
                call_id=request.call_id,
                success=False,
                error=ToolErrorInfo(
                    code=ToolExecutionError.code,
                    message=str(error),
                ),
            )

        if result.call_id != request.call_id:
            raise RuntimeError("tool result call_id does not match request")
        duration = perf_counter() - started
        if self._trace is not None:
            self._trace.emit(
                "tool.completed" if result.success else "tool.failed",
                duration_ms=duration * 1000,
                outcome="succeeded" if result.success else "failed",
                metadata={"capability": request.capability},
            )
        return result.model_copy(update={"duration_seconds": duration})
