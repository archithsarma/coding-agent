"""Async runtime for policy-checked internal tool invocation."""

from time import perf_counter

from coding_agent.domain import (
    InvalidRequestError,
    ToolErrorInfo,
    ToolExecutionError,
    ToolRequest,
    ToolResult,
)
from coding_agent.policies import PolicyChain
from coding_agent.tools.registry import ToolRegistry


class ToolRuntime:
    def __init__(
        self, registry: ToolRegistry, *, policies: PolicyChain | None = None
    ) -> None:
        self._registry = registry
        self._policies = policies or PolicyChain()

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
        return result.model_copy(update={"duration_seconds": perf_counter() - started})
