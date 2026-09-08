"""Instance-scoped registry for internal tools."""

from coding_agent.domain import InvalidRequestError, OperationConflictError
from coding_agent.tools.descriptor import ToolDescriptor
from coding_agent.tools.protocol import InternalTool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, InternalTool] = {}
        self._capabilities: dict[str, InternalTool] = {}

    def register(self, tool: InternalTool) -> None:
        descriptor = tool.descriptor
        if descriptor.tool_name in self._tools:
            raise OperationConflictError(
                f"Tool '{descriptor.tool_name}' is already registered",
                context={"tool_name": descriptor.tool_name},
            )
        if descriptor.capability in self._capabilities:
            raise OperationConflictError(
                f"Capability '{descriptor.capability}' already has an implementation",
                context={"capability": descriptor.capability},
            )
        self._tools[descriptor.tool_name] = tool
        self._capabilities[descriptor.capability] = tool

    def resolve_by_name(self, tool_name: str) -> InternalTool:
        try:
            return self._tools[tool_name]
        except KeyError as error:
            raise InvalidRequestError(
                f"Unknown tool '{tool_name}'", context={"tool_name": tool_name}
            ) from error

    def resolve_by_capability(self, capability: str) -> InternalTool:
        try:
            return self._capabilities[capability]
        except KeyError as error:
            raise InvalidRequestError(
                f"No tool provides capability '{capability}'",
                context={"capability": capability},
            ) from error

    def list_descriptors(self) -> tuple[ToolDescriptor, ...]:
        return tuple(
            tool.descriptor
            for tool in sorted(
                self._tools.values(), key=lambda item: item.descriptor.tool_name
            )
        )
