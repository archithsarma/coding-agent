"""Protocol implemented by internal tool adapters."""

from typing import Protocol

from coding_agent.domain import ToolRequest, ToolResult
from coding_agent.tools.descriptor import ToolDescriptor


class InternalTool(Protocol):
    descriptor: ToolDescriptor

    async def execute(self, request: ToolRequest) -> ToolResult:
        """Execute a validated request and return the normalized result."""
