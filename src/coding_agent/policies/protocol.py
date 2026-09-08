"""Protocol for async execution policies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from coding_agent.domain import ToolRequest

if TYPE_CHECKING:
    from coding_agent.tools.descriptor import ToolDescriptor


class ExecutionPolicy(Protocol):
    async def validate(self, descriptor: ToolDescriptor, request: ToolRequest) -> None:
        """Allow the request or raise PolicyViolationError."""
