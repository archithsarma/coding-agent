"""Deterministic composition of execution policies."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from coding_agent.domain import ToolRequest
from coding_agent.policies.protocol import ExecutionPolicy

if TYPE_CHECKING:
    from coding_agent.tools.descriptor import ToolDescriptor


class PolicyChain:
    def __init__(self, policies: Iterable[ExecutionPolicy] = ()) -> None:
        self._policies = tuple(policies)

    async def validate(self, descriptor: ToolDescriptor, request: ToolRequest) -> None:
        for policy in self._policies:
            await policy.validate(descriptor, request)
