import asyncio

import pytest

from coding_agent.domain import PolicyViolationError, ToolRequest
from coding_agent.policies import PolicyChain
from coding_agent.tools import ToolDescriptor


class RecordingPolicy:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    async def validate(self, descriptor: ToolDescriptor, request: ToolRequest) -> None:
        self.calls.append(descriptor.tool_name)


class RejectingPolicy:
    async def validate(self, descriptor: ToolDescriptor, request: ToolRequest) -> None:
        raise PolicyViolationError("rejected by policy")


def test_policy_chain_preserves_order() -> None:
    calls: list[str] = []
    chain = PolicyChain([RecordingPolicy(calls), RecordingPolicy(calls)])
    descriptor = ToolDescriptor(
        tool_name="echo",
        capability="testing.echo",
        description="Echo",
        mutating=False,
    )

    asyncio.run(
        chain.validate(descriptor, ToolRequest(call_id="1", capability="testing.echo"))
    )

    assert calls == ["echo", "echo"]


def test_policy_chain_stops_at_rejection() -> None:
    calls: list[str] = []
    chain = PolicyChain([RejectingPolicy(), RecordingPolicy(calls)])
    descriptor = ToolDescriptor(
        tool_name="echo",
        capability="testing.echo",
        description="Echo",
        mutating=False,
    )

    with pytest.raises(PolicyViolationError):
        asyncio.run(
            chain.validate(
                descriptor, ToolRequest(call_id="1", capability="testing.echo")
            )
        )

    assert calls == []
