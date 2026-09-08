"""Serializable state and records owned by the workflow orchestrator."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, JsonValue
from typing_extensions import TypedDict

from coding_agent.domain import Trajectory
from coding_agent.domain.models import DomainModel


class RoutingDecision(DomainModel):
    """The result of one routing strategy."""

    trajectory: Trajectory | None = None
    source: Literal["deterministic", "model"]
    reason: str | None = None


class OrchestrationFailure(DomainModel):
    """A structured expected workflow failure."""

    code: str
    message: str
    node: str
    retryable: bool = False
    details: dict[str, JsonValue] = Field(default_factory=dict)


class ExecutionCounters(TypedDict):
    """Counts reserved for future budget-consuming graph nodes."""

    llm_calls: int
    tool_calls: int
    repair_attempts: int


class InventoryEntry(TypedDict):
    path: str
    kind: Literal["file", "directory"]


class ExploreFile(TypedDict):
    path: str
    content: str


class ExploreState(TypedDict, total=False):
    mode: Literal["inventory", "code_question"]
    inventory: list[InventoryEntry]
    inventory_truncated: bool
    directories_visited: int
    selected_paths: list[str]
    file_contents: list[ExploreFile]
    answer: str


class RunState(TypedDict, total=False):
    command: dict[str, JsonValue]
    tool_result: JsonValue | None
    verification: dict[str, JsonValue]
    answer: str


class OrchestrationState(TypedDict, total=False):
    """JSON-shaped state shared by LangGraph nodes."""

    task_id: str
    user_request: str
    trajectory: str | None
    routing_decision: dict[str, JsonValue] | None
    current_node: str
    execution_budget: dict[str, int | float]
    counters: ExecutionCounters
    failure: dict[str, JsonValue] | None
    explore: ExploreState
    run: RunState
