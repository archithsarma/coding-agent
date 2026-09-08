"""Construction of the LangGraph orchestration workflow."""

from __future__ import annotations

from typing import Any, Literal, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import ValidationError

import coding_agent.orchestration.run as run_nodes
from coding_agent.domain import ExecutionBudget, InvalidRequestError, Trajectory
from coding_agent.orchestration.context import OrchestrationContext
from coding_agent.orchestration.edit import (
    EDIT_COMMIT,
    EDIT_COMPLETE,
    EDIT_FAILED,
    EDIT_INVENTORY,
    EDIT_PLAN,
    EDIT_PREPARE,
    EDIT_READ,
    EDIT_SELECT,
)
from coding_agent.orchestration.edit import (
    commit as edit_commit,
)
from coding_agent.orchestration.edit import (
    commit_next as edit_commit_next,
)
from coding_agent.orchestration.edit import (
    complete as edit_complete,
)
from coding_agent.orchestration.edit import (
    failed as edit_failed,
)
from coding_agent.orchestration.edit import (
    inventory as edit_inventory,
)
from coding_agent.orchestration.edit import (
    inventory_next as edit_inventory_next,
)
from coding_agent.orchestration.edit import (
    plan as edit_plan,
)
from coding_agent.orchestration.edit import (
    plan_next as edit_plan_next,
)
from coding_agent.orchestration.edit import (
    prepare as edit_prepare,
)
from coding_agent.orchestration.edit import (
    prepare_next as edit_prepare_next,
)
from coding_agent.orchestration.edit import (
    read as edit_read,
)
from coding_agent.orchestration.edit import (
    read_next as edit_read_next,
)
from coding_agent.orchestration.edit import (
    select as edit_select,
)
from coding_agent.orchestration.edit import (
    select_next as edit_select_next,
)
from coding_agent.orchestration.explore import (
    EXPLORE_ANSWER_INVENTORY,
    EXPLORE_ANSWER_ZERO,
    EXPLORE_EXPLAIN,
    EXPLORE_FAILED,
    EXPLORE_INVENTORY,
    EXPLORE_PLAN,
    EXPLORE_READ,
    EXPLORE_SELECT,
    answer_inventory,
    answer_zero,
    explain,
    failed,
    inventory,
    inventory_next,
    plan,
    plan_next,
    read,
    read_next,
    select,
    select_next,
)
from coding_agent.orchestration.routing import route_request
from coding_agent.orchestration.run import (
    RUN_COMPLETE,
    RUN_EXECUTE,
    RUN_FAILED,
    RUN_INTERPRET,
    RUN_PLAN,
    complete,
    execute,
    execute_next,
    interpret,
    interpret_next,
)
from coding_agent.orchestration.state import (
    ExecutionCounters,
    OrchestrationState,
)

RouteTarget = Literal[
    "explore_inventory",
    "run_plan",
    "edit_inventory",
    "route_correction",
    "route_unresolved",
]

INITIALIZE = "initialize"
ROUTE = "route"
BOUNDARY_EXPLORE = EXPLORE_INVENTORY
BOUNDARY_EDIT: RouteTarget = cast(RouteTarget, EDIT_INVENTORY)
BOUNDARY_RUN: RouteTarget = cast(RouteTarget, RUN_PLAN)
BOUNDARY_CORRECTION: RouteTarget = "route_correction"
BOUNDARY_UNRESOLVED: RouteTarget = "route_unresolved"

_DEFAULT_BUDGET = ExecutionBudget(
    max_llm_calls=2,
    max_tool_calls=64,
    max_repair_attempts=0,
    max_shell_execution_seconds=0,
)
_DEFAULT_COUNTERS: ExecutionCounters = {
    "llm_calls": 0,
    "tool_calls": 0,
    "repair_attempts": 0,
}


def build_graph(
    default_execution_budget: ExecutionBudget | None = None,
) -> CompiledStateGraph[
    OrchestrationState,
    OrchestrationContext,
    OrchestrationState,
    OrchestrationState,
]:
    """Build a fresh, inspectable graph; no graph singleton or side effects."""

    budget = default_execution_budget or _DEFAULT_BUDGET
    budget_state = budget.model_dump(mode="json")

    def initialize(state: OrchestrationState) -> dict[str, object]:
        task_id = state.get("task_id")
        user_request = state.get("user_request")
        if not isinstance(task_id, str) or not task_id.strip():
            raise InvalidRequestError("task_id must be a non-blank string")
        if not isinstance(user_request, str) or not user_request.strip():
            raise InvalidRequestError("user_request must be a non-blank string")

        raw_budget = state.get("execution_budget", budget_state)
        try:
            validated_budget = ExecutionBudget.model_validate(raw_budget)
        except ValidationError as error:
            raise InvalidRequestError("execution_budget is invalid") from error

        counters = state.get("counters", _DEFAULT_COUNTERS)
        if not _valid_counters(counters):
            raise InvalidRequestError("counters must contain non-negative integers")
        return {
            "task_id": task_id,
            "user_request": user_request,
            "execution_budget": validated_budget.model_dump(mode="json"),
            "counters": dict(counters),
            "trajectory": None,
            "routing_decision": None,
            "failure": None,
            "explore": {},
            "run": {},
            "edit": {},
            "current_node": INITIALIZE,
        }

    def route(state: OrchestrationState) -> dict[str, object]:
        decision = route_request(state["user_request"])
        return {
            "routing_decision": decision.model_dump(mode="json"),
            "trajectory": decision.trajectory.value if decision.trajectory else None,
            "current_node": ROUTE,
        }

    def choose_boundary(
        state: OrchestrationState,
    ) -> RouteTarget:
        routes: dict[str | None, RouteTarget] = {
            Trajectory.EXPLORE.value: cast(RouteTarget, EXPLORE_INVENTORY),
            Trajectory.EDIT.value: BOUNDARY_EDIT,
            Trajectory.RUN.value: BOUNDARY_RUN,
            Trajectory.CORRECTION.value: BOUNDARY_CORRECTION,
            None: BOUNDARY_UNRESOLVED,
        }
        return routes[state.get("trajectory")]

    builder = StateGraph[
        OrchestrationState,
        OrchestrationContext,
        OrchestrationState,
        OrchestrationState,
    ](OrchestrationState, context_schema=OrchestrationContext)
    builder.add_node(INITIALIZE, initialize)
    builder.add_node(ROUTE, route)
    # LangGraph's current overload cannot infer the state type for these
    # module-level boundary callables, so keep the cast at this integration edge.
    builder.add_node(EXPLORE_INVENTORY, cast(Any, inventory))
    builder.add_node(EXPLORE_PLAN, cast(Any, plan))
    builder.add_node(EXPLORE_SELECT, cast(Any, select))
    builder.add_node(EXPLORE_READ, cast(Any, read))
    builder.add_node(EXPLORE_EXPLAIN, cast(Any, explain))
    builder.add_node(EXPLORE_ANSWER_INVENTORY, answer_inventory)
    builder.add_node(EXPLORE_ANSWER_ZERO, answer_zero)
    builder.add_node(EXPLORE_FAILED, failed)
    builder.add_node(EDIT_INVENTORY, cast(Any, edit_inventory))
    builder.add_node(EDIT_SELECT, cast(Any, edit_select))
    builder.add_node(EDIT_READ, cast(Any, edit_read))
    builder.add_node(EDIT_PLAN, cast(Any, edit_plan))
    builder.add_node(EDIT_PREPARE, edit_prepare)
    builder.add_node(EDIT_COMMIT, cast(Any, edit_commit))
    builder.add_node(EDIT_COMPLETE, edit_complete)
    builder.add_node(EDIT_FAILED, edit_failed)
    builder.add_node(RUN_PLAN, run_nodes.plan)
    builder.add_node(RUN_EXECUTE, cast(Any, execute))
    builder.add_node(RUN_INTERPRET, interpret)
    builder.add_node(RUN_COMPLETE, complete)
    builder.add_node(RUN_FAILED, run_nodes.failed)
    builder.add_node(BOUNDARY_CORRECTION, cast(Any, route_correction))
    builder.add_node(BOUNDARY_UNRESOLVED, cast(Any, route_unresolved))
    builder.add_edge(START, INITIALIZE)
    builder.add_edge(INITIALIZE, ROUTE)
    builder.add_conditional_edges(
        ROUTE,
        choose_boundary,
        {
            EXPLORE_INVENTORY: EXPLORE_INVENTORY,
            BOUNDARY_EDIT: BOUNDARY_EDIT,
            BOUNDARY_RUN: BOUNDARY_RUN,
            BOUNDARY_CORRECTION: BOUNDARY_CORRECTION,
            BOUNDARY_UNRESOLVED: BOUNDARY_UNRESOLVED,
        },
    )
    builder.add_conditional_edges(
        EXPLORE_INVENTORY,
        inventory_next,
        {EXPLORE_PLAN: EXPLORE_PLAN, EXPLORE_FAILED: EXPLORE_FAILED},
    )
    builder.add_conditional_edges(
        EXPLORE_PLAN,
        plan_next,
        {
            EXPLORE_ANSWER_INVENTORY: EXPLORE_ANSWER_INVENTORY,
            EXPLORE_SELECT: EXPLORE_SELECT,
        },
    )
    builder.add_conditional_edges(
        EXPLORE_SELECT,
        select_next,
        {
            EXPLORE_FAILED: EXPLORE_FAILED,
            EXPLORE_ANSWER_ZERO: EXPLORE_ANSWER_ZERO,
            EXPLORE_READ: EXPLORE_READ,
        },
    )
    builder.add_conditional_edges(
        EXPLORE_READ,
        read_next,
        {EXPLORE_FAILED: EXPLORE_FAILED, EXPLORE_EXPLAIN: EXPLORE_EXPLAIN},
    )
    builder.add_conditional_edges(
        EDIT_INVENTORY,
        edit_inventory_next,
        {EDIT_FAILED: EDIT_FAILED, EDIT_SELECT: EDIT_SELECT},
    )
    builder.add_conditional_edges(
        EDIT_SELECT,
        edit_select_next,
        {EDIT_FAILED: EDIT_FAILED, EDIT_READ: EDIT_READ},
    )
    builder.add_conditional_edges(
        EDIT_READ,
        edit_read_next,
        {EDIT_FAILED: EDIT_FAILED, EDIT_PLAN: EDIT_PLAN},
    )
    builder.add_conditional_edges(
        EDIT_PLAN,
        edit_plan_next,
        {EDIT_FAILED: EDIT_FAILED, EDIT_PREPARE: EDIT_PREPARE},
    )
    builder.add_conditional_edges(
        EDIT_PREPARE,
        edit_prepare_next,
        {EDIT_FAILED: EDIT_FAILED, EDIT_COMMIT: EDIT_COMMIT},
    )
    builder.add_conditional_edges(
        EDIT_COMMIT,
        edit_commit_next,
        {EDIT_FAILED: EDIT_FAILED, EDIT_COMPLETE: EDIT_COMPLETE},
    )
    builder.add_conditional_edges(
        RUN_PLAN,
        run_nodes.plan_next,
        {RUN_FAILED: RUN_FAILED, RUN_EXECUTE: RUN_EXECUTE},
    )
    builder.add_conditional_edges(
        RUN_EXECUTE,
        execute_next,
        {RUN_FAILED: RUN_FAILED, RUN_INTERPRET: RUN_INTERPRET},
    )
    builder.add_conditional_edges(
        RUN_INTERPRET,
        interpret_next,
        {RUN_FAILED: RUN_FAILED, RUN_COMPLETE: RUN_COMPLETE},
    )
    for name in (
        EXPLORE_ANSWER_INVENTORY,
        EXPLORE_ANSWER_ZERO,
        EXPLORE_EXPLAIN,
        EXPLORE_FAILED,
        RUN_COMPLETE,
        RUN_FAILED,
        EDIT_COMPLETE,
        EDIT_FAILED,
        BOUNDARY_CORRECTION,
        BOUNDARY_UNRESOLVED,
    ):
        builder.add_edge(name, END)
    return builder.compile()


def route_explore(_state: OrchestrationState) -> OrchestrationState:
    return {"current_node": BOUNDARY_EXPLORE}


def route_correction(_state: OrchestrationState) -> OrchestrationState:
    return {"current_node": BOUNDARY_CORRECTION}


def route_unresolved(_state: OrchestrationState) -> OrchestrationState:
    return {"current_node": BOUNDARY_UNRESOLVED}


def _valid_counters(counters: object) -> bool:
    if not isinstance(counters, dict):
        return False
    expected = {"llm_calls", "tool_calls", "repair_attempts"}
    return set(counters) == expected and all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in counters.values()
    )
