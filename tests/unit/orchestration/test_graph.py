import pytest

from coding_agent.domain import ExecutionBudget, InvalidRequestError, Trajectory
from coding_agent.orchestration.graph import (
    BOUNDARY_CORRECTION,
    BOUNDARY_EDIT,
    BOUNDARY_EXPLORE,
    BOUNDARY_RUN,
    BOUNDARY_UNRESOLVED,
    build_graph,
)


@pytest.mark.parametrize(
    ("user_request", "boundary", "trajectory"),
    [
        ("run tests", BOUNDARY_RUN, Trajectory.RUN.value),
        ("undo that", BOUNDARY_CORRECTION, Trajectory.CORRECTION.value),
        ("what does create_task do", BOUNDARY_EXPLORE, Trajectory.EXPLORE.value),
        ("add title validation", BOUNDARY_EDIT, Trajectory.EDIT.value),
        ("validation", BOUNDARY_UNRESOLVED, None),
    ],
)
def test_compiled_graph_reaches_expected_boundary(
    user_request: str, boundary: str, trajectory: str | None
) -> None:
    result = build_graph().invoke({"task_id": "task-1", "user_request": user_request})

    assert result["current_node"] == boundary
    assert result["trajectory"] == trajectory
    assert result["task_id"] == "task-1"
    assert result["user_request"] == user_request


def test_graph_preserves_budget_and_initializes_counters() -> None:
    budget = ExecutionBudget(
        max_llm_calls=2,
        max_tool_calls=3,
        max_repair_attempts=1,
        max_shell_execution_seconds=10,
    )

    result = build_graph().invoke(
        {
            "task_id": "task-1",
            "user_request": "run tests",
            "execution_budget": budget.model_dump(mode="json"),
        }
    )

    assert result["execution_budget"] == budget.model_dump(mode="json")
    assert result["counters"] == {
        "llm_calls": 0,
        "tool_calls": 0,
        "repair_attempts": 0,
    }


@pytest.mark.parametrize("field", ["task_id", "user_request"])
def test_graph_rejects_missing_required_request_state(field: str) -> None:
    state = {"task_id": "task-1", "user_request": "run tests"}
    del state[field]

    with pytest.raises(InvalidRequestError):
        build_graph().invoke(state)
