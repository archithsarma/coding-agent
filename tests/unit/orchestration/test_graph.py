import pytest

from coding_agent.domain import ExecutionBudget, InvalidRequestError, Trajectory
from coding_agent.memory import SQLitePreferenceStore
from coding_agent.orchestration.context import OrchestrationContext
from coding_agent.orchestration.graph import (
    BOUNDARY_UNRESOLVED,
    build_graph,
)


@pytest.mark.parametrize(
    ("user_request", "boundary", "trajectory"),
    [
        ("undo that", "correction_failed", Trajectory.CORRECTION.value),
        ("validation", BOUNDARY_UNRESOLVED, None),
    ],
)
@pytest.mark.anyio
async def test_compiled_graph_reaches_expected_boundary(
    user_request: str, boundary: str, trajectory: str | None
) -> None:
    result = await build_graph().ainvoke(
        {"task_id": "task-1", "user_request": user_request}
    )

    assert result["current_node"] == boundary
    assert result["trajectory"] == trajectory
    assert result["task_id"] == "task-1"
    assert result["user_request"] == user_request


@pytest.mark.anyio
async def test_graph_preserves_budget_and_initializes_counters() -> None:
    budget = ExecutionBudget(
        max_llm_calls=2,
        max_tool_calls=3,
        max_repair_attempts=1,
        max_shell_execution_seconds=10,
    )

    result = await build_graph().ainvoke(
        {
            "task_id": "task-1",
            "user_request": "validation",
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
@pytest.mark.anyio
async def test_graph_rejects_missing_required_request_state(field: str) -> None:
    state = {"task_id": "task-1", "user_request": "run tests"}
    del state[field]

    with pytest.raises(InvalidRequestError):
        await build_graph().ainvoke(state)


@pytest.mark.anyio
async def test_explicit_remember_request_persists_without_new_trajectory(
    tmp_path,
) -> None:
    store = SQLitePreferenceStore(tmp_path / "preferences.sqlite3")

    result = await build_graph().ainvoke(
        {
            "task_id": "task-1",
            "user_request": "Remember: always use type hints.",
        },
        context=OrchestrationContext(
            model=object(),
            tools=object(),
            preference_store=store,
        ),
    )

    assert result["current_node"] == BOUNDARY_UNRESOLVED
    assert result["memory"]["preference_saved"] is True
    assert store.list_preferences()[0].value == "always use type hints."
