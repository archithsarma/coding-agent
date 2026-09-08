import pytest

from coding_agent.domain import Trajectory
from coding_agent.orchestration.routing import route_request


@pytest.mark.parametrize(
    ("user_request", "trajectory"),
    [
        ("undo that", Trajectory.CORRECTION),
        ("revert the last change", Trajectory.CORRECTION),
        ("add validation to create_task", Trajectory.EDIT),
        ("fix the failing tests", Trajectory.EDIT),
        ("change the task model", Trajectory.EDIT),
        ("run tests", Trajectory.RUN),
        ("run pytest", Trajectory.RUN),
        ("check lint", Trajectory.RUN),
        ("run ruff", Trajectory.RUN),
        ("run mypy", Trajectory.RUN),
        ("what files are in this project?", Trajectory.EXPLORE),
        ("what does create_task do?", Trajectory.EXPLORE),
        ("show me how tasks are created", Trajectory.EXPLORE),
        ("how are tasks created?", Trajectory.EXPLORE),
    ],
)
def test_high_confidence_requests_route(
    user_request: str, trajectory: Trajectory
) -> None:
    decision = route_request(user_request)

    assert decision.trajectory == trajectory
    assert decision.source == "deterministic"


@pytest.mark.parametrize("user_request", ["tasks", "look at this", "validation"])
def test_ambiguous_requests_remain_unresolved(user_request: str) -> None:
    assert route_request(user_request).trajectory is None


def test_routing_normalizes_case_and_whitespace_without_replacing_request() -> None:
    decision = route_request("  RUN   TESTS  ")

    assert decision.trajectory == Trajectory.RUN


def test_correction_precedes_edit_when_both_intents_are_present() -> None:
    assert route_request("undo the validation fix").trajectory == Trajectory.CORRECTION


def test_edit_precedes_run_when_fixing_tests() -> None:
    assert route_request("fix the failing tests").trajectory == Trajectory.EDIT
