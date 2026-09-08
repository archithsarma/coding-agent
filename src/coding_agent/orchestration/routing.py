"""High-confidence deterministic request routing."""

from __future__ import annotations

import re

from coding_agent.domain import Trajectory
from coding_agent.orchestration.state import RoutingDecision

_CORRECTION_EXACT = {
    "undo",
    "undo that",
    "revert that",
    "revert the last change",
}
_RUN_EXACT = {
    "run tests",
    "run the tests",
    "run pytest",
    "check lint",
    "run lint",
    "run mypy",
    "type check",
}
_EXPLORE_EXACT = {
    "what files are in this project?",
    "list files",
    "show me the project structure",
}
_EDIT_PREFIXES = ("add ", "change ", "fix ", "update ", "remove ", "rename ")


def normalize_request(request: str) -> str:
    """Normalize only routing input; callers retain the original request."""

    return re.sub(r"\s+", " ", request.strip()).casefold()


def route_request(request: str) -> RoutingDecision:
    """Select a trajectory only when a small explicit rule is high-confidence."""

    normalized = normalize_request(request)

    # Conflict precedence is intentional: correction > edit > run > explore.
    if normalized in _CORRECTION_EXACT or normalized.startswith(("undo ", "revert ")):
        return RoutingDecision(
            trajectory=Trajectory.CORRECTION,
            source="deterministic",
            reason="explicit correction request",
        )
    if normalized.startswith(_EDIT_PREFIXES):
        return RoutingDecision(
            trajectory=Trajectory.EDIT,
            source="deterministic",
            reason="explicit editing verb",
        )
    if normalized in _RUN_EXACT or normalized.startswith(
        ("run tests ", "run the tests ", "run pytest ", "run mypy ")
    ):
        return RoutingDecision(
            trajectory=Trajectory.RUN,
            source="deterministic",
            reason="explicit verification request",
        )
    if normalized in _EXPLORE_EXACT or normalized.startswith(
        ("what does ", "show me how ")
    ):
        return RoutingDecision(
            trajectory=Trajectory.EXPLORE,
            source="deterministic",
            reason="explicit information request",
        )
    return RoutingDecision(source="deterministic", reason="no high-confidence rule")
