from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from coding_agent.domain import (
    AgentTaskContext,
    ChangeType,
    ExecutionBudget,
    FileChange,
    OperationRecord,
    OperationStatus,
    ToolErrorInfo,
    ToolRequest,
    ToolResult,
    Trajectory,
    VerificationKind,
    VerificationResult,
)


def test_enums_use_stable_serializable_values() -> None:
    assert Trajectory.EDIT.value == "edit"
    assert OperationStatus.SUCCEEDED.value == "succeeded"
    assert ChangeType.DELETE.value == "delete"
    assert VerificationKind.TYPE_CHECK.value == "type_check"


def test_verification_result_supports_commandless_and_command_checks() -> None:
    result = VerificationResult(
        kind=VerificationKind.SYNTAX,
        passed=True,
        duration_seconds=0.25,
        metadata={"files_checked": 2},
    )

    assert VerificationResult.model_validate_json(result.model_dump_json()) == result


def test_execution_budget_rejects_negative_limits() -> None:
    with pytest.raises(ValidationError):
        ExecutionBudget(
            max_llm_calls=-1,
            max_tool_calls=1,
            max_repair_attempts=1,
            max_shell_execution_seconds=1,
        )


def test_file_change_accepts_only_portable_workspace_relative_paths() -> None:
    change = FileChange(
        path="src\\coding_agent\\domain.py", change_type=ChangeType.MODIFY
    )

    assert change.path == "src/coding_agent/domain.py"

    with pytest.raises(ValidationError):
        FileChange(path="../outside.py", change_type=ChangeType.CREATE)

    with pytest.raises(ValidationError):
        FileChange(path="/absolute.py", change_type=ChangeType.CREATE)

    with pytest.raises(ValidationError):
        FileChange(path="C:relative.py", change_type=ChangeType.CREATE)


def test_operation_parent_identifier_cannot_be_blank() -> None:
    with pytest.raises(ValidationError):
        OperationRecord(
            operation_id="op-2",
            trajectory=Trajectory.EDIT,
            user_request="Update the file",
            parent_operation_id=" ",
        )


def test_operation_lifecycle_requires_consistent_completion_timestamp() -> None:
    created_at = datetime(2026, 1, 1, tzinfo=UTC)

    with pytest.raises(ValidationError):
        OperationRecord(
            operation_id="op-1",
            trajectory=Trajectory.EDIT,
            user_request="Update the file",
            status=OperationStatus.SUCCEEDED,
            created_at=created_at,
        )

    operation = OperationRecord(
        operation_id="op-1",
        trajectory=Trajectory.EDIT,
        user_request="Update the file",
        status=OperationStatus.SUCCEEDED,
        created_at=created_at,
        completed_at=datetime(2026, 1, 1, 1, tzinfo=UTC),
    )
    restored = OperationRecord.model_validate_json(operation.model_dump_json())

    assert restored == operation
    assert restored.created_at.tzinfo == UTC


def test_tool_contracts_accept_json_payloads_and_require_failure_details() -> None:
    request = ToolRequest(
        call_id="call-1",
        capability="filesystem.read",
        arguments={"path": "src/main.py", "line": 4, "include": ["code"]},
    )
    result = ToolResult(
        call_id=request.call_id,
        success=False,
        error=ToolErrorInfo(code="not_found", message="File was not found"),
        metadata={"retryable": False},
    )

    assert ToolRequest.model_validate_json(request.model_dump_json()) == request
    assert ToolResult.model_validate_json(result.model_dump_json()) == result

    with pytest.raises(ValidationError):
        ToolResult(call_id="call-2", success=False)


def test_required_identifiers_and_task_context_are_not_blank() -> None:
    with pytest.raises(ValidationError):
        AgentTaskContext(
            task_id=" ", user_request="Inspect this", trajectory=Trajectory.EXPLORE
        )

    context = AgentTaskContext(
        task_id="task-1",
        user_request="Inspect this",
        trajectory=Trajectory.EXPLORE,
    )

    assert context.current_operation_id is None
