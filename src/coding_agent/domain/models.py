"""Framework-independent models for agent operations and tool boundaries."""

from datetime import UTC, datetime
from pathlib import PurePosixPath, PureWindowsPath
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from coding_agent.domain.enums import (
    ChangeType,
    OperationStatus,
    Trajectory,
    VerificationKind,
)

NonNegativeFloat = Annotated[float, Field(ge=0)]


def _non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("value must not be blank")
    return value


def _optional_non_blank(value: str | None) -> str | None:
    if value is not None:
        _non_blank(value)
    return value


def _utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    return value


def _workspace_relative_path(value: str) -> str:
    normalized = value.replace("\\", "/")
    if not normalized.strip() or "\x00" in normalized:
        raise ValueError("path must be a non-blank path without null bytes")
    windows_path = PureWindowsPath(normalized)
    if (
        PurePosixPath(normalized).is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
    ):
        raise ValueError("path must be workspace-relative")

    path = PurePosixPath(normalized)
    if ".." in path.parts:
        raise ValueError("path must not escape the workspace")
    if str(path) == ".":
        raise ValueError("path must identify a file")
    return str(path)


class DomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VerificationResult(DomainModel):
    kind: VerificationKind
    passed: bool
    command: str | None = None
    exit_code: int | None = None
    stdout: str | None = None
    stderr: str | None = None
    duration_seconds: NonNegativeFloat | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class FileChange(DomainModel):
    path: str
    change_type: ChangeType
    before_hash: str | None = None
    after_hash: str | None = None
    patch: str | None = None

    _validate_path = field_validator("path")(_workspace_relative_path)


class ToolErrorInfo(DomainModel):
    code: str
    message: str
    details: dict[str, JsonValue] = Field(default_factory=dict)

    _validate_code = field_validator("code", "message")(_non_blank)


class ToolRequest(DomainModel):
    call_id: str
    capability: str
    arguments: dict[str, JsonValue] = Field(default_factory=dict)

    _validate_identifiers = field_validator("call_id", "capability")(_non_blank)


class ToolResult(DomainModel):
    call_id: str
    success: bool
    data: JsonValue | None = None
    error: ToolErrorInfo | None = None
    duration_seconds: NonNegativeFloat | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    _validate_call_id = field_validator("call_id")(_non_blank)

    @model_validator(mode="after")
    def validate_outcome(self) -> "ToolResult":
        if self.success and self.error is not None:
            raise ValueError("successful tool results cannot include an error")
        if not self.success and self.error is None:
            raise ValueError("failed tool results must include an error")
        return self


class OperationRecord(DomainModel):
    operation_id: str
    trajectory: Trajectory
    user_request: str
    status: OperationStatus = OperationStatus.PENDING
    file_changes: list[FileChange] = Field(default_factory=list)
    verifications: list[VerificationResult] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    parent_operation_id: str | None = None

    _validate_required_text = field_validator("operation_id", "user_request")(
        _non_blank
    )
    _validate_parent_operation_id = field_validator("parent_operation_id")(
        _optional_non_blank
    )
    _validate_created_at = field_validator("created_at")(_utc_datetime)
    _validate_completed_at = field_validator("completed_at")(_utc_datetime)

    @model_validator(mode="after")
    def validate_lifecycle(self) -> "OperationRecord":
        terminal = self.status in {
            OperationStatus.SUCCEEDED,
            OperationStatus.FAILED,
            OperationStatus.REVERTED,
        }
        if terminal != (self.completed_at is not None):
            raise ValueError(
                "terminal operations require completed_at and others must not set it"
            )
        if self.completed_at is not None and self.completed_at < self.created_at:
            raise ValueError("completed_at cannot precede created_at")
        return self


class ExecutionBudget(DomainModel):
    max_llm_calls: int = Field(ge=0)
    max_tool_calls: int = Field(ge=0)
    max_repair_attempts: int = Field(ge=0)
    max_shell_execution_seconds: NonNegativeFloat


class AgentTaskContext(DomainModel):
    task_id: str
    user_request: str
    trajectory: Trajectory
    current_operation_id: str | None = None

    _validate_required_text = field_validator("task_id", "user_request")(_non_blank)
    _validate_operation_id = field_validator("current_operation_id")(_non_blank)
