"""Stable enum values shared by domain contracts."""

from enum import StrEnum


class Trajectory(StrEnum):
    EXPLORE = "explore"
    EDIT = "edit"
    RUN = "run"
    CORRECTION = "correction"


class OperationStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REVERTED = "reverted"


class ChangeType(StrEnum):
    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"


class VerificationKind(StrEnum):
    TEST = "test"
    LINT = "lint"
    TYPE_CHECK = "type_check"
    SYNTAX = "syntax"
