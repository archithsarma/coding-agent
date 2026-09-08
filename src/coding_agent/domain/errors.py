"""Application-understood errors for domain and boundary failures."""

from collections.abc import Mapping

from pydantic import JsonValue


class DomainError(Exception):
    """Base error with a stable code and optional structured context."""

    code = "domain_error"

    def __init__(
        self, message: str, *, context: Mapping[str, JsonValue] | None = None
    ) -> None:
        super().__init__(message)
        self.context = dict(context or {})


class InvalidRequestError(DomainError):
    code = "invalid_request"


class ToolExecutionError(DomainError):
    code = "tool_execution_failed"


class PolicyViolationError(DomainError):
    code = "policy_violation"


class VerificationFailureError(DomainError):
    code = "verification_failed"


class OperationConflictError(DomainError):
    code = "operation_conflict"
