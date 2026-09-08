from coding_agent.domain.errors import (
    DomainError,
    InvalidRequestError,
    OperationConflictError,
    PolicyViolationError,
    ToolExecutionError,
    VerificationFailureError,
)


def test_domain_errors_expose_machine_readable_context() -> None:
    errors = [
        InvalidRequestError("bad request"),
        ToolExecutionError("tool failed"),
        PolicyViolationError("blocked"),
        VerificationFailureError("check failed"),
        OperationConflictError("cannot revert"),
    ]

    assert all(isinstance(error, DomainError) for error in errors)
    assert [error.code for error in errors] == [
        "invalid_request",
        "tool_execution_failed",
        "policy_violation",
        "verification_failed",
        "operation_conflict",
    ]
