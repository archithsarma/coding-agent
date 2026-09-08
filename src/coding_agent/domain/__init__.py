"""Public framework-independent contracts for the coding agent."""

from coding_agent.domain.enums import (
    ChangeType,
    OperationStatus,
    Trajectory,
    VerificationKind,
)
from coding_agent.domain.errors import (
    DomainError,
    InvalidRequestError,
    OperationConflictError,
    PolicyViolationError,
    ToolExecutionError,
    VerificationFailureError,
)
from coding_agent.domain.models import (
    AgentTaskContext,
    ExecutionBudget,
    FileChange,
    OperationRecord,
    ToolErrorInfo,
    ToolRequest,
    ToolResult,
    VerificationResult,
)

__all__ = [
    "AgentTaskContext",
    "ChangeType",
    "DomainError",
    "ExecutionBudget",
    "FileChange",
    "InvalidRequestError",
    "OperationConflictError",
    "OperationRecord",
    "OperationStatus",
    "PolicyViolationError",
    "ToolErrorInfo",
    "ToolExecutionError",
    "ToolRequest",
    "ToolResult",
    "Trajectory",
    "VerificationFailureError",
    "VerificationKind",
    "VerificationResult",
]
