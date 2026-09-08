"""Deterministic, session-local Correction/Undo trajectory."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

from langgraph.runtime import Runtime

from coding_agent.content import sha256_text
from coding_agent.domain import (
    FileChange,
    OperationRecord,
    OperationStatus,
    ToolErrorInfo,
    ToolRequest,
    ToolResult,
    Trajectory,
)
from coding_agent.editing import ContentRestoreTransaction, RestoreItem
from coding_agent.journal import ReversibleOperation
from coding_agent.orchestration.context import OrchestrationContext
from coding_agent.orchestration.state import CorrectionState, OrchestrationState
from coding_agent.policies import WorkspacePathPolicy

CORRECTION_LOOKUP = "correction_lookup"
CORRECTION_PREFLIGHT = "correction_preflight"
CORRECTION_PREPARE = "correction_prepare"
CORRECTION_COMMIT = "correction_commit"
CORRECTION_COMPLETE = "correction_complete"
CORRECTION_FAILED = "correction_failed"


def _correction(state: OrchestrationState) -> CorrectionState:
    return state.get("correction", {})


def _failure(
    state: OrchestrationState,
    *,
    code: str,
    message: str,
    node: str,
    details: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "failure": {
            "code": code,
            "message": message,
            "node": node,
            "retryable": False,
            "details": details or {},
        },
        "current_node": node,
    }


def _context(runtime: Runtime[OrchestrationContext]) -> OrchestrationContext:
    if runtime.context is None:
        raise RuntimeError("Correction requires runtime context")
    return runtime.context


def lookup(
    state: OrchestrationState, runtime: Runtime[OrchestrationContext]
) -> dict[str, object]:
    if runtime.context is None:
        return _failure(
            state,
            code="no_reversible_operation",
            message="no safely reversible edit is available in this session",
            node=CORRECTION_LOOKUP,
        )
    context = _context(runtime)
    entry = context.journal.latest()
    if entry is None:
        return _failure(
            state,
            code="no_reversible_operation",
            message="no safely reversible edit is available in this session",
            node=CORRECTION_LOOKUP,
        )
    if not entry.available:
        return _failure(
            state,
            code="reversible_snapshot_unavailable",
            message="the reversible snapshot is no longer available",
            node=CORRECTION_LOOKUP,
        )
    return {
        "correction": {
            "operation_id": entry.operation.operation_id,
            "preflight_complete": False,
        },
        "current_node": CORRECTION_LOOKUP,
    }


def lookup_next(state: OrchestrationState) -> str:
    return CORRECTION_FAILED if state.get("failure") else CORRECTION_PREFLIGHT


async def preflight(
    state: OrchestrationState, runtime: Runtime[OrchestrationContext]
) -> dict[str, object]:
    context = _context(runtime)
    entry = _entry(context, state)
    if entry is None:
        return _failure(
            state,
            code="reversible_snapshot_unavailable",
            message="the reversible snapshot is no longer available",
            node=CORRECTION_PREFLIGHT,
        )
    required = len(entry.files)
    if not _reserve(state, required):
        return _failure(
            state,
            code="correction_budget_exhausted",
            message="insufficient tool budget to preflight correction",
            node=CORRECTION_PREFLIGHT,
            details={"required_calls": required, "remaining_calls": _remaining(state)},
        )
    calls = 0
    for index, item in enumerate(entry.files):
        result = await context.tools.invoke(
            ToolRequest(
                call_id=f"correction-preflight-{index}-{item.path}",
                capability="filesystem.read",
                arguments={"path": item.path},
            )
        )
        calls += 1
        if not result.success:
            return _tool_failure(state, result, CORRECTION_PREFLIGHT, calls)
        content = result.data.get("content") if isinstance(result.data, dict) else None
        if not isinstance(content, str):
            return _failure(
                cast(OrchestrationState, {**state, "counters": _count(state, calls)}),
                code="correction_write_failed",
                message="filesystem read content was malformed",
                node=CORRECTION_PREFLIGHT,
            ) | {"counters": _count(state, calls)}
        if sha256_text(content) != item.final_hash:
            return _failure(
                cast(OrchestrationState, {**state, "counters": _count(state, calls)}),
                code="undo_conflict",
                message="the workspace changed after the edit; undo was not applied",
                node=CORRECTION_PREFLIGHT,
                details={"path": item.path},
            ) | {"counters": _count(state, calls)}
    return {
        "counters": _count(state, calls),
        "correction": {**_correction(state), "preflight_complete": True},
        "current_node": CORRECTION_PREFLIGHT,
    }


def preflight_next(state: OrchestrationState) -> str:
    return CORRECTION_FAILED if state.get("failure") else CORRECTION_PREPARE


def prepare(
    state: OrchestrationState, runtime: Runtime[OrchestrationContext]
) -> dict[str, object]:
    context = _context(runtime)
    entry = _entry(context, state)
    if entry is None:
        return _failure(
            state,
            code="reversible_snapshot_unavailable",
            message="the reversible snapshot is no longer available",
            node=CORRECTION_PREPARE,
        )
    if not _reserve(state, _restore_budget(len(entry.files))):
        return _failure(
            state,
            code="correction_budget_exhausted",
            message="insufficient tool budget to restore correction safely",
            node=CORRECTION_PREPARE,
        )
    return {"current_node": CORRECTION_PREPARE}


def prepare_next(state: OrchestrationState) -> str:
    return CORRECTION_FAILED if state.get("failure") else CORRECTION_COMMIT


async def commit(
    state: OrchestrationState, runtime: Runtime[OrchestrationContext]
) -> dict[str, object]:
    context = _context(runtime)
    entry = _entry(context, state)
    if entry is None:
        return _failure(
            state,
            code="reversible_snapshot_unavailable",
            message="the reversible snapshot is no longer available",
            node=CORRECTION_COMMIT,
        )
    calls = 0

    async def invoke(request: ToolRequest) -> ToolResult:
        nonlocal calls
        if _remaining(state) - calls <= 0:
            return ToolResult(
                call_id=request.call_id,
                success=False,
                error=ToolErrorInfo(
                    code="correction_budget_exhausted",
                    message="tool budget exhausted during correction",
                ),
            )
        calls += 1
        return await context.tools.invoke(request)

    result = await ContentRestoreTransaction(
        invoke,
        path_policy=_require_path_policy(context),
        max_write_bytes=context.edit_config.max_write_bytes,
    ).execute([RestoreItem(item.path, item.final, item.before) for item in entry.files])
    counters = _count(state, calls)
    if not result.success:
        details: dict[str, object] = {
            "rollback_complete": result.rollback_complete,
            "rollback_performed": result.rollback_performed,
            "partial_mutation_risk": result.rollback_complete is False,
            "rollback_errors": list(result.rollback_errors),
            "rollback_conflicts": [item.path for item in result.rollback_conflicts],
        }
        return _failure(
            cast(OrchestrationState, {**state, "counters": counters}),
            code=(
                "correction_partial_mutation"
                if result.rollback_complete is False
                else "correction_write_failed"
            ),
            message=str(result.error),
            node=CORRECTION_COMMIT,
            details=details,
        ) | {"counters": counters}
    return {
        "counters": counters,
        "correction": {
            **_correction(state),
            "file_changes": [
                item.model_dump(mode="json") for item in result.file_changes
            ],
        },
        "current_node": CORRECTION_COMMIT,
    }


def commit_next(state: OrchestrationState) -> str:
    return CORRECTION_FAILED if state.get("failure") else CORRECTION_COMPLETE


def complete(
    state: OrchestrationState, runtime: Runtime[OrchestrationContext]
) -> dict[str, object]:
    context = _context(runtime)
    entry = _entry(context, state)
    if entry is None:
        return _failure(
            state,
            code="reversible_snapshot_unavailable",
            message="the reversible snapshot is no longer available",
            node=CORRECTION_COMPLETE,
        )
    entry.operation.status = OperationStatus.REVERTED
    context.journal.mark_reverted(entry.operation.operation_id)
    raw_changes = _correction(state).get("file_changes", [])
    changes = (
        [FileChange.model_validate(item) for item in raw_changes]
        if isinstance(raw_changes, list)
        else []
    )
    record = OperationRecord(
        operation_id=f"correction-{uuid4().hex}",
        trajectory=Trajectory.CORRECTION,
        user_request=state["user_request"],
        status=OperationStatus.SUCCEEDED,
        file_changes=changes,
        parent_operation_id=entry.operation.operation_id,
        created_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )
    return {
        "correction": {
            **_correction(state),
            "operation_record": record.model_dump(mode="json"),
            "answer": (
                "Reverted the previous edit. You can give me the corrected "
                "change you'd like."
            ),
        },
        "current_node": CORRECTION_COMPLETE,
    }


def failed(state: OrchestrationState) -> dict[str, object]:
    failure = state.get("failure") or {}
    details = failure.get("details", {}) if isinstance(failure, dict) else {}
    if isinstance(details, dict) and details.get("partial_mutation_risk"):
        answer = (
            "Undo failed and rollback was incomplete. The workspace may contain "
            "partial changes."
        )
    elif failure.get("code") == "undo_conflict":
        answer = "Undo was not applied because the workspace changed after the edit."
    else:
        answer = (
            "Undo failed: "
            f"{failure.get('message', 'no reversible operation is available')}"
        )
    return {
        "correction": {**_correction(state), "answer": answer},
        "current_node": CORRECTION_FAILED,
    }


def _entry(
    context: OrchestrationContext, state: OrchestrationState
) -> ReversibleOperation | None:
    operation_id = _correction(state).get("operation_id")
    entry = context.journal.latest()
    if entry is None or entry.operation.operation_id != operation_id:
        return None
    return entry


def _remaining(state: OrchestrationState) -> int:
    return (
        int(state["execution_budget"]["max_tool_calls"])
        - state["counters"]["tool_calls"]
    )


def _reserve(state: OrchestrationState, required: int) -> bool:
    return _remaining(state) >= required


def _count(state: OrchestrationState, calls: int) -> dict[str, int]:
    counters = state["counters"]
    return {
        "llm_calls": counters["llm_calls"],
        "tool_calls": counters["tool_calls"] + calls,
        "repair_attempts": counters["repair_attempts"],
    }


def _restore_budget(file_count: int) -> int:
    return 6 * file_count


def _tool_failure(
    state: OrchestrationState, result: ToolResult, node: str, calls: int
) -> dict[str, object]:
    error = result.error
    return _failure(
        cast(OrchestrationState, {**state, "counters": _count(state, calls)}),
        code=error.code if error else "correction_write_failed",
        message=error.message if error else "correction tool failed",
        node=node,
    ) | {"counters": _count(state, calls)}


def _require_path_policy(context: OrchestrationContext) -> WorkspacePathPolicy:
    if context.path_policy is None:
        raise RuntimeError("Correction requires an explicit workspace path policy")
    return context.path_policy
