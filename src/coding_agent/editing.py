"""Deterministic, workspace-constrained filesystem edit transactions."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from difflib import unified_diff

from coding_agent.content import sha256_text
from coding_agent.domain import (
    ChangeType,
    FileChange,
    InvalidRequestError,
    OperationConflictError,
    ToolErrorInfo,
    ToolRequest,
    ToolResult,
)
from coding_agent.policies import WorkspacePathPolicy

ToolInvoker = Callable[[ToolRequest], Awaitable[ToolResult]]


@dataclass(frozen=True)
class FileSnapshot:
    path: str
    content: str
    sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.path.strip() or "\x00" in self.path:
            raise ValueError(
                "snapshot path must be non-blank and contain no null bytes"
            )
        actual = sha256_text(self.content)
        if self.sha256 is not None and self.sha256 != actual:
            raise ValueError("snapshot sha256 does not match content")
        object.__setattr__(self, "sha256", actual)


@dataclass(frozen=True)
class TextReplacement:
    old_text: str
    new_text: str
    expected_occurrences: int = 1

    def __post_init__(self) -> None:
        if not self.old_text:
            raise ValueError("old_text must not be empty")
        if self.old_text == self.new_text:
            raise ValueError("replacement must change the content")
        if self.expected_occurrences <= 0:
            raise ValueError("expected_occurrences must be greater than zero")


@dataclass(frozen=True)
class FileEditPlan:
    path: str
    replacements: tuple[TextReplacement, ...]

    def __post_init__(self) -> None:
        if not self.path.strip() or "\x00" in self.path:
            raise ValueError("edit path must be non-blank and contain no null bytes")
        if not self.replacements:
            raise ValueError("edit plan requires at least one replacement")
        object.__setattr__(self, "replacements", tuple(self.replacements))


@dataclass(frozen=True)
class RollbackConflict:
    path: str
    current_hash: str
    expected_after_hash: str


@dataclass(frozen=True)
class EditTransactionResult:
    success: bool
    file_changes: tuple[FileChange, ...] = ()
    rollback_performed: bool = False
    rollback_complete: bool | None = None
    error: ToolResult | Exception | None = None
    rollback_errors: tuple[str, ...] = ()
    rollback_conflicts: tuple[RollbackConflict, ...] = ()


@dataclass(frozen=True)
class _Candidate:
    snapshot: FileSnapshot
    content: str
    change: FileChange


class EditTransaction:
    """Apply validated existing-file edits through a caller-controlled invoker."""

    def __init__(
        self,
        invoker: ToolInvoker,
        *,
        path_policy: WorkspacePathPolicy,
        max_write_bytes: int,
    ) -> None:
        if max_write_bytes <= 0:
            raise ValueError("max_write_bytes must be greater than zero")
        self._invoke = invoker
        self._path_policy = path_policy
        self._max_write_bytes = max_write_bytes

    async def execute(
        self,
        plans: Iterable[FileEditPlan],
        snapshots: Iterable[FileSnapshot],
    ) -> EditTransactionResult:
        plans_tuple = tuple(plans)
        snapshots_tuple = tuple(snapshots)
        if len({snapshot.path for snapshot in snapshots_tuple}) != len(snapshots_tuple):
            raise InvalidRequestError("snapshot paths must be unique")
        snapshots_by_path = {snapshot.path: snapshot for snapshot in snapshots_tuple}
        candidates = self._validate_and_compute(plans_tuple, snapshots_by_path)

        preflight_failure = await self._preflight(candidates)
        if preflight_failure is not None:
            return preflight_failure

        written: list[_Candidate] = []
        for candidate in candidates:
            request = ToolRequest(
                call_id=f"edit-write-{len(written)}-{candidate.snapshot.path}",
                capability="filesystem.write",
                arguments={
                    "path": candidate.snapshot.path,
                    "content": candidate.content,
                    "expected_sha256": candidate.snapshot.sha256,
                },
            )
            result = await self._invoke(request)
            if not result.success:
                return await self._failed(result, written)
            written.append(candidate)

            verified = await self._read(candidate.snapshot.path, len(written), "verify")
            if isinstance(verified, ToolResult):
                return await self._failed(verified, written)
            if verified.sha256 != candidate.change.after_hash:
                mismatch = OperationConflictError(
                    f"post-write content mismatch for '{candidate.snapshot.path}'"
                )
                return await self._failed(mismatch, written)

        return EditTransactionResult(
            success=True,
            file_changes=tuple(candidate.change for candidate in candidates),
        )

    def _validate_and_compute(
        self,
        plans: tuple[FileEditPlan, ...],
        snapshots: dict[str, FileSnapshot],
    ) -> tuple[_Candidate, ...]:
        if not plans:
            raise InvalidRequestError("edit transaction requires at least one plan")
        if len({plan.path for plan in plans}) != len(plans):
            raise InvalidRequestError("edit plan paths must be unique")

        candidates: list[_Candidate] = []
        for plan in plans:
            self._path_policy.resolve_path(plan.path)
            try:
                snapshot = snapshots[plan.path]
            except KeyError as error:
                raise OperationConflictError(
                    f"no snapshot supplied for '{plan.path}'"
                ) from error
            content = snapshot.content
            for replacement in plan.replacements:
                occurrences = content.count(replacement.old_text)
                if occurrences != replacement.expected_occurrences:
                    raise InvalidRequestError(
                        f"replacement for '{plan.path}' expected "
                        f"{replacement.expected_occurrences} occurrence(s), "
                        f"found {occurrences}"
                    )
                content = content.replace(
                    replacement.old_text,
                    replacement.new_text,
                    replacement.expected_occurrences,
                )
            if content == snapshot.content:
                raise InvalidRequestError(f"edit for '{plan.path}' makes no change")
            if len(content.encode("utf-8")) > self._max_write_bytes:
                raise InvalidRequestError(f"edit for '{plan.path}' exceeds write limit")
            candidates.append(
                _Candidate(
                    snapshot=snapshot,
                    content=content,
                    change=FileChange(
                        path=plan.path,
                        change_type=ChangeType.MODIFY,
                        before_hash=snapshot.sha256,
                        after_hash=sha256_text(content),
                        patch=_unified_diff(plan.path, snapshot.content, content),
                    ),
                )
            )
        return tuple(candidates)

    async def _preflight(
        self, candidates: tuple[_Candidate, ...]
    ) -> EditTransactionResult | None:
        for index, candidate in enumerate(candidates):
            current = await self._read(candidate.snapshot.path, index, "preflight")
            if isinstance(current, ToolResult):
                return EditTransactionResult(success=False, error=current)
            if current.sha256 != candidate.snapshot.sha256:
                return EditTransactionResult(
                    success=False,
                    error=OperationConflictError(
                        f"file changed since snapshot: '{candidate.snapshot.path}'"
                    ),
                )
        return None

    async def _read(
        self, path: str, index: int, purpose: str
    ) -> FileSnapshot | ToolResult:
        result = await self._invoke(
            ToolRequest(
                call_id=f"edit-{purpose}-{index}-{path}",
                capability="filesystem.read",
                arguments={"path": path},
            )
        )
        if not result.success:
            return result
        if not isinstance(result.data, dict):
            return ToolResult(
                call_id=result.call_id,
                success=False,
                error=ToolErrorInfo(
                    code="malformed_response",
                    message="filesystem read data was malformed",
                ),
            )
        content = result.data.get("content")
        if not isinstance(content, str):
            return ToolResult(
                call_id=result.call_id,
                success=False,
                error=ToolErrorInfo(
                    code="malformed_response",
                    message="filesystem read content was malformed",
                ),
            )
        return FileSnapshot(path=path, content=content)

    async def _failed(
        self, error: ToolResult | Exception, written: list[_Candidate]
    ) -> EditTransactionResult:
        if not written:
            return EditTransactionResult(success=False, error=error)
        rollback_errors: list[str] = []
        rollback_conflicts: list[RollbackConflict] = []
        for index, candidate in reversed(list(enumerate(written))):
            current = await self._read(candidate.snapshot.path, index, "rollback-check")
            if isinstance(current, ToolResult):
                rollback_errors.append(f"{candidate.snapshot.path}: {current.error}")
                continue
            if current.sha256 != candidate.change.after_hash:
                rollback_conflicts.append(
                    RollbackConflict(
                        path=candidate.snapshot.path,
                        current_hash=current.sha256 or "",
                        expected_after_hash=candidate.change.after_hash or "",
                    )
                )
                continue
            restore = await self._invoke(
                ToolRequest(
                    call_id=f"edit-rollback-{index}-{candidate.snapshot.path}",
                    capability="filesystem.write",
                    arguments={
                        "path": candidate.snapshot.path,
                        "content": candidate.snapshot.content,
                        "expected_sha256": candidate.change.after_hash,
                    },
                )
            )
            if not restore.success:
                rollback_errors.append(f"{candidate.snapshot.path}: {restore.error}")
        return EditTransactionResult(
            success=False,
            file_changes=tuple(candidate.change for candidate in written),
            rollback_performed=True,
            rollback_complete=not rollback_errors and not rollback_conflicts,
            error=error,
            rollback_errors=tuple(rollback_errors),
            rollback_conflicts=tuple(rollback_conflicts),
        )


def _unified_diff(path: str, before: str, after: str) -> str:
    return "".join(
        unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )
