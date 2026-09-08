from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from coding_agent.content import sha256_text
from coding_agent.domain import (
    InvalidRequestError,
    OperationConflictError,
    PolicyViolationError,
    ToolErrorInfo,
    ToolRequest,
    ToolResult,
)
from coding_agent.editing import (
    EditTransaction,
    FileEditPlan,
    FileSnapshot,
    TextReplacement,
)
from coding_agent.policies import WorkspacePathPolicy


def replacement(old: str, new: str, expected: int = 1) -> TextReplacement:
    return TextReplacement(old, new, expected)


def test_replacements_are_exact_sequential_and_bounded() -> None:
    plan = FileEditPlan(
        "src/main.py",
        (
            replacement("one", "two"),
            replacement("two", "three"),
        ),
    )
    transaction = EditTransaction(
        lambda request: _unexpected(request),
        path_policy=WorkspacePathPolicy(Path.cwd()),
        max_write_bytes=100,
    )
    candidate = transaction._validate_and_compute(
        (plan,), {"src/main.py": FileSnapshot("src/main.py", "one\n")}
    )[0]
    assert candidate.content == "three\n"
    assert "a/src/main.py" in candidate.change.patch
    assert candidate.change.before_hash == sha256_text("one\n")


@pytest.mark.parametrize(
    "replacement_definition",
    [
        lambda: TextReplacement("", "new"),
        lambda: TextReplacement("old", "old"),
        lambda: TextReplacement("old", "new", 0),
    ],
)
def test_invalid_replacements_are_rejected(replacement_definition) -> None:
    with pytest.raises(ValueError):
        replacement_definition()


def test_missing_or_ambiguous_text_fails_without_mutation() -> None:
    transaction = EditTransaction(
        lambda request: _unexpected(request),
        path_policy=WorkspacePathPolicy(Path.cwd()),
        max_write_bytes=100,
    )
    snapshot = FileSnapshot("main.py", "same same")
    with pytest.raises(InvalidRequestError):
        transaction._validate_and_compute(
            (FileEditPlan("main.py", (replacement("missing", "x"),)),),
            {snapshot.path: snapshot},
        )
    with pytest.raises(InvalidRequestError):
        transaction._validate_and_compute(
            (FileEditPlan("main.py", (replacement("same", "x"),)),),
            {snapshot.path: snapshot},
        )


class FakeInvoker:
    def __init__(self, contents: dict[str, str]) -> None:
        self.contents = contents
        self.calls: list[ToolRequest] = []
        self.fail_write_path: str | None = None
        self.mismatch_path: str | None = None
        self.read_counts: dict[str, int] = {}
        self.rollback_fail = False
        self.post_write_mismatch_path: str | None = None

    async def __call__(self, request: ToolRequest) -> ToolResult:
        self.calls.append(request)
        path = request.arguments.get("path")
        assert isinstance(path, str)
        if request.capability == "filesystem.read":
            self.read_counts[path] = self.read_counts.get(path, 0) + 1
            content = self.contents[path]
            if self.post_write_mismatch_path == path and self.read_counts[path] >= 2:
                content = "unexpected"
                self.contents[path] = content
            if self.mismatch_path == path and self.read_counts[path] >= 3:
                content = "external"
                self.contents[path] = content
            return ToolResult(
                call_id=request.call_id,
                success=True,
                data={"path": path, "content": content},
            )
        content = request.arguments.get("content")
        assert isinstance(content, str)
        if path == self.fail_write_path or (
            self.rollback_fail and "rollback" in request.call_id
        ):
            return ToolResult(
                call_id=request.call_id,
                success=False,
                error=ToolErrorInfo(code="mcp_tool_error", message="write failed"),
            )
        self.contents[path] = content
        return ToolResult(call_id=request.call_id, success=True, data={"path": path})


def transaction(fake: FakeInvoker, tmp_path: Path, limit: int = 100) -> EditTransaction:
    return EditTransaction(
        fake,
        path_policy=WorkspacePathPolicy(tmp_path),
        max_write_bytes=limit,
    )


def test_transaction_happy_path_and_deterministic_order(tmp_path: Path) -> None:
    async def scenario() -> None:
        fake = FakeInvoker({"a.py": "A\n", "b.py": "B\n"})
        result = await transaction(fake, tmp_path).execute(
            [
                FileEditPlan("a.py", (replacement("A", "AA"),)),
                FileEditPlan("b.py", (replacement("B", "BB"),)),
            ],
            [FileSnapshot("a.py", "A\n"), FileSnapshot("b.py", "B\n")],
        )
        assert result.success
        assert [call.capability for call in fake.calls] == [
            "filesystem.read",
            "filesystem.read",
            "filesystem.write",
            "filesystem.read",
            "filesystem.write",
            "filesystem.read",
        ]
        assert [change.path for change in result.file_changes] == ["a.py", "b.py"]
        assert result.rollback_performed is False

    asyncio.run(scenario())


def test_stale_preflight_writes_zero_files(tmp_path: Path) -> None:
    async def scenario() -> None:
        fake = FakeInvoker({"a.py": "v2"})
        result = await transaction(fake, tmp_path).execute(
            [FileEditPlan("a.py", (replacement("v1", "v3"),))],
            [FileSnapshot("a.py", "v1")],
        )
        assert not result.success
        assert isinstance(result.error, Exception)
        assert not [
            call for call in fake.calls if call.capability == "filesystem.write"
        ]

    asyncio.run(scenario())


def test_second_write_failure_rolls_back_in_reverse_order(tmp_path: Path) -> None:
    async def scenario() -> None:
        fake = FakeInvoker({"a.py": "A\n", "b.py": "B\n"})
        fake.fail_write_path = "b.py"
        result = await transaction(fake, tmp_path).execute(
            [
                FileEditPlan("a.py", (replacement("A", "AA"),)),
                FileEditPlan("b.py", (replacement("B", "BB"),)),
            ],
            [FileSnapshot("a.py", "A\n"), FileSnapshot("b.py", "B\n")],
        )
        assert not result.success
        assert result.rollback_complete is True
        assert fake.contents["a.py"] == "A\n"
        writes = [call for call in fake.calls if call.capability == "filesystem.write"]
        assert [call.arguments["path"] for call in writes] == ["a.py", "b.py", "a.py"]

    asyncio.run(scenario())


def test_rollback_conflict_is_not_overwritten(tmp_path: Path) -> None:
    async def scenario() -> None:
        fake = FakeInvoker({"a.py": "A\n", "b.py": "B\n"})
        fake.fail_write_path = "b.py"
        fake.mismatch_path = "a.py"
        result = await transaction(fake, tmp_path).execute(
            [
                FileEditPlan("a.py", (replacement("A", "AA"),)),
                FileEditPlan("b.py", (replacement("B", "BB"),)),
            ],
            [FileSnapshot("a.py", "A\n"), FileSnapshot("b.py", "B\n")],
        )
        assert result.rollback_complete is False
        assert result.rollback_conflicts[0].path == "a.py"
        assert fake.contents["a.py"] == "external"
        assert (
            len([call for call in fake.calls if call.capability == "filesystem.write"])
            == 2
        )

    asyncio.run(scenario())


def test_policy_violation_propagates_before_write(tmp_path: Path) -> None:
    async def scenario() -> None:
        fake = FakeInvoker({"../outside.py": "A"})
        with pytest.raises(PolicyViolationError):
            await transaction(fake, tmp_path).execute(
                [FileEditPlan("../outside.py", (replacement("A", "B"),))],
                [FileSnapshot("../outside.py", "A")],
            )
        assert not fake.calls

    asyncio.run(scenario())


def test_final_size_bound_rejects_before_any_external_call(tmp_path: Path) -> None:
    async def scenario() -> None:
        fake = FakeInvoker({"main.py": "a"})
        with pytest.raises(InvalidRequestError):
            await transaction(fake, tmp_path, limit=1).execute(
                [FileEditPlan("main.py", (replacement("a", "é"),))],
                [FileSnapshot("main.py", "a")],
            )
        assert not fake.calls

    asyncio.run(scenario())


def test_rollback_write_failure_preserves_original_and_rollback_errors(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        fake = FakeInvoker({"a.py": "A\n", "b.py": "B\n"})
        fake.fail_write_path = "b.py"
        fake.rollback_fail = True
        result = await transaction(fake, tmp_path).execute(
            [
                FileEditPlan("a.py", (replacement("A", "AA"),)),
                FileEditPlan("b.py", (replacement("B", "BB"),)),
            ],
            [FileSnapshot("a.py", "A\n"), FileSnapshot("b.py", "B\n")],
        )
        assert not result.success
        assert isinstance(result.error, ToolResult)
        assert result.error.error is not None
        assert result.error.error.message == "write failed"
        assert result.rollback_complete is False
        assert result.rollback_errors

    asyncio.run(scenario())


def test_post_write_hash_mismatch_triggers_rollback(tmp_path: Path) -> None:
    async def scenario() -> None:
        fake = FakeInvoker({"a.py": "A\n"})
        fake.post_write_mismatch_path = "a.py"
        result = await transaction(fake, tmp_path).execute(
            [FileEditPlan("a.py", (replacement("A", "AA"),))],
            [FileSnapshot("a.py", "A\n")],
        )
        assert not result.success
        assert isinstance(result.error, OperationConflictError)
        assert result.rollback_performed
        assert result.rollback_complete is False

    asyncio.run(scenario())


async def _unexpected(request: ToolRequest) -> ToolResult:
    raise AssertionError(f"unexpected call: {request}")
