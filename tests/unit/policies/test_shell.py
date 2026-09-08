import asyncio

import pytest

from coding_agent.domain import PolicyViolationError, ToolRequest
from coding_agent.policies import ShellCommandPolicy, WorkspacePathPolicy
from coding_agent.tools import ToolDescriptor


def descriptor() -> ToolDescriptor:
    return ToolDescriptor(
        tool_name="shell-execute",
        capability="shell.execute",
        description="Execute approved commands.",
        mutating=True,
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["pytest", "tests/unit"],
        ["ruff", "check", "."],
        ["ruff", "format", "--check", "."],
        ["mypy", "src"],
        ["git", "status"],
        ["git", "diff", "--check"],
    ],
)
def test_shell_policy_allows_approved_commands(argv: list[str]) -> None:
    asyncio.run(
        ShellCommandPolicy().validate(
            descriptor(),
            ToolRequest(
                call_id="1", capability="shell.execute", arguments={"argv": argv}
            ),
        )
    )


@pytest.mark.parametrize(
    "argv",
    [
        [],
        [""],
        ["bash", "-c", "echo unsafe"],
        ["sh"],
        ["python", "-c", "print('unsafe')"],
        ["unknown-tool"],
        ["/absolute/path/to/pytest"],
        ["git", "commit"],
        ["git", "reset"],
        ["git", "-C", ".", "status"],
        ["git", "--git-dir", ".git", "status"],
        ["git", "-c", "core.sshCommand=unsafe", "status"],
        ["git", "diff", "--output", "outside.txt"],
        ["git", "diff", "--no-index", "a", "b"],
        ["git", "status", "--porcelain"],
        ["ruff", "format"],
    ],
)
def test_shell_policy_rejects_unsafe_commands(argv: list[str]) -> None:
    with pytest.raises(PolicyViolationError):
        asyncio.run(
            ShellCommandPolicy().validate(
                descriptor(),
                ToolRequest(
                    call_id="1", capability="shell.execute", arguments={"argv": argv}
                ),
            )
        )


def test_shell_policy_rejects_nulls_and_excessive_timeout() -> None:
    policy = ShellCommandPolicy(max_timeout_seconds=5)
    for arguments in (
        {"argv": ["pytest", "bad\x00arg"]},
        {"argv": ["pytest"], "timeout_seconds": 6},
    ):
        with pytest.raises(PolicyViolationError):
            asyncio.run(
                policy.validate(
                    descriptor(),
                    ToolRequest(
                        call_id="1", capability="shell.execute", arguments=arguments
                    ),
                )
            )


def test_shell_policy_rejects_cwd_escape(tmp_path) -> None:
    policy = ShellCommandPolicy(path_policy=WorkspacePathPolicy(tmp_path))
    with pytest.raises(PolicyViolationError):
        asyncio.run(
            policy.validate(
                descriptor(),
                ToolRequest(
                    call_id="1",
                    capability="shell.execute",
                    arguments={"argv": ["pytest"], "cwd": "../outside"},
                ),
            )
        )


def test_shell_policy_rejects_symlink_cwd_escape(tmp_path) -> None:
    outside = tmp_path.parent / "shell-policy-outside"
    outside.mkdir()
    link = tmp_path / "link"
    link.symlink_to(outside, target_is_directory=True)
    policy = ShellCommandPolicy(path_policy=WorkspacePathPolicy(tmp_path))

    with pytest.raises(PolicyViolationError):
        asyncio.run(
            policy.validate(
                descriptor(),
                ToolRequest(
                    call_id="1",
                    capability="shell.execute",
                    arguments={"argv": ["pytest"], "cwd": "link"},
                ),
            )
        )


def test_shell_policy_allowlist_can_only_be_narrowed() -> None:
    policy = ShellCommandPolicy(allowed_executables=("pytest",))
    asyncio.run(
        policy.validate(
            descriptor(),
            ToolRequest(
                call_id="1", capability="shell.execute", arguments={"argv": ["pytest"]}
            ),
        )
    )
    with pytest.raises(PolicyViolationError):
        asyncio.run(
            policy.validate(
                descriptor(),
                ToolRequest(
                    call_id="2",
                    capability="shell.execute",
                    arguments={"argv": ["ruff", "check"]},
                ),
            )
        )
