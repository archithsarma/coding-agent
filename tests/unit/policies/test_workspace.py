from pathlib import Path

import pytest

from coding_agent.domain import PolicyViolationError, ToolRequest
from coding_agent.policies import WorkspacePathPolicy
from coding_agent.tools import ToolDescriptor


def filesystem_descriptor() -> ToolDescriptor:
    return ToolDescriptor(
        tool_name="filesystem-read",
        capability="filesystem.read",
        description="Read text files.",
        mutating=False,
    )


def shell_descriptor() -> ToolDescriptor:
    return ToolDescriptor(
        tool_name="shell-execute",
        capability="shell.execute",
        description="Execute approved development commands.",
        mutating=True,
    )


def test_workspace_policy_accepts_nested_relative_paths(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('ok')", encoding="utf-8")
    policy = WorkspacePathPolicy(tmp_path)

    policy.validate_request(
        filesystem_descriptor(),
        ToolRequest(
            call_id="1", capability="filesystem.read", arguments={"path": "src/main.py"}
        ),
    )


@pytest.mark.parametrize("path", ["../outside.txt", "../../etc/passwd", "/etc/passwd"])
def test_workspace_policy_rejects_path_escape(tmp_path: Path, path: str) -> None:
    policy = WorkspacePathPolicy(tmp_path)

    with pytest.raises(PolicyViolationError):
        policy.validate_request(
            filesystem_descriptor(),
            ToolRequest(
                call_id="1", capability="filesystem.read", arguments={"path": path}
            ),
        )


def test_workspace_policy_rejects_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-secret.txt"
    outside.write_text("secret", encoding="utf-8")
    link = tmp_path / "link"
    try:
        link.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")

    with pytest.raises(PolicyViolationError):
        WorkspacePathPolicy(tmp_path).validate_request(
            filesystem_descriptor(),
            ToolRequest(
                call_id="1",
                capability="filesystem.read",
                arguments={"path": "link"},
            ),
        )


def test_workspace_policy_validates_shell_cwd(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    policy = WorkspacePathPolicy(tmp_path)

    policy.validate_request(
        shell_descriptor(),
        ToolRequest(
            call_id="1", capability="shell.execute", arguments={"cwd": "tests"}
        ),
    )

    with pytest.raises(PolicyViolationError):
        policy.validate_request(
            shell_descriptor(),
            ToolRequest(
                call_id="2", capability="shell.execute", arguments={"cwd": "../outside"}
            ),
        )
