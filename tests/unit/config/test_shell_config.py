from pathlib import Path

import pytest
from pydantic import ValidationError

from coding_agent.config import ShellMcpSettings


def test_shell_settings_require_command_timeout_to_fit_mcp_timeout(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValidationError):
        ShellMcpSettings(
            workspace_root=tmp_path,
            operation_timeout_seconds=5,
            default_timeout_seconds=2,
            max_timeout_seconds=6,
        )

    with pytest.raises(ValidationError):
        ShellMcpSettings(
            workspace_root=tmp_path,
            operation_timeout_seconds=10,
            default_timeout_seconds=6,
            max_timeout_seconds=5,
        )
