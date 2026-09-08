"""Validated configuration for the read-only filesystem integration."""

from pathlib import Path

from pydantic import Field, field_validator

from coding_agent.domain.models import DomainModel


def _non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("value must not be blank")
    return value


class FilesystemMcpSettings(DomainModel):
    workspace_root: Path
    mcp_command: str = "npx"
    server_package: str = "@modelcontextprotocol/server-filesystem"
    server_args: tuple[str, ...] = ("-y",)
    operation_timeout_seconds: float = Field(default=30.0, gt=0)
    max_read_bytes: int = Field(default=1_048_576, gt=0)

    _validate_text = field_validator("mcp_command", "server_package")(_non_blank)

    def resolved_workspace_root(self) -> Path:
        try:
            root = self.workspace_root.expanduser().resolve(strict=True)
        except OSError as error:
            raise ValueError("workspace_root must exist") from error
        if not root.is_dir():
            raise ValueError("workspace_root must be a directory")
        return root

    def server_arguments(self, workspace_root: Path | None = None) -> list[str]:
        root = workspace_root or self.resolved_workspace_root()
        return [
            *self.server_args,
            self.server_package,
            str(root),
        ]
