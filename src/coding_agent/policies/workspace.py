"""Workspace-relative path validation with symlink-aware containment checks."""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING

from coding_agent.domain import InvalidRequestError, PolicyViolationError, ToolRequest

if TYPE_CHECKING:
    from coding_agent.tools.descriptor import ToolDescriptor


class WorkspacePathResolver:
    def __init__(self, workspace_root: Path) -> None:
        try:
            root = workspace_root.expanduser().resolve(strict=True)
        except OSError as error:
            raise ValueError("workspace root must exist") from error
        if not root.is_dir():
            raise ValueError("workspace root must be a directory")
        self.root = root

    def resolve(self, relative_path: str) -> Path:
        normalized = relative_path.replace("\\", "/")
        if not normalized.strip() or "\x00" in normalized:
            raise PolicyViolationError(
                "workspace path must not be blank or contain null bytes"
            )
        posix_path = PurePosixPath(normalized)
        windows_path = PureWindowsPath(normalized)
        if posix_path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
            raise PolicyViolationError("workspace paths must be relative")
        if ".." in posix_path.parts:
            raise PolicyViolationError(
                "workspace path must not traverse parent directories"
            )

        candidate = self.root / posix_path
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(self.root)
        except ValueError as error:
            raise PolicyViolationError(
                "workspace path resolves outside the workspace"
            ) from error
        return resolved


class WorkspacePathPolicy:
    def __init__(self, workspace_root: Path | WorkspacePathResolver) -> None:
        self.resolver = (
            workspace_root
            if isinstance(workspace_root, WorkspacePathResolver)
            else WorkspacePathResolver(workspace_root)
        )

    def resolve_path(self, relative_path: str) -> Path:
        return self.resolver.resolve(relative_path)

    def validate_request(
        self, descriptor: ToolDescriptor, request: ToolRequest
    ) -> None:
        if descriptor.capability not in {"filesystem.list", "filesystem.read"}:
            return
        raw_path = request.arguments.get("path")
        if not isinstance(raw_path, str):
            raise InvalidRequestError("filesystem requests require a string path")
        self.resolve_path(raw_path)

    async def validate(self, descriptor: ToolDescriptor, request: ToolRequest) -> None:
        self.validate_request(descriptor, request)
