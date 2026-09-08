import asyncio
import shutil
from pathlib import Path

import pytest

from coding_agent.config import FilesystemMcpSettings
from coding_agent.domain import ToolRequest
from coding_agent.policies import PolicyChain, WorkspacePathPolicy
from coding_agent.tools import ToolRegistry, ToolRuntime
from coding_agent.tools.mcp import FilesystemMcpAdapter, StdioMcpClient

pytestmark = pytest.mark.integration


def test_official_filesystem_server_lists_and_reads(tmp_path: Path) -> None:
    if shutil.which("npx") is None:
        pytest.skip(
            "npx is unavailable; install Node.js to run the MCP integration test"
        )

    (tmp_path / "hello.txt").write_text("hello from mcp", encoding="utf-8")
    settings = FilesystemMcpSettings(
        workspace_root=tmp_path,
        operation_timeout_seconds=30,
    )
    workspace_root = settings.resolved_workspace_root()

    async def scenario() -> tuple[dict[str, object], dict[str, object]]:
        adapter = FilesystemMcpAdapter(
            StdioMcpClient(settings, workspace_root=workspace_root),
            path_policy=WorkspacePathPolicy(workspace_root),
            max_read_bytes=settings.max_read_bytes,
            operation_timeout_seconds=settings.operation_timeout_seconds,
        )
        async with adapter:
            registry = ToolRegistry()
            adapter.register_tools(registry)
            runtime = ToolRuntime(
                registry,
                policies=PolicyChain([WorkspacePathPolicy(workspace_root)]),
            )
            listing = await runtime.invoke(
                ToolRequest(
                    call_id="list",
                    capability="filesystem.list",
                    arguments={"path": "."},
                )
            )
            reading = await runtime.invoke(
                ToolRequest(
                    call_id="read",
                    capability="filesystem.read",
                    arguments={"path": "hello.txt"},
                )
            )
            assert listing.success and isinstance(listing.data, dict)
            assert reading.success and isinstance(reading.data, dict)
            return listing.data, reading.data

    listing, reading = asyncio.run(scenario())
    assert {entry["name"] for entry in listing["entries"]} == {"hello.txt"}
    assert reading == {"path": "hello.txt", "content": "hello from mcp"}
