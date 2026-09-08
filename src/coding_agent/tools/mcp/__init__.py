"""MCP-specific implementation details kept behind the internal tool boundary."""

from coding_agent.tools.mcp.client import (
    McpCallResult,
    McpConnection,
    McpToolDescription,
    StdioMcpClient,
)
from coding_agent.tools.mcp.filesystem import FilesystemMcpAdapter
from coding_agent.tools.mcp.shell import ShellMcpAdapter

__all__ = [
    "FilesystemMcpAdapter",
    "McpCallResult",
    "McpConnection",
    "McpToolDescription",
    "StdioMcpClient",
    "ShellMcpAdapter",
]
