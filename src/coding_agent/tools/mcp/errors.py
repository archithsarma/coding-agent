"""Expected failures at the MCP integration boundary."""


class McpIntegrationError(Exception):
    """Base class for expected MCP lifecycle and response failures."""


class McpConnectionError(McpIntegrationError):
    """The MCP process or protocol connection could not be established."""


class McpToolUnavailableError(McpIntegrationError):
    """A required external tool was not advertised by the server."""


class McpResponseError(McpIntegrationError):
    """The external response could not be normalized."""


class McpTimeoutError(McpIntegrationError):
    """An MCP operation exceeded its configured timeout."""
