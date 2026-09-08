
# Coding Agent

A controlled engineering coding agent orchestrated with explicit, testable
workflow boundaries.

## Prerequisites

- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/)
- Node.js and `npx` for the filesystem MCP integration test

## Setup

```bash
uv sync
```

## Basic commands

```bash
uv run coding-agent health
uv run coding-agent version
```

The filesystem integration launches `npx -y @modelcontextprotocol/server-filesystem
<workspace-root>` with a single configured workspace. Filesystem tools are
isolated to that workspace and only read/list capabilities are registered in
this phase. The `shell.execute` capability launches the local Shell MCP server
and permits only structured argv commands for `pytest`, `ruff`, `mypy`, and
read-only `git status`/`git diff` forms. It never invokes a shell parser;
working directories stay inside the workspace, and command timeouts and
stdout/stderr limits are enforced. Shell execution is marked mutating because
development tools may create caches or other files.

## Checks

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

Run the real MCP integration test explicitly (it may download the official
server package through `npx`):

```bash
uv run pytest -m integration
```

This runs both the filesystem and local Shell MCP integration tests; no network
download is required for the shell server.
