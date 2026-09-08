
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
isolated to that workspace. Internal transactional filesystem writes are
workspace-constrained, expected-hash checked, and conflict-checked; a
user-facing Edit trajectory is not implemented yet. The `shell.execute` capability launches the local Shell MCP server
and permits only structured argv commands for `pytest`, `ruff`, `mypy`, and
read-only `git status`/`git diff` forms. It never invokes a shell parser;
working directories stay inside the workspace, and command timeouts and
stdout/stderr limits are enforced. Shell execution is marked mutating because
development tools may create caches or other files.

Explore requests use a bounded LangGraph trajectory: inventory is breadth-first,
generated directories are ignored, selected paths must come from the inventory,
and file content is read through the filesystem MCP boundary before the model
receives it. Repository paths and contents are treated as untrusted evidence.

Run requests support deterministic, approved verification commands for `pytest`,
Ruff, and mypy through the secure Shell MCP boundary. Run does not select
commands with a model or perform automatic fixes.

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
