
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
uv run coding-agent chat --workspace ./demo
uv run coding-agent run "run tests" --workspace ./demo
```

The CLI keeps one session runtime across interactive turns, including its
session memory and Undo journal. `run` is the same single-turn graph path used
by `chat`. Use `--dry-run` with either command to plan an Edit and show its
bounded diff without writing files or running verification. Use
`--trace-file ./agent-trace.jsonl` to write bounded JSONL execution events;
trace output should be kept outside the target workspace for normal use.

Set `OPENAI_API_KEY` (and optionally `OPENAI_MODEL`) for Explore explanations,
Edit planning, repair, and ambiguous-request routing. Deterministic Run and
Correction requests remain available without a configured model and model
requests fail clearly with `model_not_configured`.

The filesystem integration launches `npx -y @modelcontextprotocol/server-filesystem
<workspace-root>` with a single configured workspace. Filesystem tools are
isolated to that workspace. Internal transactional filesystem writes are
workspace-constrained, expected-hash checked, and conflict-checked; a
The `shell.execute` capability launches the local Shell MCP server
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

Edit requests locate and read bounded existing files, propose exact structured
text replacements, commit them through the transactional filesystem MCP
boundary, and then run the configured verification suite (`pytest`, `ruff
check .`, and `mypy src`) fail-fast. A failed check may trigger a bounded,
exact-replacement auto-repair within the originally selected files. The
maximum number of repair attempts is configurable through `ExecutionBudget`
and defaults to two.

Correction requests (`undo`, `undo that`, `revert that`, `revert the last
change`, or `that's wrong`) deterministically revert the latest safely
reversible Edit in the current session. Undo re-reads every target through the
filesystem MCP boundary and reports a conflict instead of overwriting later
workspace changes. It restores known content transactionally, does not run the
verification suite, and does not provide redo or persistent cross-session undo.

The agent has three memory tiers:

- Working memory is the current typed LangGraph state. Temporary source content
  is cleared as trajectories finish; no second working-memory database exists.
- Session memory is an in-memory, session-ID-isolated, bounded list of compact
  Explore/Run/Edit/Correction outcome events. It stores paths and metadata, not
  raw source, model prompts/responses, or complete command output. Oldest events
  are deterministically evicted when event or serialized-byte limits are hit.
- Persistent memory is an injected SQLite `PreferenceStore` for explicit,
  bounded coding preferences only. The database path is configurable and should
  be placed in application data, outside the target workspace. SQLite uses
  parameterized SQL, a schema version, transactions, and short-lived
  connections.

Preferences are captured only from high-confidence explicit wording such as
`Remember: always use type hints.` Ordinary feedback is not persisted. Edit
planning and repair retrieve only relevant style/documentation/formatting
preferences; the current request is always higher precedence, and preference
text cannot authorize tools, files, commands, or scope changes. Retrieval is
fail-open with a warning, while an explicit persistence failure is surfaced.

Context growth follows HOT/WARM/COLD boundaries: current request, graph state,
and current filesystem reads are hot; recent compact events and relevant
preferences are warm; old events are evicted and source is re-read when needed.
Current filesystem content remains the source of truth.

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
