"""Production CLI for the controlled coding agent."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast
from uuid import uuid4

import typer

from coding_agent import __version__
from coding_agent.config import FilesystemMcpSettings, ShellMcpSettings
from coding_agent.domain import ExecutionBudget
from coding_agent.journal import InMemoryOperationJournal
from coding_agent.memory import SQLitePreferenceStore
from coding_agent.model import (
    ModelClient,
    ModelNotConfiguredError,
    OpenAIModelClient,
    OpenAIModelConfig,
)
from coding_agent.observability import (
    InMemoryTraceSink,
    JsonlTraceSink,
    TraceSink,
    TraceState,
    TracingModelClient,
)
from coding_agent.orchestration.context import OrchestrationContext
from coding_agent.orchestration.graph import build_graph
from coding_agent.policies import PolicyChain, ShellCommandPolicy, WorkspacePathPolicy
from coding_agent.tools import ToolRegistry, ToolRuntime
from coding_agent.tools.mcp import FilesystemMcpAdapter, ShellMcpAdapter, StdioMcpClient
from coding_agent.tools.mcp.errors import McpIntegrationError

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def health() -> None:
    """Report that the CLI is available."""
    typer.echo("ok")


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def run(
    request: str = typer.Argument(..., help="One natural-language coding request."),
    workspace: Path = typer.Option(Path("."), "--workspace", "-w"),  # noqa: B008
    dry_run: bool = typer.Option(False, "--dry-run"),  # noqa: B008
    trace_file: Path | None = typer.Option(None, "--trace-file"),  # noqa: B008
) -> None:
    """Execute one request and exit with a workflow-aware status code."""

    try:
        result = asyncio.run(
            _run_request(request, workspace, dry_run=dry_run, trace_file=trace_file)
        )
    except (ValueError, OSError, RuntimeError, McpIntegrationError) as error:
        typer.echo(f"Startup error: {error}", err=True)
        raise typer.Exit(2) from error
    typer.echo(render_result(result))
    if result.get("failure"):
        raise typer.Exit(1)


@app.command()
def chat(
    workspace: Path = typer.Option(Path("."), "--workspace", "-w"),  # noqa: B008
    dry_run: bool = typer.Option(False, "--dry-run"),  # noqa: B008
    trace_file: Path | None = typer.Option(None, "--trace-file"),  # noqa: B008
) -> None:
    """Start a simple multi-turn coding-agent session."""

    try:
        asyncio.run(_chat(workspace, dry_run=dry_run, trace_file=trace_file))
    except (ValueError, OSError, RuntimeError, McpIntegrationError) as error:
        typer.echo(f"Startup error: {error}", err=True)
        raise typer.Exit(2) from error


class _UnavailableModel:
    async def generate_text(self, **_kwargs: object) -> object:
        raise ModelNotConfiguredError(
            "model_not_configured: set OPENAI_API_KEY to use this trajectory"
        )

    async def generate_structured(self, **_kwargs: object) -> object:
        raise ModelNotConfiguredError(
            "model_not_configured: set OPENAI_API_KEY to use this trajectory"
        )


class _AgentSession:
    def __init__(
        self,
        context: OrchestrationContext,
        *,
        dry_run: bool,
        workspace: Path,
        model_configured: bool,
    ) -> None:
        self.context = context
        self.dry_run = dry_run
        self.workspace = workspace
        self.model_configured = model_configured

    async def execute(self, request: str) -> dict[str, object]:
        return await build_graph(
            ExecutionBudget(
                max_llm_calls=4,
                max_tool_calls=64,
                max_repair_attempts=2,
                max_shell_execution_seconds=0,
            ),
            dry_run=self.dry_run,
        ).ainvoke(
            {"task_id": f"cli-{uuid4().hex}", "user_request": request},
            context=self.context,
        )


@asynccontextmanager
async def _open_session(
    workspace: Path,
    *,
    trace_file: Path | None,
    dry_run: bool,
) -> AsyncIterator[_AgentSession]:
    filesystem_settings = FilesystemMcpSettings(workspace_root=workspace)
    root = filesystem_settings.resolved_workspace_root()
    workspace_policy = WorkspacePathPolicy(root)
    shell_settings = ShellMcpSettings(workspace_root=root)
    filesystem = FilesystemMcpAdapter(
        StdioMcpClient(filesystem_settings, workspace_root=root),
        path_policy=workspace_policy,
        max_read_bytes=filesystem_settings.max_read_bytes,
        max_write_bytes=filesystem_settings.max_write_bytes,
        operation_timeout_seconds=filesystem_settings.operation_timeout_seconds,
    )
    shell = ShellMcpAdapter(
        StdioMcpClient(shell_settings, workspace_root=root),
        path_policy=workspace_policy,
        settings=shell_settings,
    )
    trace_sink: TraceSink = (
        JsonlTraceSink(trace_file) if trace_file else InMemoryTraceSink()
    )
    session_id = uuid4().hex
    trace_state = TraceState(session_id=session_id, sink=trace_sink)
    async with filesystem, shell:
        registry = ToolRegistry()
        filesystem.register_tools(registry)
        shell.register_tools(registry)
        runtime = ToolRuntime(
            registry,
            policies=PolicyChain(
                [
                    workspace_policy,
                    ShellCommandPolicy(
                        max_timeout_seconds=shell_settings.max_timeout_seconds,
                        path_policy=workspace_policy,
                    ),
                ]
            ),
        )
        preference_dir = Path(
            os.environ.get(
                "CODING_AGENT_DATA_DIR",
                str(Path.home() / ".local" / "share" / "coding-agent"),
            )
        )
        preference_store = SQLitePreferenceStore(preference_dir / "preferences.sqlite3")
        config = OpenAIModelConfig.from_environment()
        raw_model = (
            OpenAIModelClient(config)
            if config.api_key is not None
            else _UnavailableModel()
        )
        model = TracingModelClient(cast(ModelClient, raw_model), trace_state)
        context = OrchestrationContext(
            model=model,
            tools=runtime,
            path_policy=workspace_policy,
            journal=InMemoryOperationJournal(),
            session_id=session_id,
            preference_store=preference_store,
            trace_sink=trace_sink,
            trace=trace_state,
        )
        try:
            yield _AgentSession(
                context,
                dry_run=dry_run,
                workspace=root,
                model_configured=config.api_key is not None,
            )
        finally:
            preference_store.close()
            close = getattr(raw_model, "aclose", None)
            if callable(close):
                await close()


async def _run_request(
    request: str, workspace: Path, *, dry_run: bool, trace_file: Path | None
) -> dict[str, object]:
    async with _open_session(
        workspace, dry_run=dry_run, trace_file=trace_file
    ) as session:
        return await session.execute(request)


async def _chat(workspace: Path, *, dry_run: bool, trace_file: Path | None) -> None:
    async with _open_session(
        workspace, dry_run=dry_run, trace_file=trace_file
    ) as session:
        typer.echo("Coding Agent")
        typer.echo(f"Workspace: {session.workspace}")
        typer.echo(
            f"Session: {session.context.session_id[:8]}  "
            f"Model configured: {session.model_configured}  "
            f"Dry-run: {session.dry_run}"
        )
        typer.echo("Type /help for commands; /exit to leave.")
        while True:
            try:
                request = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                typer.echo("\nSession closed.")
                return
            if request in {"/exit", "/quit"}:
                typer.echo("Session closed.")
                return
            if request == "/help":
                typer.echo(
                    "/help  show commands\n"
                    "/memory  show saved preferences\n"
                    "/exit  close session"
                )
                continue
            if request == "/memory":
                for preference in session.context.preference_store.list_preferences():
                    typer.echo(f"- {preference.value}")
                continue
            if not request:
                continue
            try:
                result = await session.execute(request)
                typer.echo(render_result(result))
            except (ValueError, OSError, RuntimeError, McpIntegrationError) as error:
                typer.echo(f"Request error: {error}", err=True)


def render_result(result: dict[str, object]) -> str:
    """Render bounded user-facing output without exposing graph state."""

    memory = result.get("memory")
    if isinstance(memory, dict) and memory.get("answer"):
        return str(memory["answer"])
    failure = result.get("failure")
    if isinstance(failure, dict):
        code = failure.get("code", "workflow_failed")
        message = failure.get("message", "workflow failed")
        return f"Failed [{code}]: {message}"
    for key in ("explore", "run", "edit", "correction"):
        value = result.get(key)
        if isinstance(value, dict):
            answer = value.get("answer")
            if isinstance(answer, str):
                if key == "edit":
                    changes = value.get("file_changes", [])
                    patches: list[str] = []
                    if isinstance(changes, list):
                        for item in changes:
                            if isinstance(item, dict):
                                patch = item.get("patch")
                                if isinstance(patch, str):
                                    patches.append(patch)
                    diff = "\n".join(patches)[:12_000]
                    if diff and diff not in answer:
                        return f"{answer}\n\n{diff}"
                return answer
    return "Request completed."
