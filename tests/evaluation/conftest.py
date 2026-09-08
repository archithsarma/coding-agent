"""Shared real-MCP harness for the submission evaluation."""

from __future__ import annotations

import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from coding_agent.config import FilesystemMcpSettings, ShellMcpSettings
from coding_agent.domain import ExecutionBudget
from coding_agent.model import TextGenerationResult
from coding_agent.orchestration.context import OrchestrationContext
from coding_agent.orchestration.edit import (
    EditPlanOutput,
    EditSelection,
    RepairPlanOutput,
)
from coding_agent.orchestration.explore import FileSelection
from coding_agent.orchestration.graph import build_graph
from coding_agent.policies import PolicyChain, ShellCommandPolicy, WorkspacePathPolicy
from coding_agent.tools import ToolRegistry, ToolRuntime
from coding_agent.tools.mcp import FilesystemMcpAdapter, ShellMcpAdapter, StdioMcpClient

DEMO_ROOT = Path(__file__).parents[2] / "examples" / "task_app"


class EvaluationModel:
    """Deterministic model whose outputs are explicit per test scenario."""

    def __init__(self, *, repair: bool = False, malicious_path: str | None = None):
        self.repair = repair
        self.malicious_path = malicious_path
        self.structured_calls = 0
        self.text_calls = 0
        self.inputs: list[str] = []
        self.instructions: list[str] = []

    async def generate_structured(self, *, output_type, input, **_kwargs):
        self.structured_calls += 1
        self.inputs.append(input)
        self.instructions.append(_kwargs.get("instructions", ""))
        if output_type is FileSelection:
            return output_type(paths=["src/task_app/validation.py"])
        if output_type is EditSelection:
            return output_type(
                paths=[self.malicious_path or "src/task_app/validation.py"]
            )
        if output_type is RepairPlanOutput:
            if not self.repair:
                return output_type(can_repair=False, summary="no repair")
            return output_type(
                can_repair=True,
                summary="restore normalization",
                files=[
                    {
                        "path": "src/task_app/validation.py",
                        "replacements": [
                            {
                                "old_text": "    return title\n",
                                "new_text": "    return normalized\n",
                                "expected_occurrences": 1,
                            }
                        ],
                    }
                ],
            )
        if self.repair:
            return EditPlanOutput(
                files=[
                    {
                        "path": "src/task_app/validation.py",
                        "replacements": [
                            {
                                "old_text": "    return normalized\n",
                                "new_text": "    return title\n",
                                "expected_occurrences": 1,
                            }
                        ],
                    }
                ]
            )
        return EditPlanOutput(
            files=[
                {
                    "path": "src/task_app/validation.py",
                    "replacements": [
                        {
                            "old_text": "    return normalized\n",
                            "new_text": (
                                "    if len(normalized) > 100:\n"
                                "        raise ValueError(\n"
                                "            'task title must be at most 100 "
                                "characters'\n"
                                "        )\n"
                                "    return normalized\n"
                            ),
                            "expected_occurrences": 1,
                        }
                    ],
                }
            ]
        )

    async def generate_text(self, *, input, **_kwargs) -> TextGenerationResult:
        self.text_calls += 1
        self.inputs.append(input)
        self.instructions.append(_kwargs.get("instructions", ""))
        return TextGenerationResult(
            text="Tasks are created by task_app.service.create_task.", model="fake"
        )


@pytest.fixture
def demo_workspace(tmp_path: Path) -> Path:
    target = tmp_path / "task_app"
    shutil.copytree(DEMO_ROOT, target)
    return target


@asynccontextmanager
async def mcp_runtime(root: Path, model: object) -> AsyncIterator[OrchestrationContext]:
    """Start the official filesystem and local shell MCP servers for a test."""

    workspace_policy = WorkspacePathPolicy(root.resolve())
    filesystem_settings = FilesystemMcpSettings(
        workspace_root=root, operation_timeout_seconds=30
    )
    shell_settings = ShellMcpSettings(
        workspace_root=root,
        operation_timeout_seconds=130,
        default_timeout_seconds=30,
        max_timeout_seconds=120,
    )
    filesystem = FilesystemMcpAdapter(
        StdioMcpClient(filesystem_settings, workspace_root=root.resolve()),
        path_policy=workspace_policy,
        max_read_bytes=filesystem_settings.max_read_bytes,
        max_write_bytes=filesystem_settings.max_write_bytes,
        operation_timeout_seconds=filesystem_settings.operation_timeout_seconds,
    )
    shell = ShellMcpAdapter(
        StdioMcpClient(shell_settings, workspace_root=root.resolve()),
        path_policy=workspace_policy,
        settings=shell_settings,
    )
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
        yield OrchestrationContext(
            model=model, tools=runtime, path_policy=workspace_policy
        )


async def invoke(
    context: OrchestrationContext,
    request: str,
    budget: ExecutionBudget | None = None,
    *,
    dry_run: bool = False,
) -> dict[str, object]:
    return await build_graph(budget, dry_run=dry_run).ainvoke(
        {"task_id": "evaluation", "user_request": request}, context=context
    )
