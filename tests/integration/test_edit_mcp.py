import asyncio
import json
import shutil
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
from coding_agent.orchestration.graph import build_graph
from coding_agent.policies import PolicyChain, WorkspacePathPolicy
from coding_agent.tools import ToolRegistry, ToolRuntime
from coding_agent.tools.mcp import (
    FilesystemMcpAdapter,
    ShellMcpAdapter,
    StdioMcpClient,
)

pytestmark = pytest.mark.integration


class EditIntegrationModel:
    def __init__(self) -> None:
        self.structured_calls = 0
        self.repair_inputs: list[str] = []

    async def generate_structured(self, *, output_type, **kwargs):
        self.structured_calls += 1
        if output_type is EditSelection:
            return output_type(paths=["routes/tasks.py"])
        if output_type is RepairPlanOutput:
            self.repair_inputs.append(kwargs["input"])
            return output_type(
                can_repair=True,
                summary="restore title trimming",
                files=[
                    {
                        "path": "routes/tasks.py",
                        "replacements": [
                            {
                                "old_text": "    return title\n",
                                "new_text": "    return title.strip()\n",
                                "expected_occurrences": 1,
                            }
                        ],
                    }
                ],
            )
        return EditPlanOutput(
            files=[
                {
                    "path": "routes/tasks.py",
                    "replacements": [
                        {
                            "old_text": "    return title.strip()\n",
                            "new_text": (
                                "    if not title.strip():\n"
                                "        raise ValueError('title')\n"
                                "    return title\n"
                            ),
                            "expected_occurrences": 1,
                        }
                    ],
                }
            ]
        )

    async def generate_text(self, **kwargs) -> TextGenerationResult:
        raise AssertionError("Edit must not call text generation")


def test_real_edit_trajectory_uses_mcp_transaction(tmp_path: Path) -> None:
    if shutil.which("npx") is None:
        pytest.skip("npx is unavailable; install Node.js to run the MCP test")

    routes = tmp_path / "routes"
    routes.mkdir()
    source = "def task(title):\n    return title.strip()\n"
    (routes / "tasks.py").write_text(source, encoding="utf-8")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_tasks.py").write_text(
        "from routes.tasks import task\n\n\n"
        "def test_title_is_trimmed():\n"
        "    assert task(' Task ') == 'Task'\n",
        encoding="utf-8",
    )
    (routes / "__init__.py").write_text("", encoding="utf-8")
    src = tmp_path / "src"
    src.mkdir()
    (src / "__init__.py").write_text("", encoding="utf-8")
    (tests / "conftest.py").write_text(
        "import sys\nfrom pathlib import Path\n"
        "\n"
        "sys.path.insert(0, str(Path(__file__).parents[1]))\n",
        encoding="utf-8",
    )
    settings = FilesystemMcpSettings(
        workspace_root=tmp_path, operation_timeout_seconds=30
    )
    workspace_root = settings.resolved_workspace_root()
    model = EditIntegrationModel()

    async def scenario() -> dict[str, object]:
        adapter = FilesystemMcpAdapter(
            StdioMcpClient(settings, workspace_root=workspace_root),
            path_policy=WorkspacePathPolicy(workspace_root),
            max_read_bytes=settings.max_read_bytes,
            max_write_bytes=settings.max_write_bytes,
            operation_timeout_seconds=settings.operation_timeout_seconds,
        )
        shell_settings = ShellMcpSettings(
            workspace_root=tmp_path,
            operation_timeout_seconds=11,
            default_timeout_seconds=5,
            max_timeout_seconds=10,
        )
        shell_adapter = ShellMcpAdapter(
            StdioMcpClient(shell_settings, workspace_root=workspace_root),
            path_policy=WorkspacePathPolicy(workspace_root),
            settings=shell_settings,
        )
        async with adapter, shell_adapter:
            registry = ToolRegistry()
            adapter.register_tools(registry)
            shell_adapter.register_tools(registry)
            runtime = ToolRuntime(
                registry,
                policies=PolicyChain([WorkspacePathPolicy(workspace_root)]),
            )
            return await build_graph(
                ExecutionBudget(
                    max_llm_calls=4,
                    max_tool_calls=32,
                    max_repair_attempts=2,
                    max_shell_execution_seconds=0,
                )
            ).ainvoke(
                {"task_id": "edit-integration", "user_request": "add title validation"},
                context=OrchestrationContext(
                    model=model,
                    tools=runtime,
                    path_policy=WorkspacePathPolicy(workspace_root),
                ),
            )

    result = asyncio.run(scenario())
    updated = (routes / "tasks.py").read_text(encoding="utf-8")
    assert result["current_node"] == "edit_complete"
    assert result["failure"] is None
    assert updated == (
        "def task(title):\n"
        "    if not title.strip():\n"
        "        raise ValueError('title')\n"
        "    return title.strip()\n"
    )
    assert result["edit"]["file_changes"][0]["path"] == "routes/tasks.py"
    assert result["edit"]["file_changes"][0]["before_hash"]
    assert result["edit"]["file_changes"][0]["after_hash"]
    assert result["edit"]["file_changes"][0]["patch"].startswith(
        "--- a/routes/tasks.py"
    )
    assert result["edit"]["operation_record"]["status"] == "succeeded"
    assert len(result["edit"]["operation_record"]["verifications"]) == 4
    verifications = result["edit"]["operation_record"]["verifications"]
    assert verifications[0]["passed"] is False
    assert all(item["passed"] for item in verifications[1:])
    assert result["counters"] == {
        "llm_calls": 3,
        "tool_calls": 15,
        "repair_attempts": 1,
    }
    assert model.structured_calls == 3
    assert len(model.repair_inputs) == 1
    repair_payload = json.loads(model.repair_inputs[0])
    repair_source = repair_payload["current_files"][0]["content"]
    assert "raise ValueError('title')" in repair_source
    assert "    return title\n" in repair_source
    assert result["edit"]["source_files"] == []
    assert result["edit"]["snapshots"] == []
    assert result["edit"]["current_files"] == []
