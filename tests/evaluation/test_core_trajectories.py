"""Observable end-to-end trajectory evaluations."""

from __future__ import annotations

import asyncio
from pathlib import Path

from coding_agent.domain import ExecutionBudget, Trajectory

from .conftest import EvaluationModel, invoke, mcp_runtime


def test_explore_and_inventory_are_bounded_and_read_only(demo_workspace: Path) -> None:
    async def scenario() -> tuple[
        dict[str, object], dict[str, object], EvaluationModel
    ]:
        model = EvaluationModel()
        async with mcp_runtime(demo_workspace, model) as context:
            explored = await invoke(context, "Show me how tasks are created")
            inventory = await invoke(context, "What files are in this project?")
        return explored, inventory, model

    explored, inventory, model = asyncio.run(scenario())
    assert explored["trajectory"] == Trajectory.EXPLORE.value
    assert explored["explore"]["answer"]
    assert explored["explore"]["file_contents"] == []
    assert inventory["trajectory"] == Trajectory.EXPLORE.value
    assert inventory["explore"]["file_contents"] == []
    assert inventory["counters"]["llm_calls"] == 0
    assert model.text_calls == 1


def test_run_pass_and_failure_are_observation_only(demo_workspace: Path) -> None:
    async def scenario() -> tuple[
        dict[str, object], dict[str, object], EvaluationModel
    ]:
        model = EvaluationModel()
        async with mcp_runtime(demo_workspace, model) as context:
            passed = await invoke(context, "run tests")
            (demo_workspace / "tests" / "test_failure.py").write_text(
                "def test_failure():\n    assert False\n", encoding="utf-8"
            )
            failed = await invoke(context, "run pytest")
        return passed, failed, model

    passed, failed, model = asyncio.run(scenario())
    assert passed["trajectory"] == Trajectory.RUN.value
    assert passed["run"]["verification"]["passed"] is True
    assert failed["run"]["verification"]["passed"] is False
    assert passed["counters"]["llm_calls"] == 0
    assert failed["counters"]["llm_calls"] == 0
    assert model.structured_calls == 0
    assert model.text_calls == 0


def test_edit_first_pass_uses_real_mcp_and_verifies(demo_workspace: Path) -> None:
    before = (demo_workspace / "src/task_app/validation.py").read_text()

    async def scenario() -> dict[str, object]:
        async with mcp_runtime(demo_workspace, EvaluationModel()) as context:
            return await invoke(context, "Add validation for long task titles")

    result = asyncio.run(scenario())
    after = (demo_workspace / "src/task_app/validation.py").read_text()
    assert result["current_node"] == "edit_complete"
    assert result["edit"]["operation_record"]["status"] == "succeeded"
    assert result["counters"]["repair_attempts"] == 0
    assert after != before
    assert "at most 100" in after
    assert all(
        item["passed"] for item in result["edit"]["operation_record"]["verifications"]
    )


def test_edit_repair_is_single_bounded_and_keeps_final_diff(
    demo_workspace: Path,
) -> None:
    validation = demo_workspace / "src/task_app/validation.py"
    original = validation.read_text()

    async def scenario() -> dict[str, object]:
        model = EvaluationModel(repair=True)
        async with mcp_runtime(demo_workspace, model) as context:
            return await invoke(
                context,
                "Add validation while preserving normalization",
                ExecutionBudget(
                    max_llm_calls=4,
                    max_tool_calls=64,
                    max_repair_attempts=1,
                    max_shell_execution_seconds=0,
                ),
            )

    result = asyncio.run(scenario())
    repaired = validation.read_text()
    assert result["current_node"] == "edit_complete"
    assert result["counters"]["repair_attempts"] == 1
    assert result["edit"]["operation_record"]["status"] == "succeeded"
    assert repaired == original
    assert len(result["edit"]["operation_record"]["verifications"]) == 4
    assert result["edit"]["source_files"] == []
    assert result["edit"]["snapshots"] == []
    assert result["edit"]["current_files"] == []


def test_undo_is_hash_safe_and_model_free(demo_workspace: Path) -> None:
    async def scenario() -> tuple[
        dict[str, object], dict[str, object], EvaluationModel
    ]:
        model = EvaluationModel()
        async with mcp_runtime(demo_workspace, model) as context:
            edited = await invoke(context, "Add validation for long task titles")
            calls = model.structured_calls
            undone = await invoke(context, "Undo that")
            assert model.structured_calls == calls
        return edited, undone, model

    edited, undone, _ = asyncio.run(scenario())
    assert edited["edit"]["operation_record"]["status"] == "succeeded"
    assert undone["trajectory"] == Trajectory.CORRECTION.value
    assert undone["correction"]["operation_record"]["status"] == "succeeded"
    assert (demo_workspace / "src/task_app/validation.py").read_text() == (
        (
            Path(__file__).parents[2] / "examples/task_app/src/task_app/validation.py"
        ).read_text()
    )
