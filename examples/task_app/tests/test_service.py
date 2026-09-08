import pytest

from task_app import create_task


def test_create_task_normalizes_title() -> None:
    assert create_task("  Buy milk  ").title == "Buy milk"


def test_create_task_rejects_empty_title() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        create_task("   ")


def test_create_task_rejects_overlong_title() -> None:
    with pytest.raises(ValueError, match="at most 100"):
        create_task("x" * 101)
