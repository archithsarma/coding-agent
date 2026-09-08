"""Application service for creating tasks."""

from task_app.models import Task
from task_app.validation import validate_title


def create_task(title: str) -> Task:
    """Create a task after applying the title validation rules."""

    return Task(title=validate_title(title))
