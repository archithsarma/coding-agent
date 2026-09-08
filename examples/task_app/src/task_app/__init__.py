"""Small task service used by the coding-agent evaluation."""

from task_app.models import Task
from task_app.service import create_task

__all__ = ["Task", "create_task"]
