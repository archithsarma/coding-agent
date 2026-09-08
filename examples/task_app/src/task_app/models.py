"""Domain models for the example task service."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    """A task with a normalized title."""

    title: str
