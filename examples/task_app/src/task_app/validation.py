"""Validation rules for task input."""

MAX_TITLE_LENGTH = 100


def validate_title(title: str) -> str:
    """Normalize and validate a task title."""

    normalized = title.strip()
    if not normalized:
        raise ValueError("task title must not be empty")
    if len(normalized) > MAX_TITLE_LENGTH:
        raise ValueError("task title must be at most 100 characters")
    return normalized
