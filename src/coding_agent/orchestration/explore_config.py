"""Bounds and ignored names for repository exploration."""

from dataclasses import dataclass, field

IGNORED_DIRECTORIES = frozenset(
    {
        ".git",
        ".venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "dist",
        "build",
    }
)


@dataclass(frozen=True)
class ExploreConfig:
    max_depth: int = 4
    max_inventory_entries: int = 256
    max_directories: int = 32
    max_selected_files: int = 3
    max_total_content_bytes: int = 64_000
    ignored_directories: frozenset[str] = field(
        default_factory=lambda: IGNORED_DIRECTORIES
    )
