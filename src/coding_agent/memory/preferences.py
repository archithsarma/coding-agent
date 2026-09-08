"""Explicit, bounded coding-preference persistence."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4

MAX_PREFERENCE_VALUE_CHARS = 500
MAX_PREFERENCES = 100
PREFERENCE_SOURCE = "explicit_user_instruction"


class PreferencePersistenceError(RuntimeError):
    """A preference could not be read or written safely."""


@dataclass(frozen=True)
class PreferenceCandidate:
    """A high-confidence preference extracted from an explicit request."""

    category: str
    value: str


@dataclass(frozen=True)
class PreferenceRecord:
    """A durable user preference with bounded text and provenance."""

    preference_id: str
    category: str
    value: str
    created_at: datetime
    updated_at: datetime
    source: str = PREFERENCE_SOURCE


class PreferenceStore(Protocol):
    """Small synchronous boundary for explicit durable preferences."""

    def remember_preference(self, category: str, value: str) -> PreferenceRecord: ...

    def list_preferences(self) -> list[PreferenceRecord]: ...

    def retrieve_preferences(self, categories: set[str]) -> list[PreferenceRecord]: ...


def capture_explicit_preference(request: str) -> PreferenceCandidate | None:
    """Capture only explicit remember/from-now-on instructions."""

    normalized = " ".join(request.strip().split())
    if not normalized:
        return None
    match = re.match(
        r"^(?:remember(?:\s+that)?|from\s+now\s+on)\s*[:,-]?\s*(.+)$",
        normalized,
        flags=re.IGNORECASE,
    )
    if match is None and re.search(r"\bremember(?:\s+that)?\.?$", normalized, re.I):
        match = re.match(r"^(.+?)\s+remember(?:\s+that)?\.?$", normalized, re.I)
    if match is None:
        return None
    value = match.group(1).strip().rstrip(".")
    if not value or len(value) > MAX_PREFERENCE_VALUE_CHARS:
        return None
    lowered = value.lower()
    if not any(
        marker in lowered
        for marker in (
            "always",
            "prefer",
            "don't",
            "do not",
            "never",
            "use ",
        )
    ):
        return None
    return PreferenceCandidate(_category_for(value), value + ".")


def _category_for(value: str) -> str:
    lowered = value.lower()
    if "docstring" in lowered or "documentation" in lowered:
        return "documentation"
    if "test" in lowered or "pytest" in lowered:
        return "testing"
    if "format" in lowered or "style" in lowered or "lint" in lowered:
        return "formatting"
    if "workflow" in lowered or "commit" in lowered:
        return "workflow"
    return "editing_style"


@dataclass
class InMemoryPreferenceStore:
    """Deterministic store used as the safe default for isolated runtimes."""

    _records: dict[str, PreferenceRecord] = field(default_factory=dict)

    def remember_preference(self, category: str, value: str) -> PreferenceRecord:
        _validate_preference(category, value)
        now = datetime.now(UTC)
        existing = self._records.get(category)
        record = PreferenceRecord(
            preference_id=existing.preference_id if existing else uuid4().hex,
            category=category,
            value=value,
            created_at=existing.created_at if existing else now,
            updated_at=now,
        )
        if existing is None and len(self._records) >= MAX_PREFERENCES:
            oldest = min(self._records, key=lambda key: self._records[key].updated_at)
            del self._records[oldest]
        self._records[category] = record
        return record

    def list_preferences(self) -> list[PreferenceRecord]:
        return sorted(self._records.values(), key=lambda item: item.updated_at)

    def retrieve_preferences(self, categories: set[str]) -> list[PreferenceRecord]:
        return [item for item in self.list_preferences() if item.category in categories]


@dataclass
class SQLitePreferenceStore:
    """SQLite-backed preference store with short-lived operation connections."""

    database_path: str | Path
    _owned_connection: sqlite3.Connection | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.database_path = Path(self.database_path)
        if str(self.database_path) == ":memory:":
            self._owned_connection = sqlite3.connect(":memory:", timeout=5)
            self._owned_connection.row_factory = sqlite3.Row
        else:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def close(self) -> None:
        """Close the special in-memory connection, if one is owned."""
        if self._owned_connection is not None:
            self._owned_connection.close()
            self._owned_connection = None

    def remember_preference(self, category: str, value: str) -> PreferenceRecord:
        _validate_preference(category, value)
        now = datetime.now(UTC)
        preference_id = uuid4().hex
        try:
            with self._connection() as connection:
                existing = connection.execute(
                    "SELECT preference_id, created_at FROM preferences "
                    "WHERE category = ?",
                    (category,),
                ).fetchone()
                if existing is not None:
                    preference_id = str(existing[0])
                    created_at = datetime.fromisoformat(str(existing[1]))
                else:
                    created_at = now
                connection.execute(
                    "INSERT INTO preferences "
                    "(preference_id, category, value, created_at, updated_at, source) "
                    "VALUES (?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(category) DO UPDATE SET value = excluded.value, "
                    "updated_at = excluded.updated_at, source = excluded.source",
                    (
                        preference_id,
                        category,
                        value,
                        created_at.isoformat(),
                        now.isoformat(),
                        PREFERENCE_SOURCE,
                    ),
                )
                connection.execute(
                    "DELETE FROM preferences WHERE preference_id IN ("
                    "SELECT preference_id FROM preferences "
                    "ORDER BY updated_at DESC LIMIT -1 OFFSET ?)",
                    (MAX_PREFERENCES,),
                )
        except (OSError, sqlite3.Error) as error:
            raise PreferencePersistenceError(
                "could not persist coding preference"
            ) from error
        return PreferenceRecord(
            preference_id=preference_id,
            category=category,
            value=value,
            created_at=created_at,
            updated_at=now,
        )

    def list_preferences(self) -> list[PreferenceRecord]:
        return self.retrieve_preferences(set())

    def retrieve_preferences(self, categories: set[str]) -> list[PreferenceRecord]:
        try:
            with self._connection() as connection:
                rows = connection.execute(
                    "SELECT preference_id, category, value, created_at, "
                    "updated_at, source FROM preferences ORDER BY updated_at ASC"
                ).fetchall()
        except (OSError, sqlite3.Error) as error:
            raise PreferencePersistenceError(
                "could not retrieve coding preferences"
            ) from error
        return [
            _record_from_row(row)
            for row in rows
            if not categories or str(row[1]) in categories
        ]

    def _connection(self) -> sqlite3.Connection:
        if self._owned_connection is not None:
            return self._owned_connection
        connection = sqlite3.connect(str(self.database_path), timeout=5)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        try:
            with self._connection() as connection:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS schema_version ("
                    "version INTEGER NOT NULL)"
                )
                version = connection.execute(
                    "SELECT version FROM schema_version LIMIT 1"
                ).fetchone()
                if version is None:
                    connection.execute("INSERT INTO schema_version(version) VALUES (1)")
                elif int(version[0]) != 1:
                    raise PreferencePersistenceError(
                        "unsupported memory schema version"
                    )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS preferences ("
                    "preference_id TEXT PRIMARY KEY, category TEXT NOT NULL UNIQUE, "
                    "value TEXT NOT NULL, created_at TEXT NOT NULL, "
                    "updated_at TEXT NOT NULL, source TEXT NOT NULL)"
                )
        except (OSError, sqlite3.Error) as error:
            raise PreferencePersistenceError(
                "could not initialize preference store"
            ) from error


def _validate_preference(category: str, value: str) -> None:
    if not category.strip() or len(category) > 64:
        raise ValueError("preference category must be 1-64 characters")
    if not value.strip() or len(value) > MAX_PREFERENCE_VALUE_CHARS:
        raise ValueError("preference value must be 1-500 characters")


def _record_from_row(row: sqlite3.Row) -> PreferenceRecord:
    return PreferenceRecord(
        preference_id=str(row[0]),
        category=str(row[1]),
        value=str(row[2]),
        created_at=datetime.fromisoformat(str(row[3])),
        updated_at=datetime.fromisoformat(str(row[4])),
        source=str(row[5]),
    )
