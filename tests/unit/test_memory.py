from pathlib import Path

import pytest

from coding_agent.memory import (
    InMemorySessionMemory,
    PreferencePersistenceError,
    SessionEvent,
    SQLitePreferenceStore,
    capture_explicit_preference,
)


def test_explicit_preference_capture_rejects_ordinary_feedback() -> None:
    candidate = capture_explicit_preference(
        "Remember: always use type hints when editing functions."
    )

    assert candidate is not None
    assert candidate.category == "editing_style"
    assert candidate.value == "always use type hints when editing functions."
    assert capture_explicit_preference("The type hints look good.") is None


def test_preference_roundtrip_survives_new_store_instance(tmp_path: Path) -> None:
    database = tmp_path / "preferences.sqlite3"
    first = SQLitePreferenceStore(database)
    first.remember_preference(
        "documentation", "Always add docstrings when editing functions."
    )

    second = SQLitePreferenceStore(database)

    assert [item.value for item in second.retrieve_preferences({"documentation"})] == [
        "Always add docstrings when editing functions."
    ]


def test_in_memory_sqlite_store_supports_multiple_operations() -> None:
    store = SQLitePreferenceStore(":memory:")

    store.remember_preference("testing", "Prefer focused tests.")

    assert len(store.list_preferences()) == 1


def test_preference_write_failure_is_explicit(tmp_path: Path) -> None:
    store = SQLitePreferenceStore(tmp_path / "preferences.sqlite3")
    store.close()
    store.database_path = tmp_path / "missing" / "preferences.sqlite3"

    with pytest.raises(PreferencePersistenceError):
        store.remember_preference("testing", "Prefer focused tests.")


def test_session_events_are_isolated_and_bounded() -> None:
    memory = InMemorySessionMemory(max_events=2, max_serialized_bytes=1000)
    memory.record(
        SessionEvent(
            session_id="session-a",
            trajectory="edit",
            summary="changed routes/tasks.py",
            files=("routes/tasks.py",),
            outcome="succeeded",
        )
    )
    memory.record(
        SessionEvent(
            session_id="session-a",
            trajectory="run",
            summary="ran pytest",
            commands=("pytest",),
            outcome="succeeded",
        )
    )
    memory.record(
        SessionEvent(
            session_id="session-a",
            trajectory="explore",
            summary="inspected routes/tasks.py",
            files=("routes/tasks.py",),
            outcome="succeeded",
        )
    )

    events = memory.list_events("session-a")

    assert len(events) == 2
    assert events[0].summary == "ran pytest"
    assert events[1].summary == "inspected routes/tasks.py"
    assert memory.list_events("session-b") == []
    assert all("def " not in event.summary for event in events)


def test_session_retrieval_prefers_file_overlap() -> None:
    memory = InMemorySessionMemory()
    memory.record(
        SessionEvent(
            session_id="session-a",
            trajectory="edit",
            summary="changed unrelated.py",
            files=("unrelated.py",),
            outcome="succeeded",
        )
    )
    memory.record(
        SessionEvent(
            session_id="session-a",
            trajectory="edit",
            summary="changed target.py",
            files=("target.py",),
            outcome="succeeded",
        )
    )

    result = memory.retrieve("session-a", trajectory="edit", files={"target.py"})

    assert result[0].summary == "changed target.py"
