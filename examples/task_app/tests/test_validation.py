from task_app.validation import validate_title


def test_validate_title_preserves_internal_whitespace() -> None:
    assert validate_title("  write  tests ") == "write  tests"
