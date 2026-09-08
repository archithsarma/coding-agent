# Task app evaluation target

This is a deliberately small Task service used as the coding-agent's
submission and evaluation target. It normalizes task titles and rejects empty
or overlong titles.

Run its checks from this directory with:

```bash
uv run pytest
uv run ruff check .
uv run mypy src
```
