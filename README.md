
# Coding Agent

A controlled engineering coding agent orchestrated with explicit, testable
workflow boundaries.

## Prerequisites

- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync
```

## Basic commands

```bash
uv run coding-agent health
uv run coding-agent version
```

## Checks

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```
