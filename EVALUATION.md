# Evaluation

The evaluation proves the existing agent architecture against the small
`examples/task_app` target. It uses deterministic fake models and real local
MCP servers; it never needs `OPENAI_API_KEY`.

Run the focused suite with:

```bash
uv run pytest tests/evaluation -v
```

Run the complete repository with `uv run pytest` and the real MCP checks with
`uv run pytest -m integration`.

| Scenario | Main invariant |
| --- | --- |
| Explore and inventory | bounded reads; no shell or writes |
| Run pass/failure | canonical pytest only; observation-only |
| Edit first pass | transactional write and passing verification |
| Edit + repair | one bounded repair; final diff is original-to-final |
| Undo | hash-safe restore with no model or shell call |
| Memory and dry-run | preference context persists; dry-run never mutates |
| Routing and budgets | deterministic routes and bounded exhaustion |
| Security | unsafe paths, commands, prompts, and stale state stay constrained |

The manual CLI demo can use the same target:

```bash
uv run coding-agent chat --workspace examples/task_app
```

Try `How are tasks created?`, `Run the tests`, `Add validation to reject empty
task titles`, and `Undo that`. A non-mutating preview is available with:

```bash
uv run coding-agent run "Add validation to reject empty task titles" \
  --workspace examples/task_app --dry-run
```
