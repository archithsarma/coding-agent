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

| Scenario | Expected behavior | Safety property |
| --- | --- | --- |
| Explore and inventory | bounded inventory, selected reads, grounded answer | no shell or write |
| Run pass/failure | canonical pytest and `VerificationResult` | observation-only |
| Edit first pass | exact transactional write and verification | existing files only |
| Edit + repair | one repair, restarted verification, final logical diff | original scope retained |
| Undo | hash-safe restore with no model call | human changes are preserved |
| Memory and dry-run | preference retrieval and candidate diff | no dry-run mutation |
| Routing and budgets | deterministic route and one fallback call | no unbounded loop |
| Security | traversal, injection, stale/conflicting state blocked | no authority expansion |

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
