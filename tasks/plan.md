# Implementation Plan: Phase 11 Tiered Agent Memory

## Overview

Add bounded session memory and explicit, SQLite-backed coding preferences while
keeping LangGraph state as working memory and the existing OperationJournal as
the specialized undo snapshot store.

## Architecture Decisions

- Use small `PreferenceStore` and `SessionMemory` protocols with SQLite and
  in-memory implementations for deterministic tests.
- Store only explicit, bounded preference text; session events contain compact
  metadata and are isolated by session ID.
- Inject stores through `OrchestrationContext`; never put connections or store
  objects in graph state.
- Retrieve relevant preferences only for Edit planning and repair. Current
  requests remain higher precedence through explicit prompt wording.

## Task List

### Foundation

- [ ] Define preference/session records, bounded deterministic capture, and
  store protocols.
- [ ] Implement SQLite preference persistence and bounded session memory.
- [ ] Inject stores and session identity through orchestration context.

### Integration

- [ ] Add preference context to Edit planning and repair prompts.
- [ ] Record terminal Explore, Run, Edit, and Correction session events.
- [ ] Preserve journal-only undo semantics and add cross-session tests.

### Documentation and verification

- [ ] Document memory tiers, context growth, isolation, and failure semantics.
- [ ] Run unit/integration tests, lint, formatting, mypy, and diff checks.

## Risks and Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Stale or excessive context | High | Bounded retrieval and deterministic event compaction |
| Preference text affecting authority | High | Prompt-only style context; existing policies remain authoritative |
| Persistence failure blocking edits | Medium | Explicit remember fails clearly; retrieval fails open |
| Undo regression | High | Keep OperationJournal unchanged and retain two-turn integration coverage |
