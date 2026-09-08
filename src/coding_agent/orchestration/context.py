"""Non-serializable runtime dependencies for orchestration nodes."""

from __future__ import annotations

from dataclasses import dataclass, field

from coding_agent.journal import InMemoryOperationJournal
from coding_agent.model import ModelClient
from coding_agent.orchestration.edit_config import EditConfig
from coding_agent.orchestration.explore_config import ExploreConfig
from coding_agent.policies import WorkspacePathPolicy
from coding_agent.tools import ToolRuntime


@dataclass(frozen=True)
class OrchestrationContext:
    model: ModelClient
    tools: ToolRuntime
    explore_config: ExploreConfig = ExploreConfig()
    edit_config: EditConfig = EditConfig()
    path_policy: WorkspacePathPolicy | None = None
    journal: InMemoryOperationJournal = field(default_factory=InMemoryOperationJournal)
