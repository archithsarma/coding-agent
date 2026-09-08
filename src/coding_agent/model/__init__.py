"""Provider-neutral model contracts and the OpenAI implementation."""

from coding_agent.model.errors import (
    ModelError,
    ModelNotConfiguredError,
    ModelProviderError,
    ModelStructuredOutputError,
    ModelTimeoutError,
)
from coding_agent.model.openai import OpenAIModelClient
from coding_agent.model.protocol import (
    ModelClient,
    ModelUsage,
    TextGenerationResult,
)
from coding_agent.model.settings import OpenAIModelConfig

__all__ = [
    "ModelClient",
    "ModelError",
    "ModelNotConfiguredError",
    "ModelProviderError",
    "ModelStructuredOutputError",
    "ModelTimeoutError",
    "ModelUsage",
    "OpenAIModelClient",
    "OpenAIModelConfig",
    "TextGenerationResult",
]
