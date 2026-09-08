"""Provider-neutral asynchronous model contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar

from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass(frozen=True)
class ModelUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True)
class TextGenerationResult:
    text: str
    model: str
    response_id: str | None = None
    usage: ModelUsage = ModelUsage()


class ModelClient(Protocol):
    async def generate_text(
        self,
        *,
        instructions: str,
        input: str,
        max_output_tokens: int | None = None,
    ) -> TextGenerationResult: ...

    async def generate_structured(
        self,
        *,
        instructions: str,
        input: str,
        output_type: type[ModelT],
        max_output_tokens: int | None = None,
    ) -> ModelT: ...
