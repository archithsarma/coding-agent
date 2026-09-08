"""Configuration for the OpenAI model adapter."""

from __future__ import annotations

import os

from pydantic import Field, SecretStr, field_validator

from coding_agent.domain.models import DomainModel


class OpenAIModelConfig(DomainModel):
    model_name: str
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_output_tokens: int = Field(default=2048, gt=0)
    max_retries: int = Field(default=0, ge=0)
    api_key: SecretStr | None = None

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model_name must not be blank")
        return value

    @classmethod
    def from_environment(cls) -> OpenAIModelConfig:
        api_key = os.environ.get("OPENAI_API_KEY")
        return cls(
            model_name=os.environ.get("OPENAI_MODEL", "gpt-5.5"),
            api_key=SecretStr(api_key) if api_key else None,
        )
