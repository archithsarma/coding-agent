"""OpenAI Responses API implementation of the internal model contract."""

from __future__ import annotations

from typing import Any

from openai import (
    APIResponseValidationError,
    APITimeoutError,
    AsyncOpenAI,
    OpenAIError,
)

from coding_agent.model.errors import (
    ModelProviderError,
    ModelStructuredOutputError,
    ModelTimeoutError,
)
from coding_agent.model.protocol import (
    ModelT,
    ModelUsage,
    TextGenerationResult,
)
from coding_agent.model.settings import OpenAIModelConfig


class OpenAIModelClient:
    """Provider adapter with explicit ownership and no provider tools."""

    def __init__(
        self,
        config: OpenAIModelConfig,
        *,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self._config = config
        self._owns_client = client is None
        if client is not None:
            self._client = client
        else:
            kwargs: dict[str, Any] = {
                "timeout": config.timeout_seconds,
                "max_retries": config.max_retries,
            }
            if config.api_key is not None:
                kwargs["api_key"] = config.api_key.get_secret_value()
            self._client = AsyncOpenAI(**kwargs)

    async def generate_text(
        self,
        *,
        instructions: str,
        input: str,
        max_output_tokens: int | None = None,
    ) -> TextGenerationResult:
        try:
            response = await self._client.responses.create(
                model=self._config.model_name,
                instructions=instructions,
                input=input,
                max_output_tokens=max_output_tokens or self._config.max_output_tokens,
            )
        except (APITimeoutError, TimeoutError) as error:
            raise ModelTimeoutError("OpenAI model request timed out") from error
        except OpenAIError as error:
            raise ModelProviderError("OpenAI model request failed") from error

        if response.status != "completed":
            raise ModelProviderError("OpenAI model response did not complete")
        text = response.output_text
        if not isinstance(text, str) or not text:
            raise ModelProviderError("OpenAI model response contained no text")
        return TextGenerationResult(
            text=text,
            model=response.model or self._config.model_name,
            response_id=response.id,
            usage=_usage(response.usage),
        )

    async def generate_structured(
        self,
        *,
        instructions: str,
        input: str,
        output_type: type[ModelT],
        max_output_tokens: int | None = None,
    ) -> ModelT:
        try:
            response = await self._client.responses.parse(
                model=self._config.model_name,
                instructions=instructions,
                input=input,
                text_format=output_type,
                max_output_tokens=max_output_tokens or self._config.max_output_tokens,
            )
        except (APITimeoutError, TimeoutError) as error:
            raise ModelTimeoutError("OpenAI model request timed out") from error
        except APIResponseValidationError as error:
            raise ModelStructuredOutputError(
                "OpenAI returned an invalid structured response"
            ) from error
        except OpenAIError as error:
            raise ModelProviderError("OpenAI model request failed") from error

        if response.status != "completed":
            raise ModelStructuredOutputError(
                "OpenAI model response did not complete with structured output"
            )
        for output_item in response.output:
            if output_item.type != "message":
                continue
            for content in output_item.content:
                if content.type == "output_text" and content.parsed is not None:
                    if isinstance(content.parsed, output_type):
                        return content.parsed
                    raise ModelStructuredOutputError(
                        "OpenAI model returned an unexpected structured type"
                    )
                if content.type == "refusal":
                    raise ModelStructuredOutputError(
                        "OpenAI model refused structured output"
                    )
        raise ModelStructuredOutputError("OpenAI model returned no structured output")

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.close()

    async def __aenter__(self) -> OpenAIModelClient:
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.aclose()


def _usage(usage: object) -> ModelUsage:
    if usage is None:
        return ModelUsage()
    return ModelUsage(
        input_tokens=_nonnegative_int(getattr(usage, "input_tokens", None)),
        output_tokens=_nonnegative_int(getattr(usage, "output_tokens", None)),
    )


def _nonnegative_int(value: object) -> int | None:
    return value if isinstance(value, int) and value >= 0 else None
