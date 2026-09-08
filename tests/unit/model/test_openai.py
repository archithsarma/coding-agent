from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
from openai import APIResponseValidationError, AsyncOpenAI, OpenAIError
from pydantic import BaseModel

from coding_agent.model.errors import (
    ModelProviderError,
    ModelStructuredOutputError,
    ModelTimeoutError,
)
from coding_agent.model.openai import OpenAIModelClient
from coding_agent.model.protocol import TextGenerationResult
from coding_agent.model.settings import OpenAIModelConfig


class Answer(BaseModel):
    value: str


class FakeResponses:
    def __init__(self) -> None:
        self.create_result: object = SimpleNamespace(
            id="resp-text",
            model="configured-model",
            status="completed",
            output_text="hello",
            usage=SimpleNamespace(input_tokens=3, output_tokens=2),
        )
        self.parse_result: object = SimpleNamespace(
            status="completed",
            output=[
                SimpleNamespace(
                    type="message",
                    content=[
                        SimpleNamespace(
                            type="output_text", parsed=Answer(value="selected")
                        )
                    ],
                )
            ],
        )
        self.create_calls: list[dict[str, object]] = []
        self.parse_calls: list[dict[str, object]] = []
        self.create_error: BaseException | None = None
        self.parse_error: BaseException | None = None

    async def create(self, **kwargs: object) -> object:
        self.create_calls.append(kwargs)
        if self.create_error is not None:
            raise self.create_error
        return self.create_result

    async def parse(self, **kwargs: object) -> object:
        self.parse_calls.append(kwargs)
        if self.parse_error is not None:
            raise self.parse_error
        return self.parse_result


class FakeClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def client(fake: FakeClient) -> OpenAIModelClient:
    return OpenAIModelClient(
        OpenAIModelConfig(model_name="configured-model"),
        client=cast(AsyncOpenAI, fake),
    )


@pytest.mark.anyio
async def test_text_generation_normalizes_provider_response() -> None:
    fake = FakeClient()

    result = await client(fake).generate_text(
        instructions="be concise", input="say hello"
    )

    assert isinstance(result, TextGenerationResult)
    assert result.text == "hello"
    assert result.model == "configured-model"
    assert result.response_id == "resp-text"
    assert result.usage.input_tokens == 3
    assert result.usage.output_tokens == 2
    assert fake.responses.create_calls[0]["instructions"] == "be concise"
    assert "tools" not in fake.responses.create_calls[0]


@pytest.mark.anyio
async def test_structured_generation_returns_caller_pydantic_type() -> None:
    fake = FakeClient()

    result = await client(fake).generate_structured(
        instructions="choose", input="select one", output_type=Answer
    )

    assert isinstance(result, Answer)
    assert result.value == "selected"
    assert fake.responses.parse_calls[0]["text_format"] is Answer
    assert "tools" not in fake.responses.parse_calls[0]


@pytest.mark.anyio
async def test_timeout_is_translated_and_cause_preserved() -> None:
    fake = FakeClient()
    cause = TimeoutError("deadline")
    fake.responses.create_error = cause

    with pytest.raises(ModelTimeoutError) as raised:
        await client(fake).generate_text(instructions="x", input="y")

    assert raised.value.__cause__ is cause


@pytest.mark.anyio
async def test_provider_error_is_translated_without_raw_response() -> None:
    fake = FakeClient()
    cause = OpenAIError("provider unavailable")
    fake.responses.create_error = cause

    with pytest.raises(ModelProviderError) as raised:
        await client(fake).generate_text(instructions="x", input="y")

    assert raised.value.__cause__ is cause
    assert "provider unavailable" not in str(raised.value)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "parse_result",
    [
        SimpleNamespace(status="completed", output=[]),
        SimpleNamespace(
            status="completed",
            output=[
                SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="output_text", parsed=None)],
                )
            ],
        ),
        SimpleNamespace(
            status="completed",
            output=[
                SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="refusal", refusal="no")],
                )
            ],
        ),
    ],
)
async def test_structured_missing_or_refused_output_fails_closed(
    parse_result: object,
) -> None:
    fake = FakeClient()
    fake.responses.parse_result = parse_result

    with pytest.raises(ModelStructuredOutputError):
        await client(fake).generate_structured(
            instructions="choose", input="select one", output_type=Answer
        )


@pytest.mark.anyio
async def test_non_completed_structured_response_is_not_defaulted() -> None:
    fake = FakeClient()
    fake.responses.parse_result = SimpleNamespace(status="incomplete", output=[])

    with pytest.raises(ModelStructuredOutputError):
        await client(fake).generate_structured(
            instructions="choose", input="select one", output_type=Answer
        )


@pytest.mark.anyio
async def test_sdk_structured_validation_error_is_translated() -> None:
    fake = FakeClient()
    fake.responses.parse_error = APIResponseValidationError(
        SimpleNamespace(request=object(), status_code=400), None, message="invalid"
    )

    with pytest.raises(ModelStructuredOutputError) as raised:
        await client(fake).generate_structured(
            instructions="choose", input="select one", output_type=Answer
        )

    assert raised.value.__cause__ is fake.responses.parse_error


def test_api_key_is_masked_and_not_serialized() -> None:
    config = OpenAIModelConfig(
        model_name="configured-model", api_key="sk-test-secret-value"
    )

    assert "sk-test-secret-value" not in repr(config)
    assert "sk-test-secret-value" not in config.model_dump_json()


@pytest.mark.anyio
async def test_injected_client_is_not_closed_by_adapter() -> None:
    fake = FakeClient()
    adapter = client(fake)

    await adapter.aclose()

    assert fake.closed is False
