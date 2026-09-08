"""Metadata describing an internal tool capability."""

from pydantic import field_validator

from coding_agent.domain.models import DomainModel


def _non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("value must not be blank")
    return value


class ToolDescriptor(DomainModel):
    tool_name: str
    capability: str
    description: str
    mutating: bool

    _validate_text = field_validator("tool_name", "capability", "description")(
        _non_blank
    )
