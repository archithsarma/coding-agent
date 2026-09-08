"""Bounds for the transactional Edit trajectory."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EditConfig:
    max_write_bytes: int = 1_048_576

    def __post_init__(self) -> None:
        if self.max_write_bytes <= 0:
            raise ValueError("max_write_bytes must be greater than zero")
