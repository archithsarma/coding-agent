"""Canonical source-content identity helpers."""

from __future__ import annotations

import hashlib


def sha256_text(content: str) -> str:
    """Hash exact UTF-8 content using lowercase SHA-256 hex."""

    return hashlib.sha256(content.encode("utf-8")).hexdigest()
