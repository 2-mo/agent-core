# Adapted for standalone use in Agent Core.
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from copy import deepcopy
from typing import Any

from .messages import Message, message_text, validate_messages


def stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def content_hash(value: Any, *, prefix: str = "sha256") -> str:
    digest = hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


def message_snapshot(messages: Sequence[Message]) -> dict[str, Any]:
    normalized = validate_messages(deepcopy(list(messages)))
    return {
        "messages": normalized,
        "text": "\n\n".join(
            message_text(message)
            for message in normalized
            if message_text(message).strip()
        ),
        "message_count": len(normalized),
    }


def prompt_hash(prompt: str | Sequence[Message] | Any) -> str:
    if isinstance(prompt, str):
        payload: Any = {"type": "text", "text": prompt}
    elif isinstance(prompt, Sequence) and not isinstance(prompt, (bytes, bytearray)):
        payload = {"type": "messages", **message_snapshot(prompt)}
    else:
        payload = prompt
    return content_hash(payload)


__all__ = [
    "content_hash",
    "message_snapshot",
    "prompt_hash",
    "stable_json",
]
