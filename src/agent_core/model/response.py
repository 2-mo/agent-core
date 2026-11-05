# Adapted for standalone use in Agent Core.
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any

from agent_core.context import (
    Conversation,
    Message,
    assistant_message,
    message_text,
    message_tool_calls,
)


def token_ids_digest(ids: list[int]) -> str:
    """Fingerprint of a token ID sequence: the served prompt is recorded as this, not as its IDs, since the
    model's tokenizer rebuilds the IDs and the fingerprint checks the rebuild."""
    return hashlib.sha256(",".join(str(int(value)) for value in ids).encode()).hexdigest()


@dataclass(slots=True)
class CompletionMetadata:
    endpoint: str | None
    responses: list[dict[str, Any]]
    latency_sec: float
    model: str | None = None
    usage: dict[str, Any] | None = None
    timings: dict[str, float] = field(default_factory=dict)
    choices: list[dict[str, Any]] = field(default_factory=list)
    request: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Response:
    """One model reply. Text, tool calls and generation metadata stay together."""

    message: Message
    model: str | None = None
    endpoint: str | None = None
    id: str | None = None
    finish_reason: str | None = None
    logprobs: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    latency_sec: float = 0.0
    timings: dict[str, float] = field(default_factory=dict)
    request: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    execution: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return message_text(self.message)

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        return message_tool_calls(self.message)

    def json(self) -> Any:
        """Decode the reply as strict JSON; never repair or silently change it."""
        return json.loads(self.text)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_completion(
        cls, completion: CompletionMetadata, index: int = 0
    ) -> Response:
        choice = completion.choices[index] if index < len(completion.choices) else {}
        return cls(
            message=deepcopy(completion.responses[index]),
            model=choice.get("model") or completion.model,
            endpoint=completion.endpoint,
            id=choice.get("id"),
            finish_reason=choice.get("finish_reason"),
            logprobs=deepcopy(choice.get("logprobs")),
            usage=deepcopy(completion.usage),
            latency_sec=completion.latency_sec,
            timings=deepcopy(completion.timings),
            request=deepcopy(completion.request),
            raw=deepcopy(choice),
        )


@dataclass(slots=True)
class StreamEvent:
    """A model stream delta or its final, fully assembled Response."""

    type: str
    text: str = ""
    tool_call: dict[str, Any] | None = None
    response: Response | None = None


@dataclass(slots=True)
class BatchItem:
    """One indexed batch outcome, suitable for incremental JSONL persistence."""

    index: int
    input: str
    response: Response | None = None
    error: dict[str, str] | None = None
    _exception: Exception | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "input": self.input,
            "ok": self.ok,
            "response": self.response.to_dict() if self.response else None,
            "error": deepcopy(self.error),
        }


def response_to_message(message: Any) -> Message:
    """Normalize one model response for the canonical model-turn executor."""
    if hasattr(message, "model_dump"):
        payload = message.model_dump(exclude_none=True)
        payload.setdefault("role", "assistant")
        return Conversation.normalize_message(payload)
    if isinstance(message, dict):
        # Normalization already copies every retained mutable field.
        payload = dict(message)
        payload.setdefault("role", "assistant")
        return Conversation.normalize_message(payload)

    tool_calls = getattr(message, "tool_calls", None)
    normalized_tool_calls = None
    if tool_calls is not None:
        normalized_tool_calls = [
            message_tool_calls(assistant_message(None, tool_calls=[tool_call]))[0]
            for tool_call in tool_calls
        ]
    return Conversation.normalize_message(
        {
            "role": getattr(message, "role", "assistant"),
            "content": getattr(message, "content", None),
            "tool_calls": normalized_tool_calls,
        }
    )


__all__ = [
    "BatchItem",
    "CompletionMetadata",
    "Response",
    "StreamEvent",
    "response_to_message",
]
