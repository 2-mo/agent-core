# Adapted for standalone use in Agent Core.
from __future__ import annotations

import json
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from agent_core.context import Message, validate_messages

type RuntimeContext = Any
ToolStatus = Literal["success", "failed", "error"]
_TOOL_STATUSES = {"success", "failed", "error"}


@dataclass(slots=True)
class ToolResult:
    """A tool-owned fact returned to the model loop.

    ``failed`` is an expected, model-repairable rejection such as invalid
    structured arguments. ``error`` means the capability could not execute and
    the domain may terminate the run. Both are visible to the model as tool
    results; the runtime does not invent a separate verification phase.
    """

    status: ToolStatus = "success"
    content: str = ""
    error: str | None = None
    messages_to_append: list[Message] = field(default_factory=list)
    state_delta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in _TOOL_STATUSES:
            raise ValueError(f"unknown tool result status: {self.status!r}")

    @property
    def ok(self) -> bool:
        return self.status not in {"failed", "error"} and self.error is None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_tool_result(raw: Any) -> ToolResult:
    if isinstance(raw, ToolResult):
        return ToolResult(
            status=raw.status,
            content=raw.content,
            error=raw.error,
            messages_to_append=validate_messages(raw.messages_to_append),
            state_delta=deepcopy(raw.state_delta),
        )
    if isinstance(raw, str):
        return ToolResult(content=raw)
    if not isinstance(raw, dict):
        return ToolResult(content=_json_content(raw))

    status = str(raw.get("status", "success"))
    if status not in _TOOL_STATUSES:
        raise ValueError(f"unknown tool result status: {status!r}")
    error = raw.get("error")
    content = raw.get("content")
    if content is None:
        content = _json_content(raw)
    elif not isinstance(content, str):
        content = _json_content(content)

    messages_to_append = raw.get("messages_to_append", [])
    if messages_to_append is None:
        messages_to_append = []
    if not isinstance(messages_to_append, Sequence) or isinstance(
        messages_to_append,
        (str, bytes),
    ):
        raise TypeError("ToolResult.messages_to_append must be a sequence of messages")

    state_delta = raw.get("state_delta", {})
    if state_delta is None:
        state_delta = {}
    if not isinstance(state_delta, dict):
        raise TypeError("ToolResult.state_delta must be a dict")

    return ToolResult(
        status=status,  # type: ignore[arg-type]
        content=content,
        error=None if error is None else str(error),
        messages_to_append=validate_messages(list(messages_to_append)),
        state_delta=deepcopy(state_delta),
    )


def _json_content(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


__all__ = ["RuntimeContext", "ToolResult", "ToolStatus", "normalize_tool_result"]
