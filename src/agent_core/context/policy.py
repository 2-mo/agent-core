# Adapted for standalone use in Agent Core.
"""Explicit model-visible context limits; stored conversation history stays complete."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class ContextLimitError(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__(
            "context_limit: the current turn and system instructions exceed the configured context policy"
        )


@dataclass(frozen=True)
class ContextPolicy:
    max_messages: int | None = None
    max_chars: int | None = None
    max_tokens: int | None = None
    overflow: str = "error"
    token_counter: Callable[[list[dict], list[dict]], int] | None = None

    def __post_init__(self):
        for name in ("max_messages", "max_chars", "max_tokens"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        if self.overflow not in {"error", "trim_oldest"}:
            raise ValueError("overflow must be 'error' or 'trim_oldest'")
        if self.max_tokens is not None and not callable(self.token_counter):
            raise ValueError(
                "max_tokens requires a model-appropriate token_counter(messages, tools)"
            )

    def apply(self, messages: list[dict], tools: list[dict]):
        def measure(items):
            # Character limits cover serialized message content and tool schemas;
            # source metadata is not model-visible. This is not a token estimate.
            visible = [
                {k: v for k, v in item.items() if k != "metadata"} for item in items
            ]
            counts = {
                "messages": len(items),
                "chars": len(json.dumps([visible, tools], ensure_ascii=False)),
            }
            if self.max_tokens is not None:
                value = self.token_counter(items, tools)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError("token_counter must return a non-negative integer")
                counts["tokens"] = value
            return counts

        selected = list(messages)
        before = counts = measure(selected)
        removed = []

        def exceeds():
            return any(
                limit is not None and counts[name] > limit
                for name, limit in (
                    ("messages", self.max_messages),
                    ("chars", self.max_chars),
                    ("tokens", self.max_tokens),
                )
            )

        while exceeds() and self.overflow == "trim_oldest":
            users = [i for i, item in enumerate(selected) if item["role"] == "user"]
            if len(users) < 2:
                break
            # Remove a whole old turn. Keep all system messages, the newest user
            # turn, and complete assistant/tool groups belonging to that turn.
            end = users[1]
            dropped = [i for i in range(end) if selected[i]["role"] != "system"]
            if not dropped:
                break
            removed.append(len(dropped))
            selected = [item for i, item in enumerate(selected) if i not in dropped]
            counts = measure(selected)
        report: dict[str, Any] = {
            "overflow": self.overflow,
            "limits": {
                "messages": self.max_messages,
                "chars": self.max_chars,
                "tokens": self.max_tokens,
            },
            "before": before,
            "after": counts,
            "removed_turns": len(removed),
            "removed_messages": sum(removed),
            "status": "exceeded" if exceeds() else "ok",
        }
        if exceeds():
            raise ContextLimitError(report)
        return selected, report
