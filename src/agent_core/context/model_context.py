# Adapted for standalone use in Agent Core.
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .messages import Message, validate_messages
from .snapshot import message_snapshot


@dataclass(frozen=True, slots=True)
class ContextStats:
    message_count: int
    text_chars: int
    content_blocks: int
    source_names: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_count": self.message_count,
            "text_chars": self.text_chars,
            "content_blocks": self.content_blocks,
            "source_names": list(self.source_names),
        }


@dataclass(frozen=True, slots=True)
class ModelContext:
    """Immutable model-visible view derived from runtime facts.

    The source mappings are snapshots. They are inspectable context inputs, not
    alternate state or trajectory stores, and are never mutated by construction.
    """

    messages: tuple[Message, ...]
    goal: dict[str, Any] | None = None
    state: dict[str, Any] | None = None
    trajectory: dict[str, Any] | None = None
    observation: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    stats: ContextStats | None = None

    def to_messages(self) -> list[Message]:
        return deepcopy(list(self.messages))

    def inspect(self) -> dict[str, Any]:
        return {
            "stats": (self.stats or _stats(self.messages, self)).to_dict(),
            "goal": deepcopy(self.goal),
            "state": deepcopy(self.state),
            "trajectory": deepcopy(self.trajectory),
            "observation": deepcopy(self.observation),
            "metadata": deepcopy(self.metadata or {}),
        }

    def summary(self) -> dict[str, Any]:
        """Small persisted description of composition, without duplicating facts."""

        return {
            "stats": (self.stats or _stats(self.messages, self)).to_dict(),
            "metadata": deepcopy(self.metadata or {}),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": self.to_messages(),
            **self.inspect(),
        }


class ContextBuilder:
    """Construct generic ModelContext snapshots without domain knowledge."""

    def build(
        self,
        *,
        messages: Sequence[Message],
        goal: Mapping[str, Any] | None = None,
        state: Mapping[str, Any] | None = None,
        trajectory: Mapping[str, Any] | None = None,
        observation: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ModelContext:
        normalized = tuple(validate_messages(deepcopy(list(messages))))
        context = ModelContext(
            messages=normalized,
            goal=None if goal is None else deepcopy(dict(goal)),
            state=None if state is None else deepcopy(dict(state)),
            trajectory=(
                None if trajectory is None else deepcopy(dict(trajectory))
            ),
            observation=(
                None if observation is None else deepcopy(dict(observation))
            ),
            metadata=None if metadata is None else deepcopy(dict(metadata)),
        )
        return _with_stats(context)


def _with_stats(context: ModelContext) -> ModelContext:
    return ModelContext(
        messages=context.messages,
        goal=context.goal,
        state=context.state,
        trajectory=context.trajectory,
        observation=context.observation,
        metadata=context.metadata,
        stats=_stats(context.messages, context),
    )


def _stats(messages: Sequence[Message], context: ModelContext) -> ContextStats:
    snapshot = message_snapshot(messages)
    source_names = tuple(
        name
        for name, value in (
            ("goal", context.goal),
            ("state", context.state),
            ("trajectory", context.trajectory),
            ("observation", context.observation),
        )
        if value is not None
    )
    return ContextStats(
        message_count=int(snapshot["message_count"]),
        text_chars=len(str(snapshot["text"])),
        content_blocks=sum(len(message.get("content", [])) for message in messages),
        source_names=source_names,
    )


__all__ = ["ContextBuilder", "ContextStats", "ModelContext"]
