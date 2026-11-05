# Adapted for standalone use in Agent Core.
from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from copy import deepcopy
from threading import RLock
from typing import Any
from uuid import uuid4

from .blocks import ContentBlock
from .messages import (
    ImageURLMode,
    Message,
    assistant_message,
    system_message,
    tool_message,
    user_message,
    validate_messages,
)
from .render import render_openai


class Conversation:
    def __init__(
        self,
        items: Sequence[Message] | None = None,
        *,
        image_url_mode: ImageURLMode = "file",
    ) -> None:
        self.image_url_mode = image_url_mode
        self.id = uuid4().hex
        self._turn_lock = RLock()
        self.items: list[Message] = validate_messages(list(items or []))

    @classmethod
    def from_messages(
        cls,
        messages: Sequence[Message] | Conversation,
        *,
        image_url_mode: ImageURLMode = "file",
    ) -> Conversation:
        if isinstance(messages, Conversation):
            return cls(messages.to_list(), image_url_mode=messages.image_url_mode)
        return cls(messages, image_url_mode=image_url_mode)

    @staticmethod
    def normalize_message(message: Message) -> Message:
        return validate_messages([message])[0]

    @classmethod
    def load_snapshot(cls, snapshot: dict[str, Any]) -> Conversation:
        conversation = cls(
            snapshot.get("messages", []),
            image_url_mode=snapshot.get("image_url_mode", "file"),
        )
        conversation.id = str(snapshot.get("id") or conversation.id)
        return conversation

    @contextmanager
    def turn(self, *, check=None):
        """Serialize calls sharing this history; independent histories never block."""
        if check is None:
            self._turn_lock.acquire()
        else:
            check()
            while not self._turn_lock.acquire(timeout=0.05):
                check()
        try:
            if check is not None:
                check()
            yield self
        finally:
            self._turn_lock.release()

    def fork(self) -> Conversation:
        """Copy a history into an independent branch."""
        with self._turn_lock:
            return Conversation.from_messages(self)

    def append(self, message: Message) -> None:
        self.items.append(self.normalize_message(message))

    def add_system(self, content: str) -> None:
        self.items.append(system_message(content))

    def add_user(self, content: list[ContentBlock] | str) -> None:
        self.items.append(user_message(content))

    def add_assistant(
        self,
        content: str | list[ContentBlock] | None,
        *,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        self.items.append(assistant_message(content, tool_calls=tool_calls))

    def add_tool(self, *, tool_call_id: str, name: str, content: str) -> None:
        self.items.append(
            tool_message(tool_call_id=tool_call_id, name=name, content=content)
        )

    def extend(self, messages: Sequence[Message] | Conversation) -> None:
        payload = messages.to_list() if isinstance(messages, Conversation) else list(messages)
        self.items.extend(validate_messages(payload))

    def capture_delta(self) -> ConversationDelta:
        return ConversationDelta(self, self.mark())

    def mark(self) -> int:
        return len(self.items)

    def new_messages_since(self, mark: int) -> list[Message]:
        if mark < 0 or mark > len(self.items):
            raise ValueError(f"message mark out of range: {mark}")
        return deepcopy(self.items[mark:])

    def compile_window(self, policy: Any | None = None) -> list[Message]:
        max_messages = _policy_max_messages(policy)
        if max_messages is None:
            return self.to_list()
        return _trim_items(self.items, max_messages=max_messages)

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "image_url_mode": self.image_url_mode,
            "messages": self.to_list(),
        }

    def load(self, snapshot: dict[str, Any] | Sequence[Message]) -> None:
        if isinstance(snapshot, dict):
            self.id = str(snapshot.get("id") or self.id)
            self.image_url_mode = snapshot.get("image_url_mode", self.image_url_mode)
            items = snapshot.get("messages", [])
        else:
            items = snapshot
        self.items = validate_messages(list(items))

    def reset(self, messages: Sequence[Message] | Conversation | None = None) -> None:
        if messages is None:
            self.items = []
            return
        payload = messages.to_list() if isinstance(messages, Conversation) else list(messages)
        self.items = validate_messages(payload)

    def trim_history(self, max_messages: int) -> None:
        self.items = _trim_items(self.items, max_messages=max_messages)

    def render_openai(
        self,
        image_url_mode: ImageURLMode | None = None,
    ) -> list[dict[str, Any]]:
        return render_openai(
            self.items,
            image_url_mode=self.image_url_mode if image_url_mode is None else image_url_mode,
        )

    def to_list(self) -> list[Message]:
        return deepcopy(self.items)

    def __iter__(self):
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Message:
        return self.items[index]


class ConversationDelta:
    def __init__(self, conversation: Conversation, mark: int) -> None:
        self._conversation = conversation
        self._mark = mark

    def append(self, message: Message) -> Message:
        message_mark = self._conversation.mark()
        self._conversation.append(message)
        return self._conversation.new_messages_since(message_mark)[0]

    def add_tool(self, *, tool_call_id: str, name: str, content: str) -> Message:
        message_mark = self._conversation.mark()
        self._conversation.add_tool(tool_call_id=tool_call_id, name=name, content=content)
        return self._conversation.new_messages_since(message_mark)[0]

    def extend(self, messages: Sequence[Message] | Conversation) -> list[Message]:
        message_mark = self._conversation.mark()
        self._conversation.extend(messages)
        return self._conversation.new_messages_since(message_mark)

    def messages(self) -> list[Message]:
        return self._conversation.new_messages_since(self._mark)


def _policy_max_messages(policy: Any | None) -> int | None:
    if policy is None:
        return None
    if isinstance(policy, int):
        return max(0, policy)
    if isinstance(policy, dict):
        value = policy.get("max_messages")
    else:
        value = getattr(policy, "max_messages", None)
    if value is None:
        return None
    return max(0, int(value))


def _trim_items(items: Sequence[Message], *, max_messages: int) -> list[Message]:
    normalized = validate_messages(list(items))
    max_messages = max(0, int(max_messages))
    system_items = [item for item in normalized if item.get("role") == "system"]
    non_system = [item for item in normalized if item.get("role") != "system"]
    if max_messages == 0:
        return deepcopy(system_items)
    if len(non_system) <= max_messages:
        return deepcopy(normalized)

    turns = _message_turns(non_system)
    kept_turns: list[list[Message]] = []
    kept_count = 0
    for turn in reversed(turns):
        turn_len = len(turn)
        if kept_count + turn_len > max_messages:
            if not kept_turns and not _turn_has_tool_pair(turn):
                kept_turns.append(turn[-max_messages:])
            break
        kept_turns.append(turn)
        kept_count += turn_len
    kept_non_system = [
        deepcopy(message)
        for turn in reversed(kept_turns)
        for message in turn
    ]
    return [*deepcopy(system_items), *kept_non_system]


def _message_turns(messages: Sequence[Message]) -> list[list[Message]]:
    turns: list[list[Message]] = []
    current: list[Message] = []
    for message in messages:
        if message.get("role") == "user" and current:
            turns.append(current)
            current = []
        current.append(message)
    if current:
        turns.append(current)
    return turns


def _turn_has_tool_pair(turn: Sequence[Message]) -> bool:
    return any(message.get("role") == "tool" for message in turn) or any(
        block.get("type") == "tool_call"
        for message in turn
        if message.get("role") == "assistant"
        for block in message.get("content", [])
    )


__all__ = ["Conversation", "ConversationDelta"]
