# Adapted for standalone use in Agent Core.
from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy

from .blocks import text_block
from .conversation import Conversation
from .messages import Message, system_message, validate_messages


def with_last_user_text_suffix(messages: Sequence[Message] | Conversation, suffix: str) -> list[Message]:
    normalized = (
        messages.to_list()
        if isinstance(messages, Conversation)
        else validate_messages(list(messages))
    )
    varied = deepcopy(normalized)

    for message in reversed(varied):
        if message.get("role") != "user":
            continue

        for block in reversed(message["content"]):
            if block["type"] == "text":
                block["text"] += suffix
                return varied

        message["content"].append(text_block(suffix))
        return varied

    return varied


def with_system_prompt(
    messages: Sequence[Message] | Conversation,
    system_prompt: str | None,
) -> list[Message]:
    normalized = (
        messages.to_list()
        if isinstance(messages, Conversation)
        else validate_messages(list(messages))
    )
    if not system_prompt:
        return deepcopy(normalized)
    return [system_message(system_prompt), *deepcopy(normalized)]


__all__ = ["with_last_user_text_suffix", "with_system_prompt"]
