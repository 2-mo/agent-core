# Adapted for standalone use in Agent Core.
from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any, Literal

from .blocks import (
    ContentBlock,
    _copy_canonical_image_block,
    _is_canonical_image_block,
    image_block,
    text_block,
    tool_call_block,
    tool_result_block,
)
from .media import validate_frame_image_block, validate_frame_video_block
from .video import video_block

MessageRole = Literal["system", "user", "assistant", "tool"]
ImageURLMode = Literal["file", "data"]
Message = dict[str, Any]

_VALID_ROLES = {"system", "user", "assistant", "tool"}


def _normalize_block(block: Any, *, role: str) -> ContentBlock:
    if not isinstance(block, dict):
        raise TypeError("message content blocks must be dictionaries")

    block_type = block.get("type")
    if block_type == "text":
        text = block.get("text", "")
        if not isinstance(text, str):
            raise TypeError("text block text must be a string")
        return text_block(text)

    if block_type == "image":
        if role != "user":
            raise ValueError("image blocks are only valid on user messages")
        if "source" in block:
            return validate_frame_image_block(block)
        if "path" not in block:
            raise ValueError("image block requires path or source")
        if _is_canonical_image_block(block):
            return _copy_canonical_image_block(block)
        metadata = {
            key: value
            for key, value in block.items()
            if key not in {"type", "path"}
        }
        return image_block(block["path"], **metadata)

    if block_type == "video":
        if role != "user":
            raise ValueError("video blocks are only valid on user messages")
        if "source" in block:
            return validate_frame_video_block(block)
        return video_block(block["frames"], **block["video_metadata"])

    if block_type == "tool_call":
        if role != "assistant":
            raise ValueError("tool_call blocks are only valid on assistant messages")
        return tool_call_block(
            id=str(block.get("id") or ""),
            name=block.get("name"),
            arguments=block.get("arguments"),
        )

    if block_type == "tool_result":
        if role != "tool":
            raise ValueError("tool_result blocks are only valid on tool messages")
        return tool_result_block(
            tool_call_id=str(block.get("tool_call_id")),
            name=str(block.get("name") or ""),
            content=block.get("content", ""),
        )

    raise ValueError(f"unsupported content block type: {block_type!r}")


def _normalize_content(role: str, content: Any, *, tool_calls: Any = None) -> list[ContentBlock]:
    blocks: list[ContentBlock] = []
    if content is None:
        pass
    elif isinstance(content, str):
        if content:
            blocks.append(text_block(content))
    elif isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        blocks.extend(_normalize_block(block, role=role) for block in content)
    else:
        raise TypeError("message content must be a string, list of content blocks, or None")

    if tool_calls is not None:
        if role != "assistant":
            raise ValueError("tool_calls are only valid on assistant messages")
        if not isinstance(tool_calls, Sequence) or isinstance(tool_calls, (str, bytes)):
            raise TypeError("tool_calls must be a sequence")
        blocks.extend(tool_call_block(tool_call) for tool_call in tool_calls)

    if role == "tool" and not blocks:
        raise ValueError("tool messages require a tool_result block")
    return blocks


def validate_messages(messages: Sequence[Message]) -> list[Message]:
    if isinstance(messages, (str, bytes)):
        raise TypeError("messages must be a sequence of message dictionaries")

    normalized_messages: list[Message] = []
    for message in messages:
        if not isinstance(message, dict):
            raise TypeError("message must be a dict")
        if "image_paths" in message:
            raise ValueError("image_paths are not part of canonical messages; use image blocks")

        role = message.get("role")
        if role not in _VALID_ROLES:
            raise ValueError(f"unsupported message role: {role!r}")

        normalized_messages.append(
            {
                "role": role,
                "content": _normalize_content(
                    role,
                    message.get("content", []),
                    tool_calls=message.get("tool_calls"),
                ),
            }
        )
        if "metadata" in message:
            if not isinstance(message["metadata"], dict):
                raise TypeError("message metadata must be a dict")
            normalized_messages[-1]["metadata"] = deepcopy(message["metadata"])
    return normalized_messages


def system_message(content: str) -> Message:
    return validate_messages([{"role": "system", "content": [text_block(content)]}])[0]


def user_message(content: Sequence[ContentBlock] | str) -> Message:
    blocks = [text_block(content)] if isinstance(content, str) else list(content)
    return validate_messages([{"role": "user", "content": blocks}])[0]


def assistant_message(
    content: Sequence[ContentBlock] | str | None = None,
    *,
    tool_calls: Sequence[Any] | None = None,
) -> Message:
    return validate_messages(
        [{"role": "assistant", "content": content, "tool_calls": tool_calls}]
    )[0]


def tool_message(*, tool_call_id: str, name: str, content: Any) -> Message:
    return validate_messages(
        [
            {
                "role": "tool",
                "content": [
                    tool_result_block(
                        tool_call_id=tool_call_id,
                        name=name,
                        content=content,
                    )
                ],
            }
        ]
    )[0]


def message_text(message: Message) -> str:
    return _message_text(validate_messages([message])[0])


def _message_text(message: Message) -> str:
    """Extract text from an already validated message owned by the caller."""
    texts: list[str] = []
    for block in message["content"]:
        if block["type"] == "text":
            texts.append(block["text"])
        elif block["type"] == "tool_result":
            texts.append(block["content"])
    return "\n".join(texts)


def message_tool_calls(message: Message) -> list[dict[str, Any]]:
    return _message_tool_calls(validate_messages([message])[0])


def _message_tool_calls(message: Message) -> list[dict[str, Any]]:
    tool_calls: list[dict[str, Any]] = []
    for block in message["content"]:
        if block["type"] != "tool_call":
            continue
        tool_calls.append(
            {
                "id": block["id"],
                "type": "function",
                "function": {
                    "name": block["name"],
                    "arguments": block["arguments"],
                },
            }
        )
    return tool_calls


def message_tool_result(message: Message) -> dict[str, str]:
    return _message_tool_result(validate_messages([message])[0])


def _message_tool_result(message: Message) -> dict[str, str]:
    if message["role"] != "tool":
        raise ValueError("message is not a tool message")
    for block in message["content"]:
        if block["type"] == "tool_result":
            return {
                "tool_call_id": block["tool_call_id"],
                "name": block["name"],
                "content": block["content"],
            }
    raise ValueError("tool message missing tool_result block")


__all__ = [
    "ContentBlock",
    "ImageURLMode",
    "Message",
    "MessageRole",
    "assistant_message",
    "message_text",
    "message_tool_calls",
    "message_tool_result",
    "system_message",
    "tool_message",
    "user_message",
    "validate_messages",
]
