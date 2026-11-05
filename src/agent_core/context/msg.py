# Adapted for standalone use in Agent Core.
from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from .blocks import ContentBlock, image_block, image_blocks, text_block
from .messages import (
    Message,
    assistant_message,
    system_message,
    tool_message,
    user_message,
)
from .media import frame_image_block as frame_image
from .media import frame_video_block as frame_video
from .video import video_block as video

ImagePaths = Iterable[str | Path]


def text(text: str) -> ContentBlock:
    return text_block(text)


def image(path: str | Path, **metadata: Any) -> ContentBlock:
    return image_block(path, **metadata)


def images(paths: ImagePaths, **metadata: Any) -> list[ContentBlock]:
    return image_blocks(paths, **metadata)


def system(content: str) -> Message:
    return system_message(content)


def user(
    content: Sequence[ContentBlock] | str | None = None,
    *,
    text: str | None = None,
    images: ImagePaths | None = None,
    image_paths: ImagePaths | None = None,
) -> Message:
    if images is not None and image_paths is not None:
        raise ValueError("use either images or image_paths, not both")

    selected_images = images if images is not None else image_paths
    blocks: list[ContentBlock] = []

    if selected_images is not None:
        blocks.extend(image_blocks(selected_images))

    if isinstance(content, str):
        if text is not None:
            raise ValueError("use either positional content or text, not both")
        if content:
            blocks.append(text_block(content))
    elif content is not None:
        blocks.extend(list(content))

    if text is not None:
        blocks.append(text_block(text))

    return user_message(blocks)


def assistant(
    content: Sequence[ContentBlock] | str | None = None,
    *,
    tool_calls: Sequence[Any] | None = None,
) -> Message:
    return assistant_message(content, tool_calls=tool_calls)


def tool(*, tool_call_id: str, name: str, content: Any) -> Message:
    return tool_message(tool_call_id=tool_call_id, name=name, content=content)


__all__ = [
    "assistant",
    "frame_image",
    "frame_video",
    "image",
    "images",
    "system",
    "text",
    "tool",
    "user",
    "video",
]
