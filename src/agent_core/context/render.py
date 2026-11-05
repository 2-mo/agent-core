# Adapted for standalone use in Agent Core.
from __future__ import annotations

import base64
import mimetypes
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from .blocks import _is_canonical_image_block
from .messages import (
    ImageURLMode,
    Message,
    _message_text,
    _message_tool_calls,
    _message_tool_result,
    validate_messages,
)
from .media import block_data_urls
from .video import encode_video_url

VideoMode = Literal["frames", "images"]


def _image_url(
    image_path: str,
    *,
    image_url_mode: ImageURLMode,
    path_is_canonical: bool,
) -> str:
    path = Path(image_path)
    if not path_is_canonical:
        path = path.expanduser().resolve()
    if image_url_mode == "data":
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        payload = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:{mime};base64,{payload}"
    return path.as_uri()


def render_openai(
    messages: Sequence[Message],
    *,
    image_url_mode: ImageURLMode = "file",
    video_mode: VideoMode = "images",
) -> list[dict[str, Any]]:
    return _render_openai(
        validate_messages(messages), image_url_mode=image_url_mode, video_mode=video_mode
    )


def _render_openai(
    messages: Sequence[Message],
    *,
    image_url_mode: ImageURLMode,
    video_mode: VideoMode = "images",
) -> list[dict[str, Any]]:
    """Render messages already validated at the request boundary.

    ``video_mode="frames"`` sends a video block as one ``video_url`` part that the
    a custom server can decode; ``"images"`` sends its frames as standard
    ``image_url`` parts, each preceded by its source time, for any OpenAI-compatible
    vision model.
    """
    rendered: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        content = message["content"]

        if role == "system":
            rendered.append({"role": "system", "content": _message_text(message)})
            continue

        if role == "user":
            parts: list[dict[str, Any]] = []
            for block in content:
                if block["type"] == "text":
                    parts.append({"type": "text", "text": block["text"]})
                elif block["type"] == "image":
                    url = (
                        block_data_urls(block)[0]
                        if "source" in block
                        else _image_url(
                            block["path"],
                            image_url_mode=image_url_mode,
                            path_is_canonical=_is_canonical_image_block(block),
                        )
                    )
                    parts.append({"type": "image_url", "image_url": {"url": url}})
                elif block["type"] == "video":
                    frame_urls = (
                        block_data_urls(block)
                        if "source" in block
                        else [
                            _image_url(path, image_url_mode="data", path_is_canonical=True)
                            for path in block["frames"]
                        ]
                    )
                    if video_mode == "images":
                        parts.extend(_frame_parts(frame_urls, block["video_metadata"]))
                        continue
                    parts.append(
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": encode_video_url(frame_urls, block["video_metadata"])
                            },
                        }
                    )
            rendered.append({"role": "user", "content": parts})
            continue

        if role == "assistant":
            tool_calls = _message_tool_calls(message)
            text = _message_text(message)
            payload: dict[str, Any] = {"role": "assistant", "content": text or None}
            if tool_calls:
                payload["tool_calls"] = tool_calls
            rendered.append(payload)
            continue

        if role == "tool":
            result = _message_tool_result(message)
            rendered.append(
                {
                    "role": "tool",
                    "content": result["content"],
                    "tool_call_id": result["tool_call_id"],
                    "name": result["name"],
                }
            )
            continue

    return rendered


def _frame_parts(frame_urls: Sequence[str], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    fps = metadata["fps"]
    parts: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": f"Video: {len(frame_urls)} frames sampled from the source video "
            f"({fps:g} fps); each frame follows its source time.",
        }
    ]
    for index, url in zip(metadata["frames_indices"], frame_urls, strict=True):
        parts.append({"type": "text", "text": f"[{index / fps:.2f}s]"})
        parts.append({"type": "image_url", "image_url": {"url": url}})
    return parts


__all__ = ["VideoMode", "render_openai"]
