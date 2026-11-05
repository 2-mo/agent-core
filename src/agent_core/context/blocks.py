# Adapted for standalone use in Agent Core.
from __future__ import annotations

import json
import os
from collections.abc import Iterable
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

ContentBlock = dict[str, Any]

_IMAGE_INPUT_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


class _CanonicalImageBlock(dict[str, Any]):
    """Image block whose path has already been resolved."""


class _CanonicalImagePath(str):
    """String path carrying the in-process canonicalization invariant."""

    def __new__(cls, path, lease=None):
        value = super().__new__(cls, path)
        value._image_lease = lease
        return value

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return _CanonicalImagePath, (str(self),)


def _resolve_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


@lru_cache(maxsize=65_536)
def _cached_canonical_image_path(path: str) -> _CanonicalImagePath:
    return _CanonicalImagePath(_resolve_path(path))


def canonical_image_path(path: str | Path) -> str:
    if isinstance(path, _CanonicalImagePath):
        return path
    if not isinstance(path, (str, Path)):
        raise TypeError("image path must be a string or Path")
    expanded = Path(path).expanduser()
    cache_key = os.path.abspath(expanded)
    # lru_cache protects its own state; resolving a miss must not block hits.
    canonical = _cached_canonical_image_path(cache_key)
    lease = getattr(path, "_image_lease", None)
    # Lease-bearing paths must never enter the global canonicalization LRU.
    return _CanonicalImagePath(canonical, lease) if lease is not None else canonical


def _expand_image_paths(paths: Iterable[str | Path]) -> list[str]:
    expanded: list[str] = []
    for raw_path in paths:
        canonical_path = canonical_image_path(raw_path)
        path = Path(canonical_path)
        if path.is_dir():
            expanded.extend(
                canonical_image_path(file_path)
                for file_path in sorted(path.rglob("*"))
                if file_path.is_file() and file_path.suffix.lower() in _IMAGE_INPUT_EXTS
            )
            continue
        expanded.append(canonical_path)
    return expanded


def text_block(text: str) -> ContentBlock:
    if not isinstance(text, str):
        raise TypeError("text block requires a string")
    return {"type": "text", "text": text}


def image_block(path: str | Path, **metadata: Any) -> ContentBlock:
    if not isinstance(path, (str, Path)):
        raise TypeError("image block path must be a string or Path")
    return _canonical_image_block(canonical_image_path(path), metadata)


def _canonical_image_block(
    path: str,
    metadata: dict[str, Any],
) -> ContentBlock:
    block: ContentBlock = _CanonicalImageBlock(
        type="image",
        path=path,
    )
    for key, value in metadata.items():
        if value is not None:
            block[str(key)] = deepcopy(value)
    return block


def _copy_canonical_image_block(block: dict[str, Any]) -> ContentBlock:
    return _CanonicalImageBlock(deepcopy(dict(block)))


def _is_canonical_image_block(block: object) -> bool:
    return isinstance(block, _CanonicalImageBlock)


def image_blocks(paths: Iterable[str | Path], **metadata: Any) -> list[ContentBlock]:
    return [
        _canonical_image_block(path, metadata)
        for path in _expand_image_paths(paths)
    ]


def _arguments_to_text(arguments: Any) -> str:
    if arguments is None:
        return "{}"
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments, ensure_ascii=False, default=str)


def _normalize_tool_call_payload(tool_call: Any) -> dict[str, Any]:
    if hasattr(tool_call, "model_dump"):
        payload = tool_call.model_dump(exclude_none=True)
    elif isinstance(tool_call, dict):
        payload = deepcopy(tool_call)
    else:
        payload = {
            "id": getattr(tool_call, "id", None),
            "type": getattr(tool_call, "type", "function"),
            "function": {
                "name": getattr(getattr(tool_call, "function", None), "name", None),
                "arguments": getattr(
                    getattr(tool_call, "function", None),
                    "arguments",
                    None,
                ),
            },
        }

    function_payload = payload.get("function")
    if not isinstance(function_payload, dict):
        function_payload = {}
    return {
        "id": str(payload.get("id") or ""),
        "name": str(function_payload.get("name") or ""),
        "arguments": _arguments_to_text(function_payload.get("arguments")),
    }


def tool_call_block(
    tool_call: Any = None,
    *,
    id: str | None = None,
    name: str | None = None,
    arguments: Any = None,
) -> ContentBlock:
    if tool_call is not None:
        payload = _normalize_tool_call_payload(tool_call)
        id = payload["id"]
        name = payload["name"]
        arguments = payload["arguments"]
    if not name or not str(name).strip():
        raise ValueError("tool_call block requires a non-empty name")
    return {
        "type": "tool_call",
        "id": "" if id is None else str(id),
        "name": str(name).strip(),
        "arguments": _arguments_to_text(arguments),
    }


def tool_result_block(*, tool_call_id: str, name: str, content: Any) -> ContentBlock:
    if not isinstance(tool_call_id, str) or not tool_call_id:
        raise ValueError("tool_result block requires a non-empty tool_call_id")
    if not isinstance(name, str) or not name:
        raise ValueError("tool_result block requires a non-empty name")
    return {
        "type": "tool_result",
        "tool_call_id": tool_call_id,
        "name": name,
        "content": content if isinstance(content, str) else str(content),
    }


__all__ = [
    "ContentBlock",
    "canonical_image_path",
    "image_block",
    "image_blocks",
    "text_block",
    "tool_call_block",
    "tool_result_block",
]
