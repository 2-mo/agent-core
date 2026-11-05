# Adapted for standalone use in Agent Core.
"""Frame references: image and video blocks that name source frames instead of files.

A reference block carries ``source`` (an opaque mapping owned by the data layer),
``width``/``height`` of the delivered image, and either ``frame_idx`` (image, with an
optional ``crop`` pixel box) or ``video_metadata.frames_indices`` (video). Requests
resolve references to encoded bytes through the resolver the data layer registers,
so stored trajectories never depend on local frame files.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any

from .video import validate_video_metadata

# resolver(source, frame_indices, crop) -> [(mime_type, encoded_bytes), ...] in order.
FrameResolver = Callable[
    [Mapping[str, Any], Sequence[int], Sequence[int] | None], list[tuple[str, bytes]]
]

_resolver: FrameResolver | None = None


def set_frame_resolver(resolver: FrameResolver | None) -> None:
    """Install the process-wide resolver for frame reference blocks."""
    global _resolver
    _resolver = resolver


def resolve_frames(
    source: Mapping[str, Any],
    frame_indices: Sequence[int],
    *,
    crop: Sequence[int] | None = None,
) -> list[tuple[str, bytes]]:
    if _resolver is None:
        raise RuntimeError(
            "frame reference blocks need a frame resolver; call set_frame_resolver() to register one"
        )
    indices = list(frame_indices)
    items = _resolver(source, indices, None if crop is None else list(crop))
    if len(items) != len(indices):
        raise ValueError("frame resolver returned a different number of frames")
    return items


def data_url(mime_type: str, data: bytes) -> str:
    return f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}"


def block_data_urls(block: Mapping[str, Any]) -> list[str]:
    """Encoded images of one reference block, in request order."""
    if block["type"] == "video":
        indices = block["video_metadata"]["frames_indices"]
        items = resolve_frames(block["source"], indices)
    else:
        items = resolve_frames(block["source"], [block["frame_idx"]], crop=block.get("crop"))
    return [data_url(mime_type, data) for mime_type, data in items]


def frame_image_block(
    source: Mapping[str, Any],
    frame_idx: int,
    *,
    width: int,
    height: int,
    crop: Sequence[int] | None = None,
    **metadata: Any,
) -> dict[str, Any]:
    block = {
        "type": "image",
        "source": _source(source),
        "frame_idx": frame_idx,
        "width": width,
        "height": height,
    }
    if crop is not None:
        block["crop"] = list(crop)
    for key, value in metadata.items():
        if value is not None and key not in block:
            block[str(key)] = deepcopy(value)
    return validate_frame_image_block(block)


def frame_video_block(
    source: Mapping[str, Any],
    *,
    width: int,
    height: int,
    fps: float,
    frames_indices: Sequence[int],
    total_num_frames: int,
    duration: float | None = None,
) -> dict[str, Any]:
    metadata = {
        "fps": fps,
        "frames_indices": list(frames_indices),
        "total_num_frames": total_num_frames,
    }
    if duration is not None:
        metadata["duration"] = duration
    return validate_frame_video_block(
        {
            "type": "video",
            "source": _source(source),
            "width": width,
            "height": height,
            "video_metadata": metadata,
        }
    )


def validate_frame_image_block(block: Mapping[str, Any]) -> dict[str, Any]:
    copied = deepcopy(dict(block))
    copied["source"] = _source(copied.get("source"))
    _size(copied)
    frame_idx = copied.get("frame_idx")
    if type(frame_idx) is not int or frame_idx < 0:
        raise ValueError("frame reference image requires a non-negative frame_idx")
    crop = copied.get("crop")
    if crop is not None and (
        not isinstance(crop, (list, tuple))
        or len(crop) != 4
        or any(type(value) is not int for value in crop)
        or not (0 <= crop[0] < crop[2] and 0 <= crop[1] < crop[3])
    ):
        raise ValueError("frame reference crop must be [left, top, right, bottom] pixels")
    if crop is not None:
        copied["crop"] = list(crop)
    return copied


def validate_frame_video_block(block: Mapping[str, Any]) -> dict[str, Any]:
    copied = deepcopy(dict(block))
    copied["source"] = _source(copied.get("source"))
    _size(copied)
    metadata = copied.get("video_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("frame reference video requires video_metadata")
    copied["video_metadata"] = validate_video_metadata(
        metadata, len(metadata.get("frames_indices") or ())
    )
    return copied


def _source(source: Any) -> dict[str, Any]:
    if not isinstance(source, Mapping) or not source:
        raise ValueError("frame reference source must be a non-empty mapping")
    return deepcopy(dict(source))


def _size(block: Mapping[str, Any]) -> None:
    for key in ("width", "height"):
        value = block.get(key)
        if type(value) is not int or value < 1:
            raise ValueError(f"frame reference block requires a positive integer {key}")


__all__ = [
    "FrameResolver",
    "block_data_urls",
    "data_url",
    "frame_image_block",
    "frame_video_block",
    "resolve_frames",
    "set_frame_resolver",
    "validate_frame_image_block",
    "validate_frame_video_block",
]
