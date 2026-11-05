# Adapted for standalone use in Agent Core.
"""An ordered set of decoded frames with original source-video metadata."""

from __future__ import annotations

import base64
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from .blocks import canonical_image_path

VIDEO_MEDIA_TYPE = "video/agent-frames+json"


def validate_video_metadata(
    metadata: Mapping[str, Any], frame_count: int
) -> dict[str, Any]:
    fps = metadata.get("fps")
    count = metadata.get("total_num_frames")
    indices = metadata.get("frames_indices")
    if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("video fps must be the positive finite original FPS")
    if type(count) is not int or count < 1:
        raise ValueError("video total_num_frames must be a positive integer")
    if (
        not isinstance(indices, (list, tuple))
        or len(indices) != frame_count
        or frame_count < 1
    ):
        raise ValueError("frames_indices must match the ordered video frames")
    if any(type(i) is not int or not 0 <= i < count for i in indices) or any(
        a >= b for a, b in itertools.pairwise(indices)
    ):
        raise ValueError(
            "frames_indices must be unique, increasing native frame indices"
        )
    duration = metadata.get("duration", count / fps)
    if (
        type(duration) not in (int, float)
        or not math.isfinite(duration)
        or duration <= (count - 1) / fps
    ):
        raise ValueError("video duration must include the last native frame")
    return {
        "fps": float(fps),
        "frames_indices": list(indices),
        "total_num_frames": count,
        "duration": float(duration),
    }


def video_block(
    frames: Sequence[str],
    *,
    fps: float,
    frames_indices: Sequence[int],
    total_num_frames: int,
    duration: float | None = None,
) -> dict[str, Any]:
    if isinstance(frames, (str, bytes)):
        raise TypeError("video frames must be ordered paths")
    metadata = {
        "fps": fps,
        "frames_indices": list(frames_indices),
        "total_num_frames": total_num_frames,
    }
    if duration is not None:
        metadata["duration"] = duration
    metadata = validate_video_metadata(metadata, len(frames))
    return {
        "type": "video",
        "frames": [canonical_image_path(path) for path in frames],
        "video_metadata": metadata,
    }


def encode_video_url(frame_urls: Sequence[str], metadata: Mapping[str, Any]) -> str:
    """Self-contained video transport: original image bytes plus exact metadata."""
    payload = {
        "frames": list(frame_urls),
        "video_metadata": validate_video_metadata(metadata, len(frame_urls)),
    }
    data = base64.b64encode(
        json.dumps(payload, separators=(",", ":")).encode()
    ).decode()
    return f"data:{VIDEO_MEDIA_TYPE};base64,{data}"


def decode_video_payload(data: str) -> tuple[list[str], dict[str, Any]]:
    payload = json.loads(base64.b64decode(data, validate=True))
    frames = payload["frames"]
    if not isinstance(frames, list) or any(
        not isinstance(frame, str) or not frame.startswith("data:image/")
        for frame in frames
    ):
        raise ValueError("video transport requires embedded image data URLs")
    return frames, validate_video_metadata(payload["video_metadata"], len(frames))
