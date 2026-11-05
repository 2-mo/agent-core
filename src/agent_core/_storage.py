"""Append complete trajectory records to a caller-selected JSONL file."""

import json
from pathlib import Path
from threading import Lock

try:
    import fcntl
except ImportError:
    fcntl = None

_append_lock = Lock()


def append_jsonl(path, value):
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(value, ensure_ascii=False, default=str) + "\n"
    with _append_lock, target.open("a", encoding="utf-8") as stream:
        if fcntl is not None:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            stream.write(line)
            stream.flush()
        finally:
            if fcntl is not None:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
