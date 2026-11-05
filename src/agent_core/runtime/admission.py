# Adapted for standalone use in Agent Core.
"""Per-model HTTP slots shared by all runs and delegates in one batch."""

from threading import BoundedSemaphore, Lock


class RequestAdmission:
    def __init__(self, concurrency="auto"):
        self.concurrency = concurrency
        self._lock = Lock()
        self._entries = {}

    def _entry(self, client, capacity=None):
        config = getattr(client, "cfg", getattr(client, "_config", None))
        endpoints = tuple(
            getattr(client, "_endpoints", (getattr(client, "endpoint", None),))
        )
        model = getattr(client, "model", getattr(config, "model", None))
        key = (endpoints, model) if any(endpoints) else id(client)
        with self._lock:
            if key not in self._entries:
                info = (
                    (client.capacity() if capacity is None else capacity)
                    if self.concurrency == "auto"
                    else {
                        "recommended_concurrency": self.concurrency,
                        "source": "explicit",
                    }
                )
                limit = max(1, info["recommended_concurrency"])
                self._entries[key] = {
                    "model": model,
                    "limit": limit,
                    "source": info["source"],
                    "semaphore": BoundedSemaphore(limit),
                    "active": 0,
                    "peak": 0,
                }
            return self._entries[key]

    def register(self, client, capacity):
        return self._entry(client, capacity)["limit"]

    def acquire(self, client, check):
        entry = self._entry(client)
        check()
        while not entry["semaphore"].acquire(timeout=0.05):
            check()
        try:
            check()
        except BaseException:
            entry["semaphore"].release()
            raise
        with self._lock:
            entry["active"] += 1
            entry["peak"] = max(entry["peak"], entry["active"])
        return entry

    def release(self, entry):
        with self._lock:
            entry["active"] -= 1
        entry["semaphore"].release()

    def snapshot(self):
        with self._lock:
            return [
                {key: entry[key] for key in ("model", "limit", "source", "peak")}
                for entry in self._entries.values()
            ]
