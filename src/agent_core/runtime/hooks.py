# Adapted for standalone use in Agent Core.
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


class RuntimeHook(Protocol):
    """Minimal observer contract for runtime lifecycle events."""

    def on_event(self, event: str, payload: Mapping[str, Any]) -> None: ...


@dataclass(slots=True)
class RuntimeHooks:
    """Optional observers; they never participate in runtime decisions."""

    observers: tuple[RuntimeHook, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.observers = tuple(self.observers)

    @property
    def enabled(self) -> bool:
        return bool(self.observers)

    def emit(self, event: str, payload: Mapping[str, Any] | None = None, **values: Any) -> None:
        if not self.observers:
            return
        data = dict(payload or {})
        data.update(values)
        for observer in self.observers:
            try:
                observer.on_event(event, deepcopy(data))
            except Exception:
                # Telemetry must never change runtime semantics or termination.
                continue


__all__ = ["RuntimeHook", "RuntimeHooks"]
