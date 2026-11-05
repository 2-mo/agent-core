# Adapted for standalone use in Agent Core.
"""Cooperative deadlines and shared accounting for an Agent run tree."""

from dataclasses import dataclass, field
from math import isfinite
from threading import Event, Lock
from time import perf_counter


class RunLimitError(RuntimeError):
    pass


def positive_seconds(name, value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be finite and positive")


@dataclass
class RunBudget:
    max_calls: int
    cancel: Event
    max_total_tokens: int | None = None
    deadline: float | None = None
    calls: int = 0
    tokens: int = 0
    missing_usage: bool = False
    lock: Lock = field(default_factory=Lock)
    parent: "RunBudget | None" = None

    def _lineage(self):
        budget = self
        while budget is not None:
            yield budget
            budget = budget.parent

    def _violation(self, *, admission=False):
        if self.cancel.is_set():
            return "cancelled"
        now = perf_counter()
        for budget in self._lineage():
            if budget.deadline is not None and now >= budget.deadline:
                return "deadline"
            if budget.max_total_tokens is not None:
                if budget.missing_usage:
                    return "usage_unavailable"
                if budget.tokens > budget.max_total_tokens or (
                    admission and budget.tokens >= budget.max_total_tokens
                ):
                    return "max_total_tokens"
        return None

    def check(self):
        with self.lock:
            reason = self._violation()
        if reason:
            raise RunLimitError(reason)

    def take(self):
        with self.lock:
            reason = self._violation(admission=True)
            if reason:
                raise RunLimitError(reason)
            lineage = list(self._lineage())
            if any(budget.calls >= budget.max_calls for budget in lineage):
                raise RunLimitError("max_calls")
            for budget in lineage:
                budget.calls += 1

    def record_usage(self, usage):
        value = (usage or {}).get("total_tokens")
        if (
            value is None
            and usage
            and all(
                isinstance(usage.get(k), int)
                for k in ("prompt_tokens", "completion_tokens")
            )
        ):
            value = usage["prompt_tokens"] + usage["completion_tokens"]
        known = isinstance(value, int) and not isinstance(value, bool) and value >= 0
        with self.lock:
            for budget in self._lineage():
                if known:
                    budget.tokens += value
                else:
                    budget.missing_usage = True

    def remaining_seconds(self):
        deadlines = [
            budget.deadline for budget in self._lineage() if budget.deadline is not None
        ]
        return max(0.001, min(deadlines) - perf_counter()) if deadlines else None

    def violation(self):
        with self.lock:
            return self._violation()

    def snapshot(self):
        with self.lock:
            return {
                "max_calls": self.max_calls,
                "model_calls": self.calls,
                "max_total_tokens": self.max_total_tokens,
                "reported_total_tokens": self.tokens,
                "usage_complete": not self.missing_usage,
                "token_overrun": max(0, self.tokens - self.max_total_tokens)
                if self.max_total_tokens is not None
                else 0,
            }
