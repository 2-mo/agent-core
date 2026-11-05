# Adapted for standalone use in Agent Core.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping


VerificationStatus = Literal["CONTINUE", "REVISE", "COMPLETE", "FAIL"]
TerminationCategory = Literal[
    "semantic_success",
    "semantic_failure",
    "hard_limit",
    "runtime_error",
]


@dataclass(frozen=True, slots=True)
class Goal:
    """Domain-owned task contract used by the generic runtime."""

    goal_id: str
    description: str
    success_criterion: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.goal_id.strip():
            raise ValueError("goal_id must not be empty")
        if not self.description.strip():
            raise ValueError("goal description must not be empty")
        if not self.success_criterion.strip():
            raise ValueError("success criterion must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "description": self.description,
            "success_criterion": self.success_criterion,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Goal":
        return cls(
            goal_id=str(value["goal_id"]),
            description=str(value["description"]),
            success_criterion=str(value["success_criterion"]),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Domain decision about a recorded transition."""

    status: VerificationStatus
    reason: str
    feedback: str | None = None
    source: str = "domain"

    def __post_init__(self) -> None:
        if self.status not in {"CONTINUE", "REVISE", "COMPLETE", "FAIL"}:
            raise ValueError(f"unknown verification status: {self.status!r}")
        if not self.reason.strip():
            raise ValueError("verification reason must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "feedback": self.feedback,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VerificationResult":
        return cls(
            status=str(value["status"]),  # type: ignore[arg-type]
            reason=str(value["reason"]),
            feedback=None if value.get("feedback") is None else str(value["feedback"]),
            source=str(value.get("source") or "domain"),
        )


@dataclass(frozen=True, slots=True)
class Termination:
    """Structured end-of-run fact, separate from raw runtime errors."""

    category: TerminationCategory
    reason: str
    source: str
    message: str | None = None
    raw_error: str | None = None
    step_index: int | None = None

    def __post_init__(self) -> None:
        if self.category not in {
            "semantic_success",
            "semantic_failure",
            "hard_limit",
            "runtime_error",
        }:
            raise ValueError(f"unknown termination category: {self.category!r}")
        if not self.reason.strip():
            raise ValueError("termination reason must not be empty")
        if self.step_index is not None and self.step_index < 0:
            raise ValueError("termination step_index must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "reason": self.reason,
            "source": self.source,
            "message": self.message,
            "raw_error": self.raw_error,
            "step_index": self.step_index,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Termination":
        return cls(
            category=str(value["category"]),  # type: ignore[arg-type]
            reason=str(value["reason"]),
            source=str(value.get("source") or "runtime"),
            message=None if value.get("message") is None else str(value["message"]),
            raw_error=None if value.get("raw_error") is None else str(value["raw_error"]),
            step_index=(
                None if value.get("step_index") is None else int(value["step_index"])
            ),
        )


__all__ = [
    "Goal",
    "Termination",
    "TerminationCategory",
    "VerificationResult",
    "VerificationStatus",
]
