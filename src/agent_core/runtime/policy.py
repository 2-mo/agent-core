# Adapted for standalone use in Agent Core.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .goal import Termination


ToolPolicyMode = Literal["auto", "allow", "force"]
LoopStatusMode = Literal["continue", "force_final", "terminal"]


@dataclass(frozen=True, slots=True)
class ToolPolicy:
    """Generic per-turn tool policy.

    ``allowed_tools`` contains executor-visible tool names only. Domain adapters
    decide what those names mean and how they are rendered in prompts.
    """

    mode: ToolPolicyMode = "auto"
    allowed_tools: tuple[str, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        if self.mode not in {"auto", "allow", "force"}:
            raise ValueError("tool policy mode must be auto, allow, or force")
        allowed_tools = tuple(dict.fromkeys(str(item) for item in self.allowed_tools))
        if self.mode == "auto" and allowed_tools:
            raise ValueError("auto tool policy must not list tools")
        if self.mode == "allow" and not allowed_tools:
            raise ValueError("allow tool policy requires at least one tool")
        if self.mode == "force" and len(allowed_tools) != 1:
            raise ValueError("force tool policy requires exactly one tool")
        object.__setattr__(self, "allowed_tools", allowed_tools)


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """A controller decision for the next model/tool turn."""

    status: LoopStatusMode
    policy: ToolPolicy | None = None
    reason: str = ""
    terminal_reason: str | None = None
    finish_reason: str | None = None
    termination: Termination | None = field(default=None, compare=False)

    @classmethod
    def continue_turn(cls) -> PolicyDecision:
        return cls(status="continue")

    @classmethod
    def force_final(cls, *, tool_name: str, reason: str) -> PolicyDecision:
        return cls(
            status="force_final",
            policy=ToolPolicy(mode="force", allowed_tools=(tool_name,), reason=reason),
            reason=reason,
        )

    @classmethod
    def terminal(
        cls,
        *,
        terminal_reason: str,
        finish_reason: str,
        termination: Termination | None = None,
    ) -> PolicyDecision:
        return cls(
            status="terminal",
            terminal_reason=terminal_reason,
            finish_reason=finish_reason,
            termination=termination,
        )


@dataclass(frozen=True, slots=True)
class LoopStatus:
    """Current generic loop state used by the tool-policy controller."""

    finished: bool
    error: str | None
    turn_count: int
    max_turns: int
    termination: Termination | None = None

    def __post_init__(self) -> None:
        if self.turn_count < 0:
            raise ValueError("turn_count must be >= 0")
        if self.max_turns < 1:
            raise ValueError("max_turns must be at least 1")


def decide_loop_status(
    status: LoopStatus,
    *,
    final_tool_name: str,
    forced_final_reason: str,
    no_final_error: str = "max_turns_without_valid_final_answer",
    no_final_finish_reason: str = "max_turns",
    force_final: bool = True,
) -> PolicyDecision:
    """Pure controller for tool-policy continuation and terminal states."""

    if status.termination is not None:
        return PolicyDecision.terminal(
            terminal_reason=status.termination.reason,
            finish_reason=status.termination.category,
            termination=status.termination,
        )
    if status.finished or status.error is not None:
        termination = (
            Termination(
                category="semantic_success",
                reason="domain marked the goal complete",
                source="runtime",
            )
            if status.finished
            else Termination(
                category="runtime_error",
                reason="runtime host reported an error",
                source="runtime",
                message=status.error,
                raw_error=status.error,
            )
        )
        return PolicyDecision.terminal(
            terminal_reason=status.error or "finished",
            finish_reason="finished" if status.finished else "error",
            termination=termination,
        )
    if status.turn_count >= status.max_turns:
        return PolicyDecision.terminal(
            terminal_reason=no_final_error,
            finish_reason=no_final_finish_reason,
            termination=Termination(
                category="hard_limit",
                reason=no_final_error,
                source="runtime",
                message=no_final_finish_reason,
                step_index=status.turn_count - 1 if status.turn_count else None,
            ),
        )
    if force_final and status.turn_count == status.max_turns - 1:
        return PolicyDecision.force_final(
            tool_name=final_tool_name,
            reason=forced_final_reason,
        )
    return PolicyDecision.continue_turn()


def should_retry_validation(*, attempt_index: int, max_retries: int) -> bool:
    """Return whether a validation failure should receive another model attempt."""

    if attempt_index < 0:
        raise ValueError("attempt_index must be >= 0")
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")
    return int(attempt_index) < int(max_retries)


def model_turn_limit(
    *,
    max_turns: int,
    validation_retries: int,
    extra_terminal_turns: int = 1,
) -> int:
    """Upper bound for scheduler-level model turns in retrying tool loops."""

    if max_turns < 1:
        raise ValueError("max_turns must be at least 1")
    if validation_retries < 0:
        raise ValueError("validation_retries must be >= 0")
    if extra_terminal_turns < 0:
        raise ValueError("extra_terminal_turns must be >= 0")
    return (int(max_turns) + int(extra_terminal_turns)) * (
        int(validation_retries) + 1
    )


def model_turn_limit_exceeded(
    *,
    current_turn: int,
    max_turns: int,
    validation_retries: int,
    extra_terminal_turns: int = 1,
) -> bool:
    """Return whether a scheduler callback is past the supported turn bound."""

    if current_turn < 0:
        raise ValueError("current_turn must be >= 0")
    return int(current_turn) > model_turn_limit(
        max_turns=max_turns,
        validation_retries=validation_retries,
        extra_terminal_turns=extra_terminal_turns,
    )


__all__ = [
    "LoopStatus",
    "PolicyDecision",
    "ToolPolicy",
    "ToolPolicyMode",
    "decide_loop_status",
    "model_turn_limit",
    "model_turn_limit_exceeded",
    "should_retry_validation",
]
