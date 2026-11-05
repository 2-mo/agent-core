from __future__ import annotations

import pytest

from agent_core.runtime.policy import (
    LoopStatus,
    PolicyDecision,
    ToolPolicy,
    decide_loop_status,
    model_turn_limit,
    model_turn_limit_exceeded,
    should_retry_validation,
)


def test_tool_policy_validates_generic_modes() -> None:
    assert ToolPolicy(mode="auto").allowed_tools == ()
    assert ToolPolicy(
        mode="allow",
        allowed_tools=("observe", "observe"),
    ).allowed_tools == (
        "observe",
    )
    assert ToolPolicy(mode="force", allowed_tools=("final",)).allowed_tools == (
        "final",
    )

    with pytest.raises(ValueError, match="auto tool policy must not list tools"):
        ToolPolicy(mode="auto", allowed_tools=("observe",))
    with pytest.raises(ValueError, match="allow tool policy requires"):
        ToolPolicy(mode="allow")
    with pytest.raises(ValueError, match="force tool policy requires exactly one"):
        ToolPolicy(mode="force", allowed_tools=("a", "b"))


def test_decide_loop_status_continues_before_last_turn() -> None:
    decision = decide_loop_status(
        LoopStatus(finished=False, error=None, turn_count=0, max_turns=3),
        final_tool_name="final",
        forced_final_reason="finish now",
    )

    assert decision == PolicyDecision.continue_turn()


def test_decide_loop_status_forces_final_on_last_turn() -> None:
    decision = decide_loop_status(
        LoopStatus(finished=False, error=None, turn_count=2, max_turns=3),
        final_tool_name="final",
        forced_final_reason="finish now",
    )

    assert decision == PolicyDecision.force_final(
        tool_name="final",
        reason="finish now",
    )


def test_decide_loop_status_marks_exhausted_turn_budget_terminal() -> None:
    decision = decide_loop_status(
        LoopStatus(finished=False, error=None, turn_count=3, max_turns=3),
        final_tool_name="final",
        forced_final_reason="finish now",
        no_final_error="no_final",
        no_final_finish_reason="max_turns",
    )

    assert decision == PolicyDecision.terminal(
        terminal_reason="no_final",
        finish_reason="max_turns",
    )


def test_decide_loop_status_preserves_existing_terminal_state() -> None:
    assert decide_loop_status(
        LoopStatus(finished=True, error=None, turn_count=1, max_turns=3),
        final_tool_name="final",
        forced_final_reason="finish now",
    ) == PolicyDecision.terminal(
        terminal_reason="finished",
        finish_reason="finished",
    )
    assert decide_loop_status(
        LoopStatus(finished=False, error="tool_error", turn_count=1, max_turns=3),
        final_tool_name="final",
        forced_final_reason="finish now",
    ) == PolicyDecision.terminal(
        terminal_reason="tool_error",
        finish_reason="error",
    )


def test_should_retry_validation_uses_zero_based_attempts() -> None:
    assert should_retry_validation(attempt_index=0, max_retries=1) is True
    assert should_retry_validation(attempt_index=1, max_retries=1) is False

    with pytest.raises(ValueError, match="attempt_index"):
        should_retry_validation(attempt_index=-1, max_retries=1)
    with pytest.raises(ValueError, match="max_retries"):
        should_retry_validation(attempt_index=0, max_retries=-1)


def test_model_turn_limit_accounts_for_retries_and_terminal_buffer() -> None:
    assert model_turn_limit(max_turns=3, validation_retries=1) == 8
    assert (
        model_turn_limit(max_turns=3, validation_retries=1, extra_terminal_turns=0)
        == 6
    )
    assert (
        model_turn_limit_exceeded(
            current_turn=8,
            max_turns=3,
            validation_retries=1,
        )
        is False
    )
    assert (
        model_turn_limit_exceeded(
            current_turn=9,
            max_turns=3,
            validation_retries=1,
        )
        is True
    )

    with pytest.raises(ValueError, match="current_turn"):
        model_turn_limit_exceeded(
            current_turn=-1,
            max_turns=3,
            validation_retries=1,
        )
