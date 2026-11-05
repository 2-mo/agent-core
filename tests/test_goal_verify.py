from __future__ import annotations

from agent_core.runtime import Goal, Termination, VerificationResult, inspect_trajectory
from agent_core.runtime.trajectory import TrajectoryRecord, TransitionRecord


def test_goal_verification_and_termination_are_structured_and_round_trip() -> None:
    goal = Goal(
        goal_id="toy.answer",
        description="Answer the toy task.",
        success_criterion="the final answer is valid",
        metadata={"kind": "synthetic"},
    )
    verification = VerificationResult(
        status="REVISE",
        reason="answer failed validation",
        feedback="Please provide a non-empty answer.",
        source="toy.validator",
    )
    termination = Termination(
        category="hard_limit",
        reason="max_turns_without_valid_final_answer",
        source="runtime",
        message="max_turns",
        step_index=2,
    )
    transition = TransitionRecord(
        kind="tool",
        step_index=2,
        state={"value": 1},
        next_state={"value": 1},
        verification=verification.to_dict(),
        ok=True,
    )
    trace = TrajectoryRecord(
        rollout_id="run",
        task_id="task",
        goal=goal.to_dict(),
        transitions=[transition],
        termination=termination.to_dict(),
    )

    restored = TrajectoryRecord.from_dict(trace.to_dict())

    assert restored.goal == goal.to_dict()
    assert restored.transitions[0].verification == verification.to_dict()
    assert restored.termination == termination.to_dict()
    assert restored.to_shared_dict()["goal"] == goal.to_dict()
    assert restored.to_rollout_dict()["termination"] == termination.to_dict()


def test_inspection_view_derives_human_facing_facts_without_writing_trace() -> None:
    trace = TrajectoryRecord(
        rollout_id="run",
        task_id="task",
        goal=Goal("toy", "finish toy task", "valid final answer").to_dict(),
        termination=Termination(
            category="semantic_success",
            reason="accepted",
            source="toy",
            step_index=0,
        ).to_dict(),
    )
    trace.add_transition(
        TransitionRecord(
            kind="tool",
            state={"budget": {"used_turns": 0}},
            tool_action={"name": "finish"},
            observation={"content": "accepted"},
            next_state={"budget": {"used_turns": 1}},
            context={"stats": {"message_count": 2}},
            verification=VerificationResult(
                status="COMPLETE", reason="accepted", source="toy"
            ).to_dict(),
            ok=True,
        )
    )

    view = inspect_trajectory(trace)

    assert view["goal"]["goal_id"] == "toy"
    assert view["context"][0]["summary"]["stats"]["message_count"] == 2
    assert view["steps"][0]["action"] == {"name": "finish"}
    assert view["steps"][0]["state_diff"] == [
        {"path": "budget.used_turns", "before": 0, "after": 1}
    ]
    assert view["steps"][0]["verification"]["status"] == "COMPLETE"
    assert view["termination"]["category"] == "semantic_success"
