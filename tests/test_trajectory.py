from __future__ import annotations

from agent_core.runtime.trajectory import TrajectoryRecord, TransitionRecord


def test_canonical_trace_round_trips_and_forks_from_transition_state() -> None:
    trace = TrajectoryRecord(
        rollout_id="run-1",
        task_id="task-1",
        domain="toy",
        domain_metadata={"label": "test"},
        initial_messages=[{"role": "user", "content": "start"}],
        messages=[{"role": "user", "content": "start"}],
        initial_state={"count": 0},
        current_state={"count": 0},
    )
    trace.add_transition(
        TransitionRecord(
            kind="tool",
            state={"count": 0},
            action={"name": "increment"},
            observation={"value": 1},
            next_state={"count": 1},
            message_delta=[{"role": "tool", "content": "1"}],
            ok=True,
        )
    )

    restored = TrajectoryRecord.from_parts(
        trace.to_shared_dict(),
        trace.to_rollout_dict(),
    )
    child = restored.fork(after_transition=0, rollout_id="run-1:fork")

    assert restored.current_state == {"count": 1}
    assert restored.transitions[0].action == {"name": "increment"}
    assert restored.transitions[0].observation == {"value": 1}
    assert child.parent_trajectory_id == restored.trajectory_id
    assert child.current_state == {"count": 1}
    assert child.messages[-1] == {"role": "tool", "content": "1"}
