from __future__ import annotations

from agent_core.context import ContextBuilder, msg


def test_context_builder_creates_immutable_non_domain_snapshot() -> None:
    messages = [msg.user("inspect this")]
    goal = {"goal_id": "toy", "description": "inspect", "success_criterion": "done"}
    state = {"counter": 1}
    trajectory = {"transition_count": 2}

    context = ContextBuilder().build(
        messages=messages,
        goal=goal,
        state=state,
        trajectory=trajectory,
        observation={"value": "seen"},
    )
    messages[0]["content"][0]["text"] = "mutated"
    state["counter"] = 99

    assert context.to_messages()[0]["content"][0]["text"] == "inspect this"
    assert context.state == {"counter": 1}
    assert context.stats is not None
    assert context.stats.source_names == (
        "goal",
        "state",
        "trajectory",
        "observation",
    )
    assert context.inspect()["stats"]["message_count"] == 1
