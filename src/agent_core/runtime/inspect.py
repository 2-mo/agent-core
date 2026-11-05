# Adapted for standalone use in Agent Core.
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .trajectory import TrajectoryRecord


def inspect_trajectory(
    trajectory: TrajectoryRecord | Mapping[str, Any],
) -> dict[str, Any]:
    """Return a concise human-facing projection of canonical trace facts."""

    trace = (
        trajectory.to_dict()
        if isinstance(trajectory, TrajectoryRecord)
        else deepcopy(dict(trajectory))
    )
    transitions = [
        dict(item)
        for item in trace.get("transitions") or []
        if isinstance(item, Mapping)
    ]
    return {
        "run": {
            "rollout_id": trace.get("rollout_id"),
            "task_id": trace.get("task_id"),
            "trajectory_id": trace.get("trajectory_id"),
            "domain": trace.get("domain"),
            "transition_count": len(transitions),
        },
        "goal": _mapping_or_none(trace.get("goal")),
        "context": [
            {
                "step_index": transition.get("step_index", index),
                "summary": _mapping_or_none(transition.get("context")),
            }
            for index, transition in enumerate(transitions)
            if transition.get("context") is not None
        ],
        "steps": [
            _step_view(transition, index=index) for index, transition in enumerate(transitions)
        ],
        "termination": _mapping_or_none(trace.get("termination")),
        "final_output": _mapping_or_none(trace.get("final_output")),
    }


def _step_view(transition: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    verification = _mapping_or_none(transition.get("verification"))
    next_state = _mapping_or_none(transition.get("next_state"))
    return {
        "step_index": transition.get("step_index", index),
        "action": _mapping_or_none(transition.get("tool_action"))
        or _mapping_or_none(transition.get("action")),
        "observation": _mapping_or_none(transition.get("observation")),
        "state_diff": _state_diff(
            _mapping_or_none(transition.get("state")) or {},
            next_state or {},
        ),
        "context": _mapping_or_none(transition.get("context")),
        "verification": verification,
        "budget": _mapping_or_none(transition.get("budget"))
        or _mapping_or_none((next_state or {}).get("budget")),
        "error": transition.get("error"),
        "recovery": None if verification is None else verification.get("feedback"),
    }


def _state_diff(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    _collect_state_diff(before, after, path="", changes=changes)
    return changes


def _collect_state_diff(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    path: str,
    changes: list[dict[str, Any]],
) -> None:
    for key in sorted(set(before) | set(after)):
        child_path = f"{path}.{key}" if path else str(key)
        previous = before.get(key)
        current = after.get(key)
        if isinstance(previous, Mapping) and isinstance(current, Mapping):
            _collect_state_diff(previous, current, path=child_path, changes=changes)
        elif previous != current:
            changes.append(
                {
                    "path": child_path,
                    "before": deepcopy(previous),
                    "after": deepcopy(current),
                }
            )


def _mapping_or_none(value: Any) -> dict[str, Any] | None:
    return None if not isinstance(value, Mapping) else deepcopy(dict(value))


__all__ = ["inspect_trajectory"]
