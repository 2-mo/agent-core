# Adapted for standalone use in Agent Core.
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from agent_core._storage import append_jsonl

TRAJECTORY_SCHEMA_VERSION = "agent_core.trajectory.v1"
TransitionKind = Literal[
    "model",
    "tool",
    "observation",
    "coordination",
    "error",
]


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class TransitionRecord:
    kind: TransitionKind
    transition_id: str = field(default_factory=lambda: uuid4().hex)
    step_index: int | None = None
    agent_id: str | None = None
    role: str | None = None
    # A transition is the sole record of a runtime fact. The older specialized
    # fields remain useful projections for generic model/tool clients; these
    # explicit fields are the stable state-graph contract used by domains.
    state: dict[str, Any] | None = None
    action: dict[str, Any] | None = None
    model_action: dict[str, Any] | None = None
    tool_action: dict[str, Any] | None = None
    observation: dict[str, Any] | None = None
    context: dict[str, Any] | None = None
    next_state: dict[str, Any] | None = None
    coordination: dict[str, Any] | None = None
    state_patch: dict[str, Any] = field(default_factory=dict)
    budget: dict[str, Any] = field(default_factory=dict)
    latency_sec: float | None = None
    endpoint: str | None = None
    model: str | None = None
    ok: bool | None = None
    error: str | None = None
    verification: dict[str, Any] | None = None
    message_delta: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(slots=True)
class TrajectoryRecord:
    rollout_id: str
    task_id: str
    trajectory_id: str = field(default_factory=lambda: uuid4().hex)
    schema_version: str = TRAJECTORY_SCHEMA_VERSION
    domain: str | None = None
    domain_metadata: dict[str, Any] = field(default_factory=dict)
    rollout_metadata: dict[str, Any] = field(default_factory=dict)
    goal: dict[str, Any] = field(default_factory=dict)
    sample_ref: dict[str, Any] | None = None
    observation_ref: dict[str, Any] | None = None
    input: dict[str, Any] = field(default_factory=dict)
    initial_messages: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    initial_state: dict[str, Any] = field(default_factory=dict)
    current_state: dict[str, Any] = field(default_factory=dict)
    transitions: list[TransitionRecord] = field(default_factory=list)
    final_output: dict[str, Any] | None = None
    finished: bool = False
    finish_reason: str | None = None
    error: str | None = None
    termination: dict[str, Any] | None = None
    artifact_refs: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    latency_sec: float | None = None
    parent_trajectory_id: str | None = None
    fork_after_transition: int | None = None

    def add_transition(self, transition: TransitionRecord) -> TransitionRecord:
        if transition.step_index is None:
            transition.step_index = len(self.transitions)
        if transition.step_index != len(self.transitions):
            raise ValueError(
                f"transition index {transition.step_index} does not follow "
                f"{len(self.transitions) - 1}"
            )
        self.transitions.append(transition)
        if transition.message_delta:
            self.messages.extend(deepcopy(transition.message_delta))
        if transition.next_state is not None:
            self.current_state = deepcopy(transition.next_state)
        return transition

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)

    def to_shared_dict(self) -> dict[str, Any]:
        """Serialize per-sample data once for compact JSONL workflow output."""
        return _jsonable(
            {
                "schema_version": self.schema_version,
                "task_id": self.task_id,
                "domain": self.domain,
                "domain_metadata": self.domain_metadata,
                "goal": self.goal,
                "sample_ref": self.sample_ref,
                "observation_ref": self.observation_ref,
                "input": self.input,
                "initial_messages": self.initial_messages,
                "initial_state": self.initial_state,
            }
        )

    def to_rollout_dict(self) -> dict[str, Any]:
        """Serialize rollout-specific trace facts without duplicating the header."""
        return _jsonable(
            {
                "trajectory_id": self.trajectory_id,
                "rollout_metadata": self.rollout_metadata,
                "rollout_id": self.rollout_id,
                "messages": self.messages,
                "transitions": self.transitions,
                "current_state": self.current_state,
                "final_output": self.final_output,
                "finished": self.finished,
                "finish_reason": self.finish_reason,
                "error": self.error,
                "termination": self.termination,
                "artifact_refs": self.artifact_refs,
                "created_at": self.created_at,
                "latency_sec": self.latency_sec,
                "parent_trajectory_id": self.parent_trajectory_id,
                "fork_after_transition": self.fork_after_transition,
            }
        )

    @classmethod
    def from_parts(
        cls,
        shared: Mapping[str, Any],
        rollout: Mapping[str, Any],
    ) -> TrajectoryRecord:
        return cls.from_dict({**shared, **rollout})

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TrajectoryRecord:
        transitions = [
            TransitionRecord(**dict(item))
            for item in value.get("transitions") or []
            if isinstance(item, Mapping)
        ]
        initial_messages = _dict_list(value.get("initial_messages"))
        messages = _dict_list(value.get("messages"))
        if not messages:
            messages = deepcopy(initial_messages)
            for transition in transitions:
                messages.extend(deepcopy(transition.message_delta))
        current_state = _mapping_dict(value.get("current_state"))
        if not current_state and transitions and transitions[-1].next_state is not None:
            current_state = deepcopy(transitions[-1].next_state)
        return cls(
            rollout_id=str(value["rollout_id"]),
            task_id=str(value["task_id"]),
            trajectory_id=str(value.get("trajectory_id") or uuid4().hex),
            schema_version=str(value.get("schema_version") or TRAJECTORY_SCHEMA_VERSION),
            domain=_optional_str(value.get("domain")),
            domain_metadata=_mapping_dict(value.get("domain_metadata")),
            rollout_metadata=_mapping_dict(value.get("rollout_metadata")),
            goal=_mapping_dict(value.get("goal")),
            sample_ref=_optional_mapping_dict(value.get("sample_ref")),
            observation_ref=_optional_mapping_dict(value.get("observation_ref")),
            input=_mapping_dict(value.get("input")),
            initial_messages=initial_messages,
            messages=messages,
            initial_state=_mapping_dict(value.get("initial_state")),
            current_state=current_state,
            transitions=transitions,
            final_output=_optional_mapping_dict(value.get("final_output")),
            finished=bool(value.get("finished", False)),
            finish_reason=_optional_str(value.get("finish_reason")),
            error=_optional_str(value.get("error")),
            termination=_optional_mapping_dict(value.get("termination")),
            artifact_refs=_dict_list(value.get("artifact_refs")),
            created_at=str(value.get("created_at") or utc_now()),
            latency_sec=_optional_float(value.get("latency_sec")),
            parent_trajectory_id=_optional_str(value.get("parent_trajectory_id")),
            fork_after_transition=(
                None
                if value.get("fork_after_transition") is None
                else int(value["fork_after_transition"])
            ),
        )

    def fork(
        self,
        *,
        after_transition: int,
        rollout_id: str | None = None,
    ) -> TrajectoryRecord:
        after_transition = int(after_transition)
        if after_transition < -1 or after_transition >= len(self.transitions):
            raise ValueError("after_transition does not identify an existing transition")
        selected = self.transitions[after_transition] if after_transition >= 0 else None
        if selected is not None and (selected.next_state is None or selected.ok is False):
            raise ValueError("fork point must be a completed transition")
        if selected is not None and self.finished and after_transition == len(self.transitions) - 1:
            raise ValueError("cannot fork after a terminal transition")
        prefix = deepcopy(self.transitions[: after_transition + 1])
        messages = deepcopy(self.initial_messages)
        for transition in prefix:
            messages.extend(deepcopy(transition.message_delta))
        return TrajectoryRecord(
            rollout_id=(
                rollout_id
                or f"{self.rollout_id}:fork-{after_transition:03d}"
            ),
            task_id=self.task_id,
            domain=self.domain,
            domain_metadata=deepcopy(self.domain_metadata),
            rollout_metadata=deepcopy(self.rollout_metadata),
            goal=deepcopy(self.goal),
            sample_ref=deepcopy(self.sample_ref),
            observation_ref=deepcopy(self.observation_ref),
            input=deepcopy(self.input),
            initial_messages=deepcopy(self.initial_messages),
            messages=messages,
            initial_state=deepcopy(self.initial_state),
            current_state=deepcopy(self.initial_state if selected is None else selected.next_state),
            transitions=prefix,
            parent_trajectory_id=self.trajectory_id,
            fork_after_transition=after_transition,
        )


def _mapping_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _optional_mapping_dict(value: Any) -> dict[str, Any] | None:
    return None if value is None else _mapping_dict(value)


def _dict_list(value: Any) -> list[dict[str, Any]]:
    return [dict(item) for item in value or [] if isinstance(item, Mapping)]


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def transition_from_step(
    step: dict[str, Any],
    *,
    agent_id: str | None = None,
    role: str | None = None,
) -> TransitionRecord:
    step_type = str(step.get("type", "model"))
    if step_type == "model_call":
        return TransitionRecord(
            kind="model",
            step_index=step.get("step_index"),
            agent_id=agent_id,
            role=role,
            model_action={
                "request_window": step.get("request_window"),
                "assistant_message": step.get("assistant_message"),
            },
            latency_sec=step.get("latency_sec"),
            endpoint=step.get("endpoint"),
            model=step.get("model"),
            ok=step.get("ok"),
            error=step.get("error"),
            message_delta=step.get("message_delta") or [],
        )
    if step_type == "tool_step":
        tool_result = step.get("tool_result") or {}
        state_patch = step.get("state_delta") or {}
        return TransitionRecord(
            kind="tool",
            step_index=step.get("step_index"),
            agent_id=agent_id,
            role=role,
            model_action={
                "request_window": step.get("request_window"),
                "assistant_message": step.get("assistant_message"),
            },
            tool_action={
                "tool_call_id": step.get("tool_call_id"),
                "tool_name": step.get("tool_name"),
                "forced_tool_name": step.get("forced_tool_name"),
                "arguments": step.get("arguments"),
                "result": tool_result,
            },
            observation={
                "content": tool_result.get("content") if isinstance(tool_result, dict) else None,
                "raw_tool_result": step.get("raw_tool_result"),
            },
            state_patch=state_patch,
            budget={"cost": step.get("cost")},
            latency_sec=step.get("latency_sec"),
            endpoint=step.get("endpoint"),
            model=step.get("model"),
            ok=step.get("ok"),
            error=step.get("error"),
            message_delta=step.get("message_delta") or [],
        )
    return TransitionRecord(
        kind="error" if step.get("error") else "observation",
        step_index=step.get("step_index"),
        agent_id=agent_id,
        role=role,
        observation=dict(step),
        ok=step.get("ok"),
        error=step.get("error"),
    )


def trajectory_from_task_record_fields(
    *,
    rollout_id: str,
    task_id: str,
    input_payload: dict[str, Any],
    steps: list[dict[str, Any]],
    final_output: Any,
    error: str | None,
    sample_ref: dict[str, Any] | None = None,
    observation_ref: dict[str, Any] | None = None,
    latency_sec: float | None = None,
    agent_id: str | None = None,
    role: str | None = None,
) -> TrajectoryRecord:
    trajectory = TrajectoryRecord(
        rollout_id=rollout_id,
        task_id=task_id,
        sample_ref=sample_ref,
        observation_ref=observation_ref,
        input=_jsonable(input_payload),
        messages=_jsonable(input_payload.get("messages", [])),
        final_output=_jsonable(final_output),
        error=error,
        latency_sec=latency_sec,
    )
    for step in steps:
        trajectory.add_transition(transition_from_step(step, agent_id=agent_id, role=role))
    if error and not steps:
        trajectory.add_transition(TransitionRecord(kind="error", error=error))
    return trajectory


class TrajectoryRecorder:
    def __init__(self, path: str) -> None:
        self.path = path

    def write(self, trajectory: TrajectoryRecord) -> None:
        append_jsonl(self.path, trajectory.to_dict())


__all__ = [
    "TRAJECTORY_SCHEMA_VERSION",
    "TrajectoryRecord",
    "TrajectoryRecorder",
    "TransitionKind",
    "TransitionRecord",
    "trajectory_from_task_record_fields",
    "transition_from_step",
]
