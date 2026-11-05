from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Sequence

from agent_core.context import Message, message_tool_calls, msg
from agent_core.model import CompletionMetadata
from agent_core.runtime import Termination
from agent_core.runtime.runtime_loop import PreparedTurn, RuntimeLoop
from agent_core.runtime.trajectory import TrajectoryRecord, TransitionRecord
from agent_core.runtime.turn_executor import ModelTurnAttempt


def test_toy_domain_records_a_canonical_trajectory() -> None:
    domain = ToyDomain()

    RuntimeLoop(
        domain,
        model_client=domain.model_client,
        final_tool_name="finish",
    ).run_until_done()

    assert domain.trajectory.finished is True
    assert domain.termination is not None
    assert domain.termination.category == "semantic_success"
    assert domain.trajectory.final_output == {"value": 1}
    assert [transition.kind for transition in domain.trajectory.transitions] == [
        "tool",
        "tool",
    ]
    assert domain.trajectory.transitions[0].state == {"value": 0}
    assert domain.trajectory.transitions[0].next_state == {"value": 1}
    assert domain.trajectory.transitions[1].action == {"tool": "finish"}


@dataclass(slots=True)
class ToyDomain:
    """Minimal RuntimeHost used to exercise the generic boundary."""

    state: int = 0
    trajectory: TrajectoryRecord = field(
        default_factory=lambda: TrajectoryRecord(
            rollout_id="toy-run",
            task_id="toy-task",
            domain="toy.counter",
            initial_messages=[msg.user("increment once, then finish")],
            messages=[msg.user("increment once, then finish")],
            initial_state={"value": 0},
            current_state={"value": 0},
        )
    )
    model_client: Any = field(default_factory=lambda: _ToyModel())

    @property
    def max_turns(self) -> int:
        return 2

    @property
    def turn_count(self) -> int:
        return len(self.trajectory.transitions)

    @property
    def finished(self) -> bool:
        return self.trajectory.finished

    @property
    def error(self) -> str | None:
        return self.trajectory.error

    @property
    def termination(self) -> Termination | None:
        value = self.trajectory.termination
        return None if value is None else Termination.from_dict(value)

    def prepare_turn(self, turn_index: int) -> PreparedTurn:
        return self._prepared(turn_index, tool_name="increment")

    def prepare_forced_final_turn(
        self,
        turn_index: int,
        *,
        reason: str,
    ) -> PreparedTurn:
        del reason
        return self._prepared(turn_index, tool_name="finish")

    def validate_tool_call(
        self,
        tool_calls: Sequence[dict[str, Any]],
        prepared: PreparedTurn,
    ) -> str:
        if len(tool_calls) != 1:
            raise ValueError("ToyDomain expects one tool call")
        name = str(tool_calls[0]["function"]["name"])
        if name not in prepared.available_tools:
            raise ValueError(f"ToyDomain does not allow {name}")
        return name

    def assistant_tool_call_payloads(
        self,
        assistant: Message,
    ) -> Sequence[dict[str, Any]]:
        return message_tool_calls(assistant)

    def normalize_assistant(self, assistant: Message, parsed: str) -> Message:
        del parsed
        return assistant

    def build_correction(self, error: str, prepared: PreparedTurn) -> Message:
        del prepared
        return msg.user(error)

    def completion_kwargs(self, prepared: PreparedTurn) -> dict[str, Any]:
        del prepared
        return {"temperature": 0.0}

    def apply_tool_call(
        self,
        *,
        prepared: PreparedTurn,
        assistant_message: Message,
        parsed: str,
        attempts: list[ModelTurnAttempt],
        prefix_messages: Sequence[Message],
        origin: str = "model",
    ) -> TransitionRecord:
        before = {"value": self.state}
        if parsed == "increment":
            self.state += 1
        elif parsed == "finish":
            pass
        else:  # pragma: no cover - validation rejects this first.
            raise ValueError(f"unknown toy tool {parsed}")
        after = {"value": self.state}
        transition = self.trajectory.add_transition(
            TransitionRecord(
                kind="tool",
                step_index=prepared.turn_index,
                state=before,
                action={"tool": parsed},
                model_action={
                    "assistant_message": deepcopy(assistant_message),
                    "attempt_count": len(attempts),
                    "origin": origin,
                },
                next_state=after,
                message_delta=[
                    *deepcopy(list(prefix_messages)),
                    deepcopy(assistant_message),
                    msg.tool(
                        tool_call_id=f"toy-{prepared.turn_index}",
                        name=parsed,
                        content=str(after),
                    ),
                ],
                ok=True,
            )
        )
        if parsed == "finish":
            self.trajectory.finished = True
            self.trajectory.finish_reason = "finish"
            self.trajectory.final_output = {"value": self.state}
            self.mark_termination(
                Termination(
                    category="semantic_success",
                    reason="counter was incremented before finish",
                    source="toy.tool.finish",
                    step_index=transition.step_index,
                )
            )
        return transition

    def mark_termination(self, termination: Termination) -> None:
        self.trajectory.termination = termination.to_dict()
        if termination.category == "runtime_error":
            self.trajectory.error = termination.raw_error or termination.reason
            self.trajectory.finish_reason = "runtime_error"
        elif termination.category == "hard_limit":
            self.trajectory.finish_reason = "max_turns"

    def record_terminal_error(
        self,
        *,
        prepared: PreparedTurn,
        attempts: list[ModelTurnAttempt],
        message_delta: Sequence[Message],
        error: str,
        finish_reason: str,
    ) -> None:
        self.trajectory.add_transition(
            TransitionRecord(
                kind="error",
                step_index=prepared.turn_index,
                state={"value": self.state},
                next_state={"value": self.state},
                model_action={"attempt_count": len(attempts)},
                message_delta=deepcopy(list(message_delta)),
                ok=False,
                error=error,
            )
        )
        self.trajectory.error = error
        self.trajectory.finish_reason = finish_reason
        self.mark_termination(
            Termination(
                category="runtime_error",
                reason="model turn failed",
                source="runtime.model",
                message=finish_reason,
                raw_error=error,
                step_index=prepared.turn_index,
            )
        )

    def _prepared(self, turn_index: int, *, tool_name: str) -> PreparedTurn:
        return PreparedTurn(
            turn_index=turn_index,
            available_tools=(tool_name,),
            messages=deepcopy(self.trajectory.messages),
            prompt_delta=[],
            tool_specs=[
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool_name,
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            tool_choice={"type": "function", "function": {"name": tool_name}},
        )


class _ToyModel:
    def __init__(self) -> None:
        self._calls = ["increment", "finish"]

    def _complete_with_metadata(self, messages: Sequence[Message], **kwargs: Any) -> CompletionMetadata:
        del messages, kwargs
        name = self._calls.pop(0)
        return CompletionMetadata(
            endpoint="toy://local",
            model="toy-model",
            responses=[
                msg.assistant(
                    tool_calls=[
                        {
                            "id": f"call-{name}",
                            "type": "function",
                            "function": {"name": name, "arguments": "{}"},
                        }
                    ]
                )
            ],
            latency_sec=0.0,
        )
