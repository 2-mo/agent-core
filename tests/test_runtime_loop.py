from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from agent_core.context import Message, message_tool_calls, msg
from agent_core.model import CompletionMetadata
from agent_core.runtime import RuntimeHooks, Termination
from agent_core.runtime.runtime_loop import PreparedTurn, RuntimeLoop
from agent_core.runtime.turn_executor import ModelTurnAttempt


def test_runtime_loop_applies_ordinary_observation_turn() -> None:
    host = _Host(max_turns=2, responses=[_assistant_call("observe")])

    RuntimeLoop(host, model_client=host.model_client).run_model_turn()

    assert host.prepared_turns == [0]
    assert host.forced_final_turns == []
    assert host.applied == [("observe", 0, "model")]
    assert host.incomplete is None


def test_runtime_loop_forces_final_answer_on_last_turn() -> None:
    host = _Host(max_turns=2, responses=[_assistant_call("final_answer")])
    host.turn_count = 1

    RuntimeLoop(host, model_client=host.model_client).run_until_done()

    assert host.prepared_turns == []
    assert host.forced_final_turns == [(1, _FORCED_REASON)]
    assert host.applied == [("final_answer", 1, "model")]
    assert host.applied_policies == [("force", ("final_answer",))]
    assert host.applied_decisions == ["force_final"]
    assert host.finished is True
    assert host.incomplete is None


def test_runtime_loop_max_turns_one_is_forced_final() -> None:
    host = _Host(max_turns=1, responses=[_assistant_call("final_answer")])

    RuntimeLoop(host, model_client=host.model_client).run_until_done()

    assert host.prepared_turns == []
    assert host.forced_final_turns == [(0, _FORCED_REASON)]
    assert host.applied == [("final_answer", 0, "model")]
    assert host.finished is True
    assert host.incomplete is None


def test_runtime_loop_validation_retry_adds_correction_before_apply() -> None:
    host = _Host(
        max_turns=2,
        responses=[_assistant_call("bad_tool"), _assistant_call("observe")],
    )

    RuntimeLoop(host, model_client=host.model_client, validation_retries=1).run_model_turn()

    assert len(host.applied_attempts) == 1
    attempts = host.applied_attempts[0]
    assert len(attempts) == 2
    assert attempts[0].error == "ValueError: tool bad_tool is not allowed"
    assert attempts[1].error is None
    assert [message["role"] for message in host.applied_prefixes[0]] == ["user"]
    assert "tool bad_tool is not allowed" in _text(host.applied_prefixes[0][0])
    assert host.model_client.request_lengths == [1, 2]


def test_runtime_loop_model_error_records_terminal_error() -> None:
    host = _Host(max_turns=2, error=RuntimeError("backend down"))

    RuntimeLoop(host, model_client=host.model_client).run_model_turn()

    assert host.applied == []
    assert len(host.terminal_errors) == 1
    terminal = host.terminal_errors[0]
    assert terminal["finish_reason"] == "model_error"
    assert terminal["error"] == "RuntimeError: backend down"
    assert len(terminal["attempts"]) == 1
    assert terminal["message_delta"] == []
    assert host.error == "RuntimeError: backend down"


def test_runtime_loop_tool_execution_terminal_stops_loop() -> None:
    host = _Host(
        max_turns=2,
        responses=[_assistant_call("observe")],
        apply_error="ValueError: tool failed",
    )

    RuntimeLoop(host, model_client=host.model_client).run_until_done()

    assert host.applied == [("observe", 0, "model")]
    assert host.error == "ValueError: tool failed"
    assert host.incomplete is None
    assert len(host.terminal_errors) == 0


def test_runtime_hooks_observe_lifecycle_without_owning_runtime_truth() -> None:
    host = _Host(max_turns=1, responses=[_assistant_call("final_answer")])
    recorder = _Recorder()

    RuntimeLoop(
        host,
        model_client=host.model_client,
        hooks=RuntimeHooks((recorder,)),
    ).run_until_done()

    assert [name for name, _ in recorder.events] == [
        "context",
        "model",
        "action",
        "transition",
        "terminal",
    ]
    assert host.finished is True
    assert host.termination is not None
    assert host.termination.category == "semantic_success"


def test_failing_runtime_hook_isolated_from_execution() -> None:
    host = _Host(max_turns=1, responses=[_assistant_call("final_answer")])

    RuntimeLoop(
        host,
        model_client=host.model_client,
        hooks=RuntimeHooks((_BrokenHook(),)),
    ).run_until_done()

    assert host.finished is True


_FORCED_REASON = "Max turns reached without a valid final answer; return final now."


class _Recorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def on_event(self, event: str, payload: dict[str, Any]) -> None:
        self.events.append((event, payload))


class _BrokenHook:
    def on_event(self, event: str, payload: dict[str, Any]) -> None:
        del event, payload
        raise RuntimeError("telemetry failed")


@dataclass(slots=True)
class _Parsed:
    name: str


@dataclass(slots=True)
class _Host:
    max_turns: int
    responses: list[Message] = field(default_factory=list)
    error: BaseException | None = None
    apply_error: str | None = None
    turn_count: int = 0
    finished: bool = False
    model_client: Any = field(init=False)
    prepared_turns: list[int] = field(default_factory=list)
    forced_final_turns: list[tuple[int, str]] = field(default_factory=list)
    applied: list[tuple[str, int, str]] = field(default_factory=list)
    applied_attempts: list[list[ModelTurnAttempt]] = field(default_factory=list)
    applied_prefixes: list[list[Message]] = field(default_factory=list)
    applied_policies: list[tuple[str, tuple[str, ...]] | None] = field(
        default_factory=list
    )
    applied_decisions: list[str | None] = field(default_factory=list)
    terminal_errors: list[dict[str, Any]] = field(default_factory=list)
    incomplete: tuple[str, str] | None = None
    termination_fact: Termination | None = None

    def __post_init__(self) -> None:
        self.model_client = _Model(self.responses, error=self.error)

    @property
    def termination(self) -> Termination | None:
        return self.termination_fact

    def prepare_turn(self, turn_index: int) -> PreparedTurn:
        self.prepared_turns.append(turn_index)
        return _prepared(turn_index, ["observe"], forced=False)

    def prepare_forced_final_turn(
        self,
        turn_index: int,
        *,
        reason: str,
    ) -> PreparedTurn:
        self.forced_final_turns.append((turn_index, reason))
        return _prepared(turn_index, ["final_answer"], forced=True)

    def validate_tool_call(
        self,
        tool_calls: Sequence[dict[str, Any]],
        prepared: PreparedTurn,
    ) -> _Parsed:
        if len(tool_calls) != 1:
            raise ValueError(f"expected one tool call, got {len(tool_calls)}")
        name = str(tool_calls[0]["function"]["name"])
        if name not in prepared.available_tools:
            raise ValueError(f"tool {name} is not allowed")
        return _Parsed(name=name)

    def assistant_tool_call_payloads(
        self,
        assistant: Message,
    ) -> Sequence[dict[str, Any]]:
        return message_tool_calls(assistant)

    def normalize_assistant(self, assistant: Message, parsed: _Parsed) -> Message:
        del parsed
        return assistant

    def build_correction(self, error: str, prepared: PreparedTurn) -> Message:
        del prepared
        return msg.user(f"correct: {error}")

    def completion_kwargs(self, prepared: PreparedTurn) -> dict[str, Any]:
        del prepared
        return {"temperature": 0.0}

    def apply_tool_call(
        self,
        *,
        prepared: PreparedTurn,
        assistant_message: Message,
        parsed: _Parsed,
        attempts: list[ModelTurnAttempt],
        prefix_messages: Sequence[Message],
        origin: str = "model",
    ) -> Any:
        del assistant_message
        self.applied.append((parsed.name, prepared.turn_index, origin))
        self.applied_attempts.append(list(attempts))
        self.applied_prefixes.append(list(prefix_messages))
        self.applied_policies.append(
            None
            if prepared.policy is None
            else (prepared.policy.mode, prepared.policy.allowed_tools)
        )
        self.applied_decisions.append(
            None if prepared.decision is None else prepared.decision.status
        )
        self.turn_count += 1
        if self.apply_error is not None:
            self.error = self.apply_error
        elif parsed.name == "final_answer":
            self.finished = True
            self.mark_termination(
                Termination(
                    category="semantic_success",
                    reason="final tool accepted",
                    source="test.tool.final_answer",
                    step_index=prepared.turn_index,
                )
            )
        return {
            "step_index": prepared.turn_index,
            "action": {"tool": parsed.name},
            "ok": self.apply_error is None,
            "error": self.apply_error,
        }

    def mark_termination(self, termination: Termination) -> None:
        self.termination_fact = termination
        if termination.category == "runtime_error":
            self.error = termination.raw_error or termination.reason

    def record_terminal_error(
        self,
        *,
        prepared: PreparedTurn,
        attempts: list[ModelTurnAttempt],
        message_delta: Sequence[Message],
        error: str,
        finish_reason: str,
    ) -> None:
        self.terminal_errors.append(
            {
                "turn_index": prepared.turn_index,
                "attempts": list(attempts),
                "message_delta": list(message_delta),
                "error": error,
                "finish_reason": finish_reason,
            }
        )
        self.error = error
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


class _Model:
    def __init__(
        self,
        responses: list[Message],
        *,
        error: BaseException | None = None,
    ) -> None:
        self._responses = list(responses)
        self._error = error
        self.request_lengths: list[int] = []

    def _complete_with_metadata(self, messages: list[Message], **kwargs: Any) -> Any:
        del kwargs
        self.request_lengths.append(len(messages))
        if self._error is not None:
            raise self._error
        if not self._responses:
            raise RuntimeError("no scripted response")
        return CompletionMetadata(
            endpoint="test",
            model="scripted",
            responses=[self._responses.pop(0)],
            latency_sec=0.0,
        )


def _prepared(
    turn_index: int,
    available_tools: list[str],
    *,
    forced: bool,
) -> PreparedTurn:
    return PreparedTurn(
        turn_index=turn_index,
        available_tools=tuple(available_tools),
        messages=[msg.user("start")],
        prompt_delta=[],
        tool_specs=[
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"{name} tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            for name in available_tools
        ],
        tool_choice=(
            {
                "type": "function",
                "function": {"name": available_tools[0]},
            }
            if forced
            else "auto"
        ),
    )


def _assistant_call(name: str) -> Message:
    return msg.assistant(
        tool_calls=[
            {
                "id": f"call_{name}",
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }
        ]
    )


def _text(message: Message) -> str:
    return "\n".join(
        block["text"] for block in message["content"] if block["type"] == "text"
    )
