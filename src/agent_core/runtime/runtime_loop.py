# Adapted for standalone use in Agent Core.
from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from agent_core.context import Message, ModelContext

from .goal import Termination
from .hooks import RuntimeHooks
from .policy import LoopStatus, PolicyDecision, ToolPolicy, decide_loop_status
from .turn_executor import (
    ModelTurnAttempt,
    ModelTurnExecutor,
    ParsedSerializer,
    RequestStatsBuilder,
)


@dataclass(slots=True)
class PreparedTurn:
    turn_index: int
    available_tools: tuple[str, ...]
    messages: list[Message]
    prompt_delta: list[Message]
    tool_specs: list[dict[str, Any]]
    model_context: ModelContext | None = None
    tool_choice: Any = "auto"
    policy: ToolPolicy | None = None
    decision: PolicyDecision | None = None


class RuntimeHost(Protocol):
    """Domain boundary for the canonical generic runtime.

    The host prepares model prompts/tool specs for a controller decision and
    commits validated tool outcomes back to domain state. It does not decide
    whether the loop should continue, force a final tool, or stop.
    """

    @property
    def max_turns(self) -> int: ...

    @property
    def turn_count(self) -> int: ...

    @property
    def finished(self) -> bool: ...

    @property
    def error(self) -> str | None: ...

    @property
    def termination(self) -> Termination | None: ...

    def prepare_turn(self, turn_index: int) -> PreparedTurn: ...

    def prepare_forced_final_turn(
        self,
        turn_index: int,
        *,
        reason: str,
    ) -> PreparedTurn: ...

    def validate_tool_call(
        self,
        tool_calls: Sequence[dict[str, Any]],
        prepared: PreparedTurn,
    ) -> Any: ...

    def assistant_tool_call_payloads(
        self, assistant: Message
    ) -> Sequence[dict[str, Any]]: ...

    def normalize_assistant(self, assistant: Message, parsed: Any) -> Message: ...

    def build_correction(
        self, error: str, prepared: PreparedTurn
    ) -> Message | None: ...

    def completion_kwargs(self, prepared: PreparedTurn) -> Mapping[str, Any]: ...

    def apply_tool_call(
        self,
        *,
        prepared: PreparedTurn,
        assistant_message: Message,
        parsed: Any,
        attempts: list[ModelTurnAttempt],
        prefix_messages: Sequence[Message],
        origin: str = "model",
    ) -> Any: ...

    def record_terminal_error(
        self,
        *,
        prepared: PreparedTurn,
        attempts: list[ModelTurnAttempt],
        message_delta: Sequence[Message],
        error: str,
        finish_reason: str,
    ) -> None: ...

    def mark_termination(self, termination: Termination) -> None: ...


class RuntimeLoop:
    """The canonical generic stateful model/tool execution loop.

    The loop owns repeated model-turn mechanics and delegates next-step policy
    decisions to ``decide_loop_status``. A host supplies prepared prompts/tool
    specs for each decision and commits state mutation plus persistence.

    State flow:

    - ``prepare``: ask the host for a prepared turn. With ``force_final=True``,
      the last logical turn uses ``prepare_forced_final_turn``; otherwise use
      ``prepare_turn``.
    - ``model_call``: ``ModelTurnExecutor`` performs one model request with the
      prepared messages, tools, tool choice, and host completion kwargs.
    - ``validate``: the host extracts and validates tool calls from the assistant
      response, including a plain-text answer when the host allows it.
    - ``correction``: validation exceptions may append a correction message and
      retry, up to ``validation_retries``.
    - ``apply_tool``: a valid call is normalized and handed back to the host for
      execution, state mutation, and persistence.
    - ``commit``: applying the tool call commits its result, including any
      recoverable rejection or terminal domain state.
    - ``terminal``: model errors, exhausted validation retries, domain
      termination records, or exhausted max turns stop the loop.
    """

    def __init__(
        self,
        host: RuntimeHost,
        *,
        model_client: Any,
        request_stats: RequestStatsBuilder | None = None,
        serialize_parsed: ParsedSerializer | None = None,
        validation_exceptions: tuple[type[BaseException], ...] = (
            TypeError,
            ValueError,
            KeyError,
            json.JSONDecodeError,
        ),
        validation_retries: int = 0,
        forced_final_reason: str = (
            "Max turns reached without a valid final answer; return final now."
        ),
        no_final_error: str = "max_turns_without_valid_final_answer",
        no_final_finish_reason: str = "max_turns",
        final_tool_name: str = "final_answer",
        force_final: bool = True,
        hooks: RuntimeHooks | None = None,
    ) -> None:
        self.host = host
        self.model_client = model_client
        self.request_stats = request_stats
        self.serialize_parsed = serialize_parsed
        self.validation_exceptions = validation_exceptions
        self.validation_retries = max(0, int(validation_retries))
        self.forced_final_reason = forced_final_reason
        self.no_final_error = no_final_error
        self.no_final_finish_reason = no_final_finish_reason
        self.final_tool_name = final_tool_name
        self.force_final = force_final
        self.hooks = hooks or RuntimeHooks()
        self._terminal_emitted = False

    def _emit_terminal(self) -> None:
        termination = self.host.termination
        if termination is not None and not self._terminal_emitted:
            self._terminal_emitted = True
            self.hooks.emit("terminal", termination=termination.to_dict())

    def run_until_done(self) -> None:
        if self.model_client is None:
            raise RuntimeError("a model client is required to continue runtime execution")
        while True:
            prepared = self.prepare_next_turn()
            if prepared is None:
                return
            self.run_model_turn(prepared=prepared)

    def prepare_next_turn(self) -> PreparedTurn | None:
        """Prepare the next generic model turn without invoking a model.

        External runtimes that own a model backend may use this method to keep
        force-final and terminal mechanics in one place while adapting only
        their request and response representation.
        """
        decision = self.decide_next()
        if decision.status == "terminal":
            if self.host.termination is None:
                self.host.mark_termination(
                    decision.termination
                    or Termination(
                        category="hard_limit",
                        reason=decision.terminal_reason or self.no_final_error,
                        source="runtime",
                        message=decision.finish_reason or self.no_final_finish_reason,
                        step_index=self.host.turn_count - 1 if self.host.turn_count else None,
                    )
                )
            self._emit_terminal()
            return None
        if decision.status == "force_final":
            prepared = self.host.prepare_forced_final_turn(
                self.host.turn_count,
                reason=decision.reason or self.forced_final_reason,
            )
            if decision.policy is not None:
                prepared.policy = decision.policy
            prepared.decision = decision
            return prepared
        prepared = self.host.prepare_turn(self.host.turn_count)
        if decision.policy is not None:
            prepared.policy = decision.policy
        prepared.decision = decision
        return prepared

    def decide_next(self) -> PolicyDecision:
        return decide_loop_status(
            LoopStatus(
                finished=self.host.finished,
                error=self.host.error,
                turn_count=self.host.turn_count,
                max_turns=self.host.max_turns,
                termination=self.host.termination,
            ),
            final_tool_name=self.final_tool_name,
            forced_final_reason=self.forced_final_reason,
            no_final_error=self.no_final_error,
            no_final_finish_reason=self.no_final_finish_reason,
            force_final=self.force_final,
        )

    def run_model_turn(
        self, *, prepared: PreparedTurn | None = None,
        prepare_messages: Callable[[list[Message]], list[Message]] | None = None,
    ) -> None:
        prepared = prepared or self.host.prepare_turn(self.host.turn_count)
        self.hooks.emit(
            "context",
            turn_index=prepared.turn_index,
            context=(
                prepared.model_context.inspect()
                if prepared.model_context is not None
                else None
            ),
        )
        result = ModelTurnExecutor(self.model_client).run(
            messages=prepared.messages,
            tools=prepared.tool_specs,
            tool_choice=prepared.tool_choice,
            completion_kwargs=self.host.completion_kwargs(prepared),
            max_attempts=self.validation_retries + 1,
            validate=lambda assistant: self.host.validate_tool_call(
                self.host.assistant_tool_call_payloads(assistant),
                prepared,
            ),
            normalize_assistant=self.host.normalize_assistant,
            build_correction=lambda error: self.host.build_correction(
                error,
                prepared,
            ),
            request_stats=self.request_stats,
            serialize_parsed=self.serialize_parsed,
            prepare_messages=prepare_messages,
            validation_exceptions=self.validation_exceptions,
        )
        if not result.ok:
            self.hooks.emit(
                "model",
                turn_index=prepared.turn_index,
                ok=False,
                error=result.error,
                attempts=len(result.attempts),
            )
            # Domains may record a bounded, side-effect-free recovery instead
            # of terminating. Hosts without this optional hook retain fail-fast behavior.
            recover = getattr(self.host, "recover_model_error", None)
            if recover is not None and recover(
                prepared=prepared, attempts=result.attempts,
                error=result.error or "RuntimeError: model turn failed",
                finish_reason=result.error_kind or "model_error",
            ):
                return
            self.host.record_terminal_error(
                prepared=prepared,
                attempts=result.attempts,
                message_delta=[*prepared.prompt_delta, *result.correction_messages],
                error=result.error or "RuntimeError: model turn failed",
                finish_reason=result.error_kind or "model_error",
            )
            self._emit_terminal()
            return

        if result.assistant_message is None or result.parsed is None:
            self.hooks.emit(
                "model",
                turn_index=prepared.turn_index,
                ok=False,
                error="missing assistant/tool call",
                attempts=len(result.attempts),
            )
            self.host.record_terminal_error(
                prepared=prepared,
                attempts=result.attempts,
                message_delta=[*prepared.prompt_delta, *result.correction_messages],
                error="RuntimeError: model turn returned no assistant/tool call",
                finish_reason="model_error",
            )
            self._emit_terminal()
            return

        self.hooks.emit(
            "model",
            turn_index=prepared.turn_index,
            ok=True,
            attempts=len(result.attempts),
            latency_sec=(
                result.completion.latency_sec
                if result.completion is not None
                else None
            ),
            timings=(
                dict(result.completion.timings)
                if result.completion is not None
                else {}
            ),
        )
        self.hooks.emit(
            "action",
            turn_index=prepared.turn_index,
            action=result.parsed,
        )
        transition = self.host.apply_tool_call(
            prepared=prepared,
            assistant_message=result.assistant_message,
            parsed=result.parsed,
            attempts=result.attempts,
            prefix_messages=[*prepared.prompt_delta, *result.correction_messages],
        )
        if (
            prepared.model_context is not None
            and hasattr(transition, "context")
            and getattr(transition, "context") is None
        ):
            transition.context = prepared.model_context.summary()
        self.hooks.emit(
            "transition",
            turn_index=prepared.turn_index,
            transition=transition.to_dict()
            if hasattr(transition, "to_dict")
            else transition,
        )
        self._emit_terminal()


__all__ = [
    "PreparedTurn",
    "RuntimeHost",
    "RuntimeLoop",
]
