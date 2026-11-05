# Adapted for standalone use in Agent Core.
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from agent_core.context import Message
from agent_core.model.response import CompletionMetadata, response_to_message

from .policy import should_retry_validation

RequestStatsBuilder = Callable[
    [Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]],
    dict[str, Any],
]
AssistantValidator = Callable[[Message], Any]
AssistantNormalizer = Callable[[Message, Any], Message]
CorrectionBuilder = Callable[[str], Message | None]
ParsedSerializer = Callable[[Any], dict[str, Any] | None]


@dataclass(slots=True)
class ModelTurnAttempt:
    attempt_index: int
    request_messages: list[Message]
    request_tools: list[dict[str, Any]] = field(default_factory=list)
    request_tool_choice: Any = None
    request_stats: dict[str, Any] = field(default_factory=dict)
    assistant_message: Message | None = None
    parsed: Any = None
    parsed_payload: dict[str, Any] | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ModelTurnResult:
    attempts: list[ModelTurnAttempt]
    correction_messages: list[Message] = field(default_factory=list)
    assistant_message: Message | None = None
    parsed: Any = None
    completion: CompletionMetadata | None = None
    error: str | None = None
    error_kind: str | None = None
    exception: BaseException | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class ModelTurnExecutor:
    """Run one model turn: completion, message normalization, validation, retry.

    Higher-level loops decide what a turn means. This executor only owns the
    mechanical model-call boundary that was otherwise duplicated by runtimes.
    """

    def __init__(self, model_client: Any) -> None:
        self.model_client = model_client

    def run(
        self,
        *,
        messages: Sequence[Message],
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: Any = None,
        completion_kwargs: Mapping[str, Any] | None = None,
        max_attempts: int = 1,
        validate: AssistantValidator | None = None,
        normalize_assistant: AssistantNormalizer | None = None,
        build_correction: CorrectionBuilder | None = None,
        request_stats: RequestStatsBuilder | None = None,
        serialize_parsed: ParsedSerializer | None = None,
        prepare_messages: Callable[[list[Message]], list[Message]] | None = None,
        validation_exceptions: tuple[type[BaseException], ...] = (
            TypeError,
            ValueError,
            KeyError,
        ),
    ) -> ModelTurnResult:
        if self.model_client is None:
            raise RuntimeError("a model client is required to run a model turn")

        base_messages = deepcopy(list(messages))
        tool_specs = None if tools is None else [deepcopy(dict(item)) for item in tools]
        request_tools = tool_specs or []
        correction_messages: list[Message] = []
        attempts: list[ModelTurnAttempt] = []
        attempts_limit = max(1, int(max_attempts))

        for attempt_index in range(attempts_limit):
            request_messages = [*base_messages, *correction_messages]
            stats = {}
            completion: CompletionMetadata | None = None
            try:
                if prepare_messages is not None:
                    request_messages = prepare_messages(request_messages)
                if request_stats is not None:
                    stats = request_stats(request_messages, request_tools)
                request_seed = dict(completion_kwargs or {}).get("request_seed")
                if request_seed is not None:
                    stats["request_seed"] = int(request_seed)
                completion = self._complete(
                    request_messages,
                    tools=tool_specs,
                    tool_choice=tool_choice,
                    completion_kwargs=completion_kwargs,
                )
                if not completion.responses:
                    raise RuntimeError("model returned no responses")
            except Exception as exc:  # noqa: BLE001
                error = _exception_string(exc)
                attempts.append(
                    ModelTurnAttempt(
                        attempt_index=attempt_index,
                        request_messages=deepcopy(request_messages),
                        request_tools=deepcopy(request_tools),
                        request_tool_choice=deepcopy(tool_choice),
                        request_stats=stats,
                        error=error,
                        metadata=(
                            completion_metadata_dict(completion)
                            if completion is not None
                            else {}
                        ),
                    )
                )
                return ModelTurnResult(
                    attempts=attempts,
                    correction_messages=deepcopy(correction_messages),
                    error=error,
                    error_kind="model_error",
                    exception=exc,
                )

            assistant: Message | None = None
            try:
                assistant = response_to_message(completion.responses[0])
                parsed = validate(assistant) if validate is not None else assistant
                if normalize_assistant is not None:
                    assistant = normalize_assistant(assistant, parsed)
                attempts.append(
                    ModelTurnAttempt(
                        attempt_index=attempt_index,
                        request_messages=deepcopy(request_messages),
                        request_tools=deepcopy(request_tools),
                        request_tool_choice=deepcopy(tool_choice),
                        request_stats=stats,
                        assistant_message=deepcopy(assistant),
                        parsed=parsed,
                        parsed_payload=(
                            serialize_parsed(parsed)
                            if serialize_parsed is not None
                            else None
                        ),
                        metadata=completion_metadata_dict(completion),
                    )
                )
                return ModelTurnResult(
                    attempts=attempts,
                    correction_messages=deepcopy(correction_messages),
                    assistant_message=assistant,
                    parsed=parsed,
                    completion=completion,
                )
            except validation_exceptions as exc:
                error = _exception_string(exc)
                attempts.append(
                    ModelTurnAttempt(
                        attempt_index=attempt_index,
                        request_messages=deepcopy(request_messages),
                        request_tools=deepcopy(request_tools),
                        request_tool_choice=deepcopy(tool_choice),
                        request_stats=stats,
                        assistant_message=deepcopy(assistant),
                        error=error,
                        metadata=completion_metadata_dict(completion),
                    )
                )
                if should_retry_validation(
                    attempt_index=attempt_index,
                    max_retries=attempts_limit - 1,
                ):
                    correction = (
                        build_correction(error)
                        if build_correction is not None
                        else None
                    )
                    if correction is not None:
                        correction_messages.append(correction)
                    continue
                return ModelTurnResult(
                    attempts=attempts,
                    correction_messages=deepcopy(correction_messages),
                    error=error,
                    error_kind="validation_error",
                    exception=exc,
                )

        error = "RuntimeError: model turn did not produce a result"
        return ModelTurnResult(
            attempts=attempts,
            correction_messages=deepcopy(correction_messages),
            error=error,
            error_kind="model_error",
            exception=RuntimeError(error),
        )

    def _complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Mapping[str, Any]] | None,
        tool_choice: Any,
        completion_kwargs: Mapping[str, Any] | None,
    ) -> CompletionMetadata:
        kwargs = dict(completion_kwargs or {})
        kwargs.setdefault("n", 1)
        if tools is not None:
            kwargs["tools"] = deepcopy(list(tools))
            if tool_choice is not None:
                kwargs["tool_choice"] = deepcopy(tool_choice)
        metadata = self.model_client._complete_with_metadata(
            messages=deepcopy(list(messages)),
            **kwargs,
        )
        return completion_metadata(metadata)


def completion_metadata(metadata: Any) -> CompletionMetadata:
    if isinstance(metadata, CompletionMetadata):
        return CompletionMetadata(
            endpoint=metadata.endpoint,
            model=metadata.model,
            responses=deepcopy(metadata.responses),
            latency_sec=float(metadata.latency_sec or 0.0),
            usage=deepcopy(metadata.usage),
            timings=deepcopy(metadata.timings),
            choices=deepcopy(metadata.choices),
            request=deepcopy(metadata.request),
        )
    return CompletionMetadata(
        endpoint=getattr(metadata, "endpoint", None),
        model=getattr(metadata, "model", None),
        responses=deepcopy(getattr(metadata, "responses", [])),
        latency_sec=float(getattr(metadata, "latency_sec", 0.0) or 0.0),
        usage=deepcopy(getattr(metadata, "usage", None)),
        timings=deepcopy(getattr(metadata, "timings", {})),
        choices=deepcopy(getattr(metadata, "choices", [])),
        request=deepcopy(getattr(metadata, "request", {})),
    )


def completion_metadata_dict(metadata: Any) -> dict[str, Any]:
    value = completion_metadata(metadata)
    return {
        "endpoint": value.endpoint,
        "model": value.model,
        "latency_sec": value.latency_sec,
        "usage": deepcopy(value.usage),
        "timings": deepcopy(value.timings),
        "choices": deepcopy(value.choices),
        "request": deepcopy(value.request),
    }


def _exception_string(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


__all__ = [
    "ModelTurnAttempt",
    "ModelTurnExecutor",
    "ModelTurnResult",
    "completion_metadata",
    "completion_metadata_dict",
]
