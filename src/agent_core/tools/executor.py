# Adapted for standalone use in Agent Core.
"""Strict generic execution boundary for function tools."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from functools import lru_cache
from typing import Any

from jsonschema import Draft202012Validator, SchemaError

from .contracts import ParsedToolCall, Tool, ToolExecutionResult
from .types import RuntimeContext, ToolResult, normalize_tool_result


class ToolArgumentError(ValueError):
    """A model-correctable violation of a tool's parameter schema."""


class ToolSchemaError(ValueError):
    """A tool published an invalid function-parameter schema."""


def execute_tool(
    tool: Tool,
    call: ParsedToolCall,
    *,
    context: RuntimeContext = None,
) -> ToolExecutionResult:
    """Validate and execute one parsed tool call.

    Parameter violations are returned as ``failed`` so a model loop can repair
    them. Handler and result-normalization failures are execution ``error``s.
    """

    started = time.perf_counter()
    try:
        validate_tool_arguments(_parameters_schema(tool), call.arguments)
    except ToolArgumentError as exc:
        return rejected_execution(call, str(exc), started=started)
    except ToolSchemaError as exc:
        return error_execution(call, str(exc), started=started)

    try:
        output = tool.run(context, deepcopy(call.arguments))
        result = normalize_tool_result(output)
    except Exception as exc:  # noqa: BLE001
        return error_execution(
            call,
            f"{type(exc).__name__}: {exc}",
            started=started,
        )
    return ToolExecutionResult(
        tool_call_id=call.tool_call_id,
        name=call.name,
        arguments=deepcopy(call.arguments),
        raw_arguments=call.raw_arguments,
        ok=result.ok,
        output=deepcopy(output),
        result=result,
        content=result.content,
        error=result.error,
        latency_sec=round(time.perf_counter() - started, 4),
    )


def validate_tool_arguments(parameters_schema: dict[str, Any], arguments: Any) -> None:
    if not isinstance(arguments, dict):
        raise ToolArgumentError("tool arguments must be an object")
    try:
        validator = _validator(json.dumps(parameters_schema, sort_keys=True))
    except SchemaError as exc:
        raise ToolSchemaError(f"invalid tool parameter schema: {exc.message}") from exc
    errors = sorted(validator.iter_errors(arguments), key=_error_sort_key)
    if not errors:
        return
    error = errors[0]
    location = ".".join(str(item) for item in error.absolute_path) or "arguments"
    raise ToolArgumentError(f"{location}: {error.message}")



@lru_cache(maxsize=256)
def _validator(schema: str) -> Draft202012Validator:
    """Check each distinct schema once; tools reuse a few schemas every turn."""
    parameters = json.loads(schema)
    Draft202012Validator.check_schema(parameters)
    return Draft202012Validator(parameters)

def rejected_execution(
    call: ParsedToolCall,
    error: str,
    *,
    started: float | None = None,
) -> ToolExecutionResult:
    rendered = f"tool call rejected: {error}"
    content = (
        f"{call.name or 'tool call'} was rejected: {error}. "
        f"Revise the arguments and call {call.name or 'the tool'} again."
    )
    result = ToolResult(status="failed", content=content, error=rendered)
    return _result_from_failure(call, result, started=started)


def error_execution(
    call: ParsedToolCall,
    error: str,
    *,
    started: float | None = None,
) -> ToolExecutionResult:
    result = ToolResult(status="error", content=error, error=error)
    return _result_from_failure(call, result, started=started)


def _parameters_schema(tool: Tool) -> dict[str, Any]:
    schema = tool.schema()
    function = schema.get("function") if isinstance(schema, dict) else None
    parameters = function.get("parameters") if isinstance(function, dict) else None
    if not isinstance(parameters, dict):
        raise ToolSchemaError(f"tool {tool.tool_name!r} has no parameter schema")
    return parameters


def _result_from_failure(
    call: ParsedToolCall,
    result: ToolResult,
    *,
    started: float | None,
) -> ToolExecutionResult:
    elapsed = 0.0 if started is None else round(time.perf_counter() - started, 4)
    return ToolExecutionResult(
        tool_call_id=call.tool_call_id,
        name=call.name,
        arguments=deepcopy(call.arguments) if call.arguments else None,
        raw_arguments=call.raw_arguments,
        ok=False,
        output=None,
        result=result,
        content=result.content,
        error=result.error,
        latency_sec=elapsed,
    )


def _error_sort_key(error: Any) -> tuple[str, str]:
    return (".".join(str(item) for item in error.absolute_path), error.message)


__all__ = [
    "ToolArgumentError",
    "ToolSchemaError",
    "error_execution",
    "execute_tool",
    "rejected_execution",
    "validate_tool_arguments",
]
