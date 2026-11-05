# Adapted for standalone use in Agent Core.
from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass
from threading import Event, Thread
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from .contracts import ParsedToolCall, Tool, ToolExecutionResult, ToolSpec
from .executor import execute_tool, rejected_execution
from .json_arguments import load_argument_object
from .types import RuntimeContext, ToolResult

RUNTIME_CONTEXT_PARAMETER = "runtime_context"


def run_with_timeout(
    func: Any, timeout: float | None, *args: Any, **kwargs: Any
) -> Any:
    if timeout is None:
        return func(*args, **kwargs)

    result_ready = Event()
    result: Any = None
    error: BaseException | None = None
    traceback_obj: Any = None

    def runner() -> None:
        nonlocal result, error, traceback_obj
        try:
            result = func(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001
            error = exc
            traceback_obj = exc.__traceback__
        finally:
            result_ready.set()

    worker = Thread(
        target=runner,
        name=f"timeout:{getattr(func, '__name__', 'call')}",
        daemon=True,
    )
    worker.start()
    if not result_ready.wait(timeout):
        raise TimeoutError(f"request timeout after {timeout}s")
    if error is not None:
        raise error.with_traceback(traceback_obj)
    return result


@dataclass(frozen=True, slots=True)
class FunctionTool:
    name: str
    description: str
    parameters_schema: dict[str, Any]
    func: Callable[..., Any]
    timeout_sec: float | None = None
    brief_desc: str = ""
    inject_runtime_context: bool = False

    @property
    def tool_name(self) -> str:
        return self.name

    def schema(self) -> dict[str, Any]:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters_schema=self.parameters_schema,
        ).schema()

    def run(self, runtime_context: RuntimeContext, arguments: Any) -> Any:
        if not isinstance(arguments, dict):
            raise TypeError("function tool arguments must be an object")
        kwargs = deepcopy(arguments)
        if self.inject_runtime_context:
            if RUNTIME_CONTEXT_PARAMETER in kwargs:
                raise TypeError(
                    f"{RUNTIME_CONTEXT_PARAMETER!r} is reserved for runtime injection"
                )
            kwargs[RUNTIME_CONTEXT_PARAMETER] = runtime_context
        return run_with_timeout(self.func, self.timeout_sec, **kwargs)


class ToolRegistry:
    def __init__(
        self, tools: Iterable[Tool | Callable[..., Any]] | None = None
    ) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(
        self,
        tool: Tool | Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        parameters_schema: dict[str, Any] | None = None,
        timeout_sec: float | None = None,
        brief_desc: str = "",
        replace: bool = False,
    ):
        if tool is None:

            def decorator(actual_func: Callable[..., Any]) -> Callable[..., Any]:
                self.register(
                    actual_func,
                    name=name,
                    description=description,
                    parameters_schema=parameters_schema,
                    timeout_sec=timeout_sec,
                    brief_desc=brief_desc,
                    replace=replace,
                )
                return actual_func

            return decorator

        normalized = _normalize_registered_tool(
            tool,
            name=name,
            description=description,
            parameters_schema=parameters_schema,
            timeout_sec=timeout_sec,
            brief_desc=brief_desc,
        )
        tool_name = normalized.tool_name
        if tool_name in self._tools and not replace:
            raise ValueError(f"tool {tool_name!r} is already registered")
        self._tools[tool_name] = normalized
        return tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name)

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"unknown tool: {name}") from exc

    def list_names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[dict[str, Any]]:
        return [deepcopy(tool.schema()) for tool in self._tools.values()]

    def has_tools(self) -> bool:
        return bool(self._tools)

    def brief_descs(self) -> str:
        return "".join(
            tool.brief_desc
            for tool in self._tools.values()
            if getattr(tool, "brief_desc", "")
        )

    def execute(
        self, call: ParsedToolCall, *, context: RuntimeContext = None
    ) -> ToolExecutionResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return rejected_execution(call, f"unknown tool: {call.name}")
        return execute_tool(tool, call, context=context)

    def execute_tool_call(
        self,
        tool_call: dict[str, Any],
        *,
        context: RuntimeContext = None,
    ) -> ToolExecutionResult:
        try:
            call = parse_tool_call(tool_call)
        except Exception as exc:  # noqa: BLE001
            return rejected_execution(
                _partial_tool_call(tool_call),
                f"invalid tool call: {type(exc).__name__}: {exc}",
            )
        return self.execute(call, context=context)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


def parse_tool_call(tool_call: dict[str, Any]) -> ParsedToolCall:
    if not isinstance(tool_call, dict):
        raise TypeError("tool_call must be a dict")

    function_payload = tool_call.get("function")
    if not isinstance(function_payload, dict):
        raise TypeError("tool_call.function must be a dict")

    name = str(function_payload.get("name") or "").strip()
    if not name:
        raise ValueError("tool_call.function.name must not be empty")

    raw = function_payload.get("arguments")
    if raw is None or raw == "":
        return ParsedToolCall(
            tool_call_id=str(tool_call.get("id") or ""),
            name=name,
            arguments={},
            raw_arguments=None if raw is None else "",
        )

    repairs = ()
    if isinstance(raw, dict):
        arguments = deepcopy(raw)
        raw_arguments = json.dumps(raw, ensure_ascii=False, default=str)
    elif isinstance(raw, str):
        arguments, repairs = load_argument_object(raw)
        raw_arguments = raw
    else:
        raise TypeError("tool_call.function.arguments must be a dict, string, or None")

    return ParsedToolCall(
        tool_call_id=str(tool_call.get("id") or ""),
        name=name,
        arguments=arguments,
        raw_arguments=raw_arguments,
        repairs=repairs,
    )


def parameters_schema_from_signature(func: Callable[..., Any]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    hints = get_type_hints(func)
    for name, parameter in inspect.signature(func).parameters.items():
        if parameter.kind in {parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD}:
            raise TypeError("tool callables must use named parameters")
        if parameter.kind is parameter.POSITIONAL_ONLY:
            raise TypeError("tool callables must not use positional-only parameters")
        if name == RUNTIME_CONTEXT_PARAMETER:
            continue
        properties[name] = _annotation_schema(hints.get(name, parameter.annotation))
        if parameter.default is inspect.Signature.empty:
            required.append(name)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _normalize_registered_tool(
    tool: Tool | Callable[..., Any],
    *,
    name: str | None,
    description: str | None,
    parameters_schema: dict[str, Any] | None,
    timeout_sec: float | None,
    brief_desc: str,
) -> Tool:
    if hasattr(tool, "tool_name") and hasattr(tool, "schema") and hasattr(tool, "run"):
        if (
            any(
                value is not None
                for value in (name, description, parameters_schema, timeout_sec)
            )
            or brief_desc
        ):
            raise ValueError(
                "tool metadata arguments are only valid for callable tools"
            )
        return tool  # type: ignore[return-value]

    if not callable(tool):
        raise TypeError("tool must implement tool_name/schema/run or be callable")

    return function_tool(
        tool,
        name=name,
        description=description,
        parameters_schema=parameters_schema,
        timeout_sec=timeout_sec,
        brief_desc=brief_desc,
    )


def function_tool(
    func: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
    parameters_schema: dict[str, Any] | None = None,
    timeout_sec: float | None = None,
    brief_desc: str = "",
) -> FunctionTool:
    if not callable(func):
        raise TypeError("function tool source must be callable")

    tool_name = str(name or getattr(func, "__name__", "")).strip()
    if not tool_name:
        raise ValueError("tool name must not be empty")
    return FunctionTool(
        name=tool_name,
        description=description or str(getattr(func, "__doc__", "") or "").strip(),
        parameters_schema=(
            deepcopy(parameters_schema)
            if parameters_schema is not None
            else parameters_schema_from_signature(func)
        ),
        func=func,
        timeout_sec=timeout_sec,
        brief_desc=brief_desc,
        inject_runtime_context=RUNTIME_CONTEXT_PARAMETER
        in inspect.signature(func).parameters,
    )


def _annotation_schema(annotation: Any) -> dict[str, Any]:
    if annotation is inspect.Signature.empty or annotation is Any:
        return {}
    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}

    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in {list, tuple, set}:
        item_annotation = args[0] if args else Any
        return {"type": "array", "items": _annotation_schema(item_annotation)}
    if origin is dict:
        return {"type": "object"}
    if origin is Literal:
        enum_values = list(args)
        schema = _annotation_schema(type(enum_values[0])) if enum_values else {}
        return {**schema, "enum": enum_values}
    if origin in {Union, UnionType}:
        non_none = [item for item in args if item is not type(None)]
        if len(non_none) == 1:
            return _annotation_schema(non_none[0])
    return {}


def _partial_tool_call(tool_call: Any) -> ParsedToolCall:
    payload = tool_call if isinstance(tool_call, dict) else {}
    function_payload = payload.get("function")
    function_payload = function_payload if isinstance(function_payload, dict) else {}
    raw_arguments = function_payload.get("arguments")
    return ParsedToolCall(
        tool_call_id=str(payload.get("id") or ""),
        name=str(function_payload.get("name") or ""),
        arguments={},
        raw_arguments=raw_arguments if isinstance(raw_arguments, str) else None,
    )


__all__ = [
    "FunctionTool",
    "ParsedToolCall",
    "Tool",
    "ToolExecutionResult",
    "ToolRegistry",
    "ToolResult",
    "function_tool",
    "parameters_schema_from_signature",
    "parse_tool_call",
    "run_with_timeout",
]
