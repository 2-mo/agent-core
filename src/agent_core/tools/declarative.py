# Adapted for standalone use in Agent Core.
"""Declarative authoring helpers for engine function tools."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, ClassVar

from .registry import FunctionTool, function_tool


class DeclarativeTool:
    name: ClassVar[str] = ""
    brief: ClassVar[str] = ""
    description: ClassVar[str] = ""
    schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def resolve_context(self, runtime_context: Any) -> Any:
        return runtime_context

    def action(self, context: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("each tool must implement `action`")

    def run(self, runtime_context: Any, arguments: Any) -> Any:
        if not isinstance(arguments, dict):
            raise TypeError("declarative tool arguments must be an object")
        return self.action(self.resolve_context(runtime_context), **deepcopy(arguments))

    def to_function_tool(self) -> FunctionTool:
        tool = self

        def _run(*, runtime_context: Any, **kwargs: Any) -> Any:
            return tool.run(runtime_context, kwargs)

        if not tool.name:
            raise ValueError(f"{type(tool).__name__} must set a non-empty `name`")
        return function_tool(
            _run,
            name=tool.name,
            description=tool.description or (type(tool).__doc__ or "").strip(),
            parameters_schema=tool.schema,
            brief_desc=tool.brief,
        )


__all__ = ["DeclarativeTool"]
