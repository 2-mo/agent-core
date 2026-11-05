# Adapted for standalone use in Agent Core.
"""Generic tool-call contracts shared by registries and domain tool sets."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol

from .types import RuntimeContext, ToolResult


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """The model-visible contract for one function tool."""

    name: str
    description: str
    parameters_schema: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("tool name must not be empty")
        if not isinstance(self.parameters_schema, dict):
            raise TypeError("tool parameters_schema must be an object")
        object.__setattr__(self, "parameters_schema", deepcopy(self.parameters_schema))

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": deepcopy(self.parameters_schema),
            },
        }


class Tool(Protocol):
    brief_desc: str

    @property
    def tool_name(self) -> str: ...

    def schema(self) -> dict[str, Any]: ...

    def run(self, runtime_context: RuntimeContext, arguments: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class ParsedToolCall:
    tool_call_id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str | None
    repairs: tuple[str, ...] = ()


@dataclass(slots=True)
class ToolExecutionResult:
    tool_call_id: str
    name: str
    arguments: dict[str, Any] | None
    raw_arguments: str | None
    ok: bool
    output: Any | None
    result: ToolResult
    content: str
    error: str | None
    latency_sec: float


class ToolSet(Protocol):
    """A domain-owned collection of generic executable tools."""

    def list_names(self) -> list[str]: ...

    def specs(
        self, names: tuple[str, ...] | list[str] | None = None
    ) -> list[dict[str, Any]]: ...

    def execute(
        self,
        call: ParsedToolCall,
        *,
        context: RuntimeContext = None,
    ) -> ToolExecutionResult: ...


__all__ = [
    "ParsedToolCall",
    "Tool",
    "ToolExecutionResult",
    "ToolSet",
    "ToolSpec",
]
