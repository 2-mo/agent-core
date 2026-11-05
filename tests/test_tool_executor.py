from __future__ import annotations

from agent_core.tools import ToolRegistry


def test_registry_rejects_schema_violations_before_handler_execution() -> None:
    registry = ToolRegistry()
    calls: list[int] = []

    @registry.register(
        name="count_frames",
        description="Count frames.",
        parameters_schema={
            "type": "object",
            "properties": {"count": {"type": "integer", "minimum": 1}},
            "required": ["count"],
            "additionalProperties": False,
        },
    )
    def count_frames(count: int) -> str:
        calls.append(count)
        return str(count)

    result = registry.execute_tool_call(
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "count_frames", "arguments": '{"count": "3"}'},
        }
    )

    assert result.ok is False
    assert result.result.status == "failed"
    assert "count" in result.content
    assert calls == []


def test_registry_treats_unknown_result_status_as_execution_error() -> None:
    registry = ToolRegistry()

    @registry.register(
        name="broken_status",
        description="Return a malformed status.",
        parameters_schema={"type": "object", "additionalProperties": False},
    )
    def broken_status() -> dict[str, str]:
        return {"status": "unknown", "content": "broken"}

    result = registry.execute_tool_call(
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "broken_status", "arguments": "{}"},
        }
    )

    assert result.ok is False
    assert result.result.status == "error"
    assert "unknown tool result status" in result.content
