"""Search local documents with scripted argument correction and a saved trace.

The model replies are a deterministic local stub. File search, parameter
validation, tool feedback, and trajectory recording use the actual runtime.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import httpx

from agent_core import Agent, ChatModel, ToolResult, function_tool
from agent_core.runtime import TrajectoryRecorder

cfg = SimpleNamespace(
    directory=Path(__file__).parent / "catalog",
    query="handoff",
    max_turns=4,
)


def search_catalog(query: str, limit: int = 3) -> ToolResult:
    """Find local Markdown documents containing the query, ignoring case."""
    matches = []
    for path in sorted(cfg.directory.glob("*.md")):
        content = path.read_text(encoding="utf-8")
        if query.casefold() in content.casefold():
            matches.append({"file": path.name, "content": content.strip()})
            if len(matches) == limit:
                break
    return ToolResult(content=json.dumps({"matches": matches}, ensure_ascii=False))


search_tool = function_tool(search_catalog, parameters_schema={
    "type": "object",
    "properties": {
        "query": {"type": "string", "minLength": 1},
        "limit": {"type": "integer", "minimum": 1, "maximum": 10},
    },
    "required": ["query"],
    "additionalProperties": False,
})


def scripted_endpoint(request):
    messages = json.loads(request.content)["messages"]
    tool_messages = [message for message in messages if message["role"] == "tool"]
    if len(tool_messages) < 2:
        # First send the wrong type; then send the corrected argument.
        arguments = {"query": cfg.query, "limit": "2" if not tool_messages else 2}
        message = {"role": "assistant", "content": None, "tool_calls": [{
            "id": f"search-{len(tool_messages) + 1}",
            "type": "function",
            "function": {"name": "search_catalog", "arguments": json.dumps(arguments)},
        }]}
        finish_reason = "tool_calls"
    else:
        matches = json.loads(tool_messages[-1]["content"])["matches"]
        lines = [f"- {item['file']}: {item['content'].splitlines()[0].lstrip('# ')}"
                 for item in matches]
        message = {"role": "assistant", "content":
                   f"Matches for {cfg.query!r}:\n" + ("\n".join(lines) or "No matching documents.")}
        finish_reason = "stop"
    return httpx.Response(200, json={
        "choices": [{"message": message, "finish_reason": finish_reason}],
    })


def main():
    with httpx.Client(transport=httpx.MockTransport(scripted_endpoint)) as http:
        model = ChatModel(model="scripted-catalog", base_url="https://model.example/v1", http_client=http)
        agent = Agent(model, tools=[search_tool], max_turns=cfg.max_turns,
                      instructions="Search the catalog. Correct invalid arguments using tool feedback.")
        result = agent.run(f"Find documents about {cfg.query}.")

    print(result.text)
    for transition in result.trajectory.transitions:
        if transition.kind == "tool":
            status = "success" if transition.ok else "failed"
            print(f"Tool {transition.tool_action['id']}: {status}")
            if transition.error:
                print(f"  {transition.error}")
    TrajectoryRecorder("outputs/catalog-runs.jsonl").write(result.trajectory)
    print(f"Success: {result.ok}; model calls: {result.model_calls}")
    print("Trace appended to outputs/catalog-runs.jsonl")
    if not result.ok:
        raise SystemExit(result.termination.reason)


if __name__ == "__main__":
    main()
