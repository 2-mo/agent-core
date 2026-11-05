"""Run the complete agent/tool loop with a deterministic local model stub."""

import json
import httpx

from agent_core import Agent, ChatModel


def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


def endpoint(request):
    messages = json.loads(request.content)["messages"]
    if messages[-1]["role"] == "tool":
        message = {"role": "assistant", "content": "The answer is 42."}
        reason = "stop"
    else:
        message = {"role": "assistant", "content": None, "tool_calls": [{
            "id": "multiply-1", "type": "function",
            "function": {"name": "multiply", "arguments": '{"a":6,"b":7}'},
        }]}
        reason = "tool_calls"
    return httpx.Response(200, json={
        "choices": [{"message": message, "finish_reason": reason}],
        "usage": {"total_tokens": 5},
    })


def main():
    with httpx.Client(transport=httpx.MockTransport(endpoint)) as http:
        model = ChatModel(model="demo", base_url="https://model.example/v1", http_client=http)
        result = Agent(model, tools=[multiply], max_turns=3).run("What is 6 times 7?")
    print(result.text)
    print(f"Success: {result.ok}; model calls: {result.model_calls}")
    print("Transitions:", ", ".join(t.kind for t in result.trajectory.transitions))


if __name__ == "__main__":
    main()
