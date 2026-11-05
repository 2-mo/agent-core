import json
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import httpx
import pytest

from agent_core import Agent, ChatModel, Conversation
from agent_core.model.parallel import bounded_map
from agent_core.runtime import TrajectoryRecord, TrajectoryRecorder


def reply(text=None, calls=None, tokens=3):
    return httpx.Response(200, json={
        "id": "reply-demo", "model": "demo-model",
        "choices": [{"index": 0, "finish_reason": "tool_calls" if calls else "stop",
                     "message": {"role": "assistant", "content": text, "tool_calls": calls or []}}],
        "usage": {"total_tokens": tokens},
    })


def tool_call(name, arguments, call_id="call-1"):
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}


def test_agent_round_trip_and_credentials_stay_out_of_trajectory():
    requests = []

    def endpoint(request):
        requests.append(json.loads(request.content))
        assert request.headers["authorization"] == "Bearer local-test-key"
        if len(requests) == 1:
            return reply(calls=[tool_call("multiply", {"a": 6, "b": 7})])
        assert requests[-1]["messages"][-1]["content"] == "42"
        assert requests[-1]["messages"][-1]["tool_call_id"] == "call-1"
        return reply("42")

    def multiply(a: int, b: int) -> int:
        return a * b

    with httpx.Client(transport=httpx.MockTransport(endpoint)) as http:
        with ChatModel(model="demo-model", base_url="https://model.example/v1",
                       api_key="local-test-key", http_client=http) as model:
            result = Agent(model, tools=[multiply]).run("Multiply six by seven")
        assert not http.is_closed
    assert result.ok and result.text == "42"
    assert result.model_calls == 2 and result.usage["total_tokens"] == 6
    assert "local-test-key" not in json.dumps(result.to_dict())
    assert [t.kind for t in result.trajectory.transitions] == ["model", "tool", "model"]


def test_call_limit_stops_before_another_request():
    count = 0

    def endpoint(request):
        nonlocal count
        count += 1
        return reply(calls=[tool_call("echo", {"text": "again"})])

    def echo(text: str):
        return text

    with httpx.Client(transport=httpx.MockTransport(endpoint)) as http:
        model = ChatModel(model="demo-model", base_url="https://model.example/v1", http_client=http)
        result = Agent(model, tools=[echo], max_calls=1).run("Echo")
    assert count == 1
    assert not result.ok and result.termination.reason == "max_calls"


def test_token_budget_fails_closed_when_usage_is_missing():
    def endpoint(request):
        response = reply("done").json()
        response.pop("usage")
        return httpx.Response(200, json=response)

    with httpx.Client(transport=httpx.MockTransport(endpoint)) as http:
        model = ChatModel(model="demo-model", base_url="https://model.example/v1", http_client=http)
        result = Agent(model, max_total_tokens=10).run("Answer")
    assert not result.ok and result.termination.reason == "usage_unavailable"


def test_chat_preserves_history_when_request_fails():
    conversation = Conversation()
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503))) as http:
        model = ChatModel(model="demo-model", base_url="https://model.example/v1", http_client=http)
        with pytest.raises(httpx.HTTPStatusError):
            model.chat(conversation, "Hello")
    assert conversation.to_list() == []


def test_parallel_tools_commit_in_model_call_order():
    requests = []

    def endpoint(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            return reply(calls=[tool_call("echo", {"text": "first"}, "one"),
                                tool_call("echo", {"text": "second"}, "two")])
        assert [m["tool_call_id"] for m in body["messages"] if m["role"] == "tool"] == ["one", "two"]
        return reply("done")

    def echo(text: str):
        return text

    with httpx.Client(transport=httpx.MockTransport(endpoint)) as http:
        model = ChatModel(model="demo-model", base_url="https://model.example/v1", http_client=http)
        result = Agent(model, tools=[echo], tool_concurrency=2).run("Echo twice")
    assert result.ok


def test_delegation_shares_call_budget_and_records_children():
    requests = []

    def endpoint(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            return reply(calls=[tool_call("writer", {"task": "Say hello"})])
        return reply("Hello")

    with httpx.Client(transport=httpx.MockTransport(endpoint)) as http:
        model = ChatModel(model="demo-model", base_url="https://model.example/v1", http_client=http)
        writer = Agent(model, name="writer")
        result = Agent(model, tools=[writer.as_tool()], max_calls=2).run("Ask the writer")
    assert len(requests) == 2
    assert len(result.children) == 1 and result.children[0].ok
    assert result.termination.reason == "max_calls"


def test_run_many_preserves_input_order():
    def endpoint(request):
        messages = json.loads(request.content)["messages"]
        return reply(messages[-1]["content"][0]["text"])

    with httpx.Client(transport=httpx.MockTransport(endpoint)) as http:
        model = ChatModel(model="demo-model", base_url="https://model.example/v1", http_client=http)
        results = Agent(model).run_many(["one", "two", "three"], concurrency=2, request_concurrency=1)
    assert [r.text for r in results] == ["one", "two", "three"]
    assert all(r.ok for r in results)


def test_bounded_map_limits_active_work_and_propagates_failures():
    lock = Lock()
    active = peak = 0

    def work(value):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            return value * 2
        finally:
            with lock:
                active -= 1

    assert bounded_map(work, range(40), 3) == list(range(0, 80, 2))
    assert peak <= 3 and active == 0
    with pytest.raises(ZeroDivisionError):
        bounded_map(lambda n: 1 / n, [1, 0, 2], 2)


def test_trajectory_recorder_appends_complete_records(tmp_path):
    path = tmp_path / "runs.jsonl"
    recorder = TrajectoryRecorder(str(path))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda i: recorder.write(TrajectoryRecord(rollout_id=str(i), task_id="demo")), range(20)))
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert {r["rollout_id"] for r in records} == {str(i) for i in range(20)}


def sse(chunks):
    return httpx.Response(200, headers={"content-type": "text/event-stream"},
                          text="".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n")


def test_stream_assembles_fragmented_tool_arguments():
    chunks = [
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "one", "function": {"name": "echo", "arguments": '{"text":'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"hello"}'}}]}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"total_tokens": 7}},
    ]
    with httpx.Client(transport=httpx.MockTransport(lambda r: sse(chunks))) as http:
        model = ChatModel(model="demo-model", base_url="https://model.example/v1", http_client=http)
        events = list(model.ask_stream("Echo"))
    response = events[-1].response
    assert response.usage == {"total_tokens": 7}
    assert response.tool_calls[0]["function"] == {"name": "echo", "arguments": '{"text":"hello"}'}


def test_stream_truncation_does_not_create_successful_response():
    with httpx.Client(transport=httpx.MockTransport(lambda r: sse([{"choices": [{"delta": {"content": "partial"}}]}]))) as http:
        model = ChatModel(model="demo-model", base_url="https://model.example/v1", http_client=http)
        with pytest.raises(RuntimeError, match="finish reason"):
            list(model.ask_stream("Hello"))


def test_agent_stream_returns_final_result():
    chunks = [{"choices": [{"delta": {"content": "Hello"}, "finish_reason": "stop"}],
               "usage": {"total_tokens": 3}}]
    with httpx.Client(transport=httpx.MockTransport(lambda r: sse(chunks))) as http:
        model = ChatModel(model="demo-model", base_url="https://model.example/v1", http_client=http)
        events = list(Agent(model).run_stream("Greet me"))
    assert events[-1].result.ok and events[-1].result.text == "Hello"
    assert events[-1].result.model_calls == 1
