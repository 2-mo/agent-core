import json
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import httpx
import pytest

from agent_core import Agent, ChatModel, Conversation, ToolPolicy
from agent_core.model.parallel import bounded_map
from agent_core.runtime import TrajectoryRecord, TrajectoryRecorder


def reply(text=None, calls=None, tokens=3, finish_reason=None):
    return httpx.Response(200, json={
        "id": "reply-demo", "model": "demo-model",
        "choices": [{"index": 0, "finish_reason": finish_reason or ("tool_calls" if calls else "stop"),
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


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("mode,returned_names", [
    ("allow", ["allowed", "outside_turn"]),
    ("force", []),
    ("force", ["outside_turn"]),
    ("named", []),
    ("required", []),
    ("none", ["allowed"]),
])
def test_agent_rejects_policy_violations_before_any_tool_runs(mode, returned_names, streaming):
    invoked = []

    def allowed():
        invoked.append("allowed")
        return "done"

    def outside_turn():
        invoked.append("outside_turn")
        return "done"

    calls = [tool_call(name, {}, f"call-{i}") for i, name in enumerate(returned_names)]

    def endpoint(request):
        body = json.loads(request.content)
        if mode in {"allow", "force"}:
            assert [spec["function"]["name"] for spec in body["tools"]] == ["allowed"]
        if streaming:
            deltas = [{**call, "index": i} for i, call in enumerate(calls)]
            return sse([{"choices": [{"delta": {"content": "done", "tool_calls": deltas},
                                      "finish_reason": "tool_calls" if calls else "stop"}]}])
        return reply("done", calls)

    options = {"tool_concurrency": 2}
    if mode in {"allow", "force"}:
        options["turn_policy"] = lambda _: ToolPolicy(mode, ("allowed",))
    else:
        options["tool_choice"] = (
            {"type": "function", "function": {"name": "allowed"}}
            if mode == "named" else mode
        )
    conversation = Conversation()
    with httpx.Client(transport=httpx.MockTransport(endpoint)) as http:
        model = ChatModel(model="demo-model", base_url="https://model.example/v1", http_client=http)
        agent = Agent(model, tools=[allowed, outside_turn], **options)
        result = (list(agent.run_stream("Check policy", conversation=conversation))[-1].result
                  if streaming else agent.run("Check policy", conversation=conversation))
    assert invoked == []
    assert not result.ok and result.termination.reason == "validation_error"
    assert result.model_calls == 1
    assert conversation.to_list() == []
    assert [t.kind for t in result.trajectory.transitions] == ["error"]


@pytest.mark.parametrize("mode", ["allow", "force", "named", "required"])
def test_agent_accepts_allowed_calls_then_returns_to_auto(mode):
    requests = []
    invoked = []

    def allowed():
        invoked.append("allowed")
        return "done"

    def endpoint(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            return reply(calls=[tool_call("allowed", {})])
        assert body["tool_choice"] == "auto"
        return reply("finished")

    if mode in {"allow", "force"}:
        options = {"turn_policy": lambda view: ToolPolicy(mode, ("allowed",))
                   if view.turn_index == 0 else ToolPolicy()}
    else:
        options = {"tool_choice": {"type": "function", "function": {"name": "allowed"}}
                   if mode == "named" else mode}
    with httpx.Client(transport=httpx.MockTransport(endpoint)) as http:
        model = ChatModel(model="demo-model", base_url="https://model.example/v1", http_client=http)
        result = Agent(model, tools=[allowed], **options).run("Check policy")
    assert result.ok and result.text == "finished"
    assert invoked == ["allowed"] and result.model_calls == 2


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("finish_reason", ["length", "content_filter", "unexpected"])
def test_incomplete_response_cannot_execute_even_a_complete_final_tool(finish_reason, streaming):
    invoked = []

    def finish(answer: str):
        invoked.append(answer)
        return answer

    call = tool_call("finish", {"answer": "complete arguments"})

    def endpoint(request):
        if streaming:
            return sse([{"choices": [{"delta": {"tool_calls": [{**call, "index": 0}]},
                                      "finish_reason": finish_reason}]}])
        return reply(calls=[call], finish_reason=finish_reason)

    with httpx.Client(transport=httpx.MockTransport(endpoint)) as http:
        model = ChatModel(model="demo-model", base_url="https://model.example/v1", http_client=http)
        agent = Agent(model, tools=[finish], final_tool="finish", max_turns=1)
        result = (list(agent.run_stream("Finish"))[-1].result
                  if streaming else agent.run("Finish"))
    assert invoked == []
    assert not result.ok and result.termination.reason == "validation_error"
    assert finish_reason in result.termination.raw_error
    assert result.model_calls == 1
    assert [t.kind for t in result.trajectory.transitions] == ["error"]


@pytest.mark.parametrize("finish_reason", ["tool_calls", "stop", None])
def test_complete_final_tool_succeeds_on_last_turn(finish_reason):
    invoked = []

    def finish(answer: str):
        invoked.append(answer)
        return answer

    def endpoint(request):
        body = reply(calls=[tool_call("finish", {"answer": "done"})]).json()
        body["choices"][0]["finish_reason"] = finish_reason
        return httpx.Response(200, json=body)

    with httpx.Client(transport=httpx.MockTransport(endpoint)) as http:
        model = ChatModel(model="demo-model", base_url="https://model.example/v1", http_client=http)
        result = Agent(model, tools=[finish], final_tool="finish", max_turns=1).run("Finish")
    assert result.ok and result.termination.reason == "final_tool"
    assert invoked == ["done"] and result.model_calls == 1
    assert result.output == {"answer": "done"}
