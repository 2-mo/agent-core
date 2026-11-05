# API guide

## Model and conversation

```python
import os
from agent_core import ChatModel, Conversation

with ChatModel(
    model=os.environ["MODEL_NAME"],
    base_url=os.environ["MODEL_BASE_URL"],
    api_key=os.environ.get("MODEL_API_KEY"),
    temperature=0,
) as model:
    response = model.ask("Define a queue in one sentence.")
    print(response.text)

    conversation = Conversation()
    model.chat(conversation, "My preferred language is Python.")
    response = model.chat(conversation, "Show a small queue example.")
```

`ask()` accepts text, canonical messages, or a `Conversation`; it does not execute tool calls. `chat()` appends the user message and successful response to the supplied conversation. The client has no hidden history.

`Response` contains `text`, `message`, `tool_calls`, `usage`, `model`, `latency_sec`, and request metadata. `response.json()` strictly parses the text as JSON. Request credentials are HTTP headers and are not included in request records.

Generation options can be set on `ChatModel`, `Agent`, or an individual call. An explicit `None` removes an inherited option. Unsupported option names raise an error. The HTTP adapter does not automatically retry failed requests.

## Tool definitions

Pass typed Python functions directly to `Agent(tools=[...])`, or construct a definition explicitly:

```python
from agent_core import ToolResult, function_tool


def lookup(code: str) -> ToolResult:
    catalog = {"sample": "Example entry"}
    if code not in catalog:
        return ToolResult(status="failed", content="Unknown code. Use sample.")
    return ToolResult(content=catalog[code])


lookup_tool = function_tool(
    lookup,
    description="Look up an entry in the catalog.",
    parameters_schema={
        "type": "object",
        "properties": {"code": {"type": "string", "minLength": 1}},
        "required": ["code"],
        "additionalProperties": False,
    },
)
```

Signature inference covers common scalar and collection annotations. Supply an explicit schema for nested structures and precise constraints. A parameter named `runtime_context` is injected by the executor and is not model-visible.

`ToolResult.messages_to_append` carries extra canonical messages, including images. `state_delta` is available to custom hosts; the standard Agent does not automatically apply it to application state.

## Runs and limits

```python
from agent_core import Agent, ContextPolicy

agent = Agent(
    model,
    tools=[lookup_tool],
    max_turns=8,
    max_calls=12,
    timeout=60,
    max_total_tokens=8000,
    context_policy=ContextPolicy(max_chars=24000, overflow="trim_oldest"),
)
result = agent.run("Look up sample.")
```

`max_turns` counts model turns. `max_calls` bounds actual requests, including delegates. Time limits are cooperative; token limits rely on provider usage and can be exceeded by the last completed request. The runtime preserves the termination reason instead of treating every stop as success.

Use `result.ok` to check completion, `result.text` for the final response, and `result.termination` for status. `result.output` contains the arguments of a designated final tool when it completes the run. `result.children` contains delegated runs.

`ContextPolicy(max_tokens=..., token_counter=...)` accepts an application-supplied counter. It does not assume one tokenizer fits all models. Trimming preserves complete prior user turns and system instructions; context that still exceeds the limit produces `context_limit`.

## Policies and delegation

```python
from agent_core import Agent, ToolPolicy


def choose_tools(view):
    if view.turn_index == 0:
        return ToolPolicy(mode="force", allowed_tools=("lookup",))
    return ToolPolicy(mode="auto")


agent = Agent(model, tools=[lookup_tool], turn_policy=choose_tools)
editor = Agent(model, name="editor", instructions="Edit text for clarity.")
coordinator = Agent(model, tools=[editor.as_tool()], max_calls=10)
```

An `allow` policy limits the exposed tool set; a `force` policy selects one tool. A separate `final_tool="name"` requires that tool to succeed before completion. Tools receive arguments through the same validation boundary in each mode.

## Concurrency and streaming

```python
results = agent.run_many(
    ["Look up sample.", "Look up another code."],
    concurrency=2,
    request_concurrency=2,
)

for event in agent.run_stream("Look up sample."):
    if event.type == "text_delta":
        print(event.data.text, end="", flush=True)
    elif event.result is not None:
        print(event.result.termination)
```

Batch results retain input order. With `auto`, request concurrency comes from the client's configured capacity, and task concurrency is selected from that limit. Streaming emits model deltas, tool lifecycle events, and a final run result. Closing the iterator requests cooperative cancellation.

## Saving history and traces

```python
from agent_core import Conversation
from agent_core.runtime import TrajectoryRecorder

snapshot = conversation.snapshot()
restored = Conversation.load_snapshot(snapshot)
branch = restored.fork()

TrajectoryRecorder("outputs/runs.jsonl").write(result.trajectory)
```

Trajectories include prompts, tool arguments, and outputs. Save them to an appropriate local destination for your application. JSONL append operations are serialized within the process; Unix platforms also use file locks across processes.

Trajectory snapshots and forks preserve recorded state. Resuming a custom domain still requires a host that reconstructs that domain; loading a trace alone does not restart arbitrary external work.
