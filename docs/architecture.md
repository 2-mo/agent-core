# Architecture

Agent Core separates model transport, conversations, executable tools, and the agent loop. Application-specific behavior lives in tools or a runtime host.

## Package structure

| Module | Responsibility |
| --- | --- |
| `context` | Canonical messages, conversations, snapshots, context selection, image and frame representations |
| `tools` | Function schemas, argument parsing, validation, registration, normalized results |
| `model` | Chat Completions HTTP adapter, responses, streaming, bounded worker scheduling |
| `runtime` | Agent runs, policies, shared budgets, model admission, runtime hosts, trajectories |

The package uses `httpx`, `jsonschema`, and `json-repair`. Worker scheduling uses Python's standard library. No model service starts on import or during installation.

## Core contracts

| Contract | Meaning |
| --- | --- |
| `Message` | A role with structured content blocks |
| `Conversation` | Explicit mutable history with a lock for shared turns |
| `ToolSpec` | Model-visible name, description, and parameter schema |
| `ParsedToolCall` | Call identity, normalized arguments, original input, repair markers |
| `ToolResult` | Success or failure, text content, optional additional messages |
| `PreparedTurn` | The exact messages, tools, and policy for one model turn |
| `TransitionRecord` | One model, tool, or state transition |
| `TrajectoryRecord` | Initial state, transitions, final output, and termination |

Tools implement `tool_name`, `schema()`, and `run(context, arguments)`. Functions can be adapted automatically. Classes can use `DeclarativeTool`. A `ToolSet` supplies named schemas and dispatch without requiring a common implementation base class.

See [Tool design](tool-design.md) ([中文](tool-design.zh-CN.md)) for the shared calling sequence, authoring forms, result semantics, and delegation boundary.

## One run

1. `Agent` creates an independent run host, conversation view, registry, and budget.
2. `RuntimeLoop` asks the host to prepare a turn.
3. `ModelTurnExecutor` invokes the model and validates its response through the host.
4. The host executes valid tool calls, appends paired results, and records transitions.
5. The loop continues until a final reply, a successful designated final tool, an error, or a limit.

An explicit `final_tool` makes completion depend on that tool's successful execution. The final available turn forces it. Without a designated final tool, an ordinary completed model reply can finish a run.

## State and concurrency

Each run owns its trajectory. Several runs may share a model client; `run_many()` separates task workers from model-request slots. A delegated agent releases its caller's completed request slot before making its own request, and shares the parent's budget lineage.

Tools may execute concurrently when `tool_concurrency` is greater than one. Completion events are emitted as work finishes; history is committed in the model's tool-call order. Tool authors are responsible for thread safety of their own shared state and external effects.

Calls sharing a `Conversation` serialize across complete turns. Use `conversation.fork()` for independent branches. Context policies trim the model-visible view while keeping stored history intact.

## Model adapter boundary

The runtime calls `_complete_with_metadata(messages, **kwargs)` and expects `CompletionMetadata`. A custom adapter must acquire and release the supplied `request_limiter` around each real model request so call budgets and admission remain correct. It should respect `time_limit` when provided.

Batch execution also uses `capacity()`, returning `recommended_concurrency` and `source`. Streaming uses `ask_stream()`, which yields `StreamEvent` values and a final `response` event. `ChatModel` implements these interfaces using explicit HTTP configuration and one request per call.

## Domain extensions

For a custom state machine, implement `RuntimeHost` rather than duplicating the model loop. The host builds prompts and tool contracts, validates actions, commits domain state, and records termination. `RuntimeLoop` retains generic turn and retry mechanics. See [the small counter host](../tests/test_toy_domain.py) for an executable example.

Image and frame-reference blocks are generic representations. Local images can be encoded for an endpoint; applications that use references register a resolver through `set_frame_resolver()`. The application owns data retrieval. `ChatModel` renders video frame groups as ordered images with timestamps.

## Failure boundaries

Schema violations return `failed` before the handler runs. Expected business rejections should also return `ToolResult(status="failed", ...)`. Unhandled handler exceptions return `error` and terminate a standard Agent run.

The Agent checks per-turn tool selection and model finish metadata before dispatching a batch. Calls outside an active `allow`/`force` policy, a missing required call, or tool calls from truncated or filtered replies end the run with `validation_error`; no handler in that batch runs. Tool argument errors remain recoverable tool results.

Cancellation and deadlines are cooperative. They stop admission of further requests and are checked between steps. Already-running Python tools are not forcibly interrupted. A function-tool timeout stops waiting; it does not kill the worker or roll back external effects.

Token budgets use reported usage. They prevent subsequent calls after exhaustion, but a completed request can overshoot the remaining budget. If token accounting is required and usage is missing, the run stops with `usage_unavailable`.
