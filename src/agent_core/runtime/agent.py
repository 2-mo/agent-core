# Adapted for standalone use in Agent Core.
"""Reusable Agent configuration on the canonical model/tool RuntimeLoop."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from time import perf_counter
from typing import Any
from uuid import uuid4

from agent_core.context import Conversation, message_tool_calls, msg
from agent_core.context.policy import ContextLimitError, ContextPolicy
from agent_core.model.parallel import bounded_map
from agent_core.model.response import CompletionMetadata, Response
from agent_core.tools.registry import FunctionTool, ToolRegistry
from agent_core.tools.types import ToolResult

from .admission import RequestAdmission
from .goal import Termination
from .limits import RunBudget, RunLimitError, positive_seconds
from .policy import ToolPolicy
from .runtime_loop import PreparedTurn, RuntimeLoop
from .trajectory import TrajectoryRecord, TransitionRecord

_UNSET = object()


@dataclass(frozen=True, slots=True)
class TurnView:
    """What a turn policy sees: the turn number and the tools called so far."""

    turn_index: int
    max_turns: int
    tool_names: tuple[str, ...]
    calls: tuple[str, ...]

    @property
    def turns_left(self) -> int:
        return self.max_turns - self.turn_index


@dataclass(slots=True)
class RunResult:
    final: Response | None
    trajectory: TrajectoryRecord
    termination: Termination
    usage: dict[str, int]
    model_calls: int
    children: list[RunResult] = field(default_factory=list)

    @property
    def text(self) -> str:
        return self.final.text if self.final is not None else ""

    @property
    def ok(self) -> bool:
        return self.termination.category == "semantic_success"

    @property
    def output(self) -> Any:
        """Arguments of the final tool when it ended the run, otherwise the final response."""
        return self.trajectory.final_output

    def to_dict(self):
        return {
            "final": self.final.to_dict() if self.final else None,
            "trajectory": self.trajectory.to_dict(),
            "termination": self.termination.to_dict(),
            "usage": dict(self.usage),
            "model_calls": self.model_calls,
            "children": [child.to_dict() for child in self.children],
        }


@dataclass(slots=True)
class RunEvent:
    type: str
    agent: str
    run_id: str
    data: Any = None
    result: RunResult | None = None


class _RequestGate:
    def __init__(self, host):
        self.host = host

    def acquire(self):
        host = self.host
        self.slot = (
            host.admission.acquire(host.agent.client, host.budget.check)
            if host.admission
            else None
        )
        try:
            host.budget.take()
            host.model_calls += 1
        except BaseException:
            self.release()
            raise

    def release(self):
        if self.slot is not None:
            self.host.admission.release(self.slot)
            self.slot = None

    def __enter__(self):
        self.acquire()

    def __exit__(self, *args):
        self.release()

    def wait(self, seconds):
        remaining = self.host.budget.remaining_seconds()
        self.host.budget.cancel.wait(
            min(seconds, remaining) if remaining is not None else seconds
        )
        self.host.budget.check()


class _RunClient:
    def __init__(self, host):
        self.host = host

    def _complete_with_metadata(self, messages, **kwargs):
        host = self.host
        host.budget.check()
        config = getattr(
            host.agent.client, "cfg", getattr(host.agent.client, "_config", None)
        )
        if getattr(config, "system_prompt", None) and not any(
            m["role"] == "system" for m in messages
        ):
            messages = [msg.system(config.system_prompt), *messages]
        if host.context_policy is not None:
            try:
                messages, report = host.context_policy.apply(
                    messages, kwargs.get("tools") or []
                )
            except ContextLimitError as exc:
                host.context_reports.append(exc.report)
                raise
            host.context_reports.append(report)
        messages = Conversation(messages, image_url_mode=host.image_url_mode)
        host.last_request_messages = messages.to_list()
        kwargs["request_limiter"] = _RequestGate(host)
        remaining = host.budget.remaining_seconds()
        if remaining is not None:
            configured = (
                kwargs.get("time_limit")
                or getattr(config, "time_limit", None)
                or getattr(config, "timeout", None)
            )
            kwargs["time_limit"] = (
                min(remaining, configured) if configured is not None else remaining
            )
        before_calls = host.model_calls
        try:
            return self._generate(messages, kwargs, before_calls)
        except BaseException:
            if host.model_calls > before_calls:
                host.budget.record_usage(None)
            raise

    def _generate(self, messages, kwargs, before_calls):
        host = self.host
        if not host.streaming:
            metadata = host.agent.client._complete_with_metadata(messages, **kwargs)
            response = Response.from_completion(metadata)
        else:
            kwargs.pop("n", None)
            kwargs.pop("max_try", None)
            kwargs.pop("wait_time", None)
            response = None
            events = host.agent.client.ask_stream(messages, **kwargs)
            try:
                for event in events:
                    host.budget.check()
                    if event.type == "response":
                        response = event.response
                    else:
                        host.emit(event.type, event)
            finally:
                events.close()
            if response is None:
                raise RuntimeError("model stream returned no final response")
            metadata = CompletionMetadata(
                response.endpoint,
                [response.message],
                response.latency_sec,
                model=response.model,
                usage=response.usage,
                timings=response.timings,
                choices=[{key: deepcopy(value) for key, value in response.raw.items() if key != "chunks"}]
                if response.raw else [],
                request=deepcopy(response.request),
            )
        host.responses.append(response)
        host.budget.record_usage(response.usage)
        if host.model_calls - before_calls > 1:
            host.budget.record_usage(None)
        return metadata


class Agent:
    """Role and tools bound to a model client; each run owns independent state."""

    def __init__(
        self,
        client,
        *,
        name="agent",
        instructions="",
        tools=(),
        max_turns=12,
        max_calls=32,
        tool_concurrency=1,
        timeout=None,
        max_total_tokens=None,
        context_policy: ContextPolicy | None = None,
        final_tool: str | None = None,
        turn_policy: Callable[[TurnView], ToolPolicy | None] | None = None,
        **generation,
    ):
        """``final_tool`` names the tool whose successful call ends the run (no
        further model turn); the last turn then forces it, and a reply without a
        tool call does not end the run. ``turn_policy(view)`` may return a
        ``ToolPolicy`` that allows or forces tools for one turn."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("agent name must be non-empty")
        for label, value in (
            ("max_turns", max_turns),
            ("max_calls", max_calls),
            ("tool_concurrency", tool_concurrency),
        ):
            _positive(label, value)
        if timeout is not None:
            positive_seconds("timeout", timeout)
        if max_total_tokens is not None:
            _positive("max_total_tokens", max_total_tokens)
        if context_policy is not None and not isinstance(context_policy, ContextPolicy):
            raise TypeError("context_policy must be a ContextPolicy")
        self.timeout = timeout
        self.max_total_tokens = max_total_tokens
        self.context_policy = context_policy
        self.client = client
        self.name = name
        self.instructions = instructions
        self.tools = tuple(tools)
        self.max_turns = max_turns
        self.max_calls = max_calls
        self.tool_concurrency = tool_concurrency
        self.final_tool = final_tool
        self.turn_policy = turn_policy
        self.generation = deepcopy(generation)

    def run(
        self,
        task: str,
        *,
        conversation: Conversation | None = None,
        images=(),
        tools=None,
        max_turns=None,
        max_calls=None,
        tool_concurrency=None,
        cancel: Event | None = None,
        timeout=_UNSET,
        max_total_tokens=_UNSET,
        context_policy=_UNSET,
        _parent=None,
        _emit=None,
        _streaming=False,
        _admission=None,
        **generation,
    ) -> RunResult:
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string")
        turns = self.max_turns if max_turns is None else max_turns
        calls = self.max_calls if max_calls is None else max_calls
        workers = (
            self.tool_concurrency if tool_concurrency is None else tool_concurrency
        )
        for label, value in (
            ("max_turns", turns),
            ("max_calls", calls),
            ("tool_concurrency", workers),
        ):
            _positive(label, value)
        if conversation is not None and not isinstance(conversation, Conversation):
            raise TypeError("conversation must be a Conversation")
        timeout = self.timeout if timeout is _UNSET else timeout
        tokens = (
            self.max_total_tokens if max_total_tokens is _UNSET else max_total_tokens
        )
        context_policy = (
            self.context_policy if context_policy is _UNSET else context_policy
        )
        if timeout is not None:
            positive_seconds("timeout", timeout)
        if tokens is not None:
            _positive("max_total_tokens", tokens)
        if context_policy is not None and not isinstance(context_policy, ContextPolicy):
            raise TypeError("context_policy must be a ContextPolicy")
        budget = RunBudget(
            calls,
            _parent.budget.cancel
            if _parent
            else (cancel if cancel is not None else Event()),
            max_total_tokens=tokens,
            deadline=perf_counter() + timeout if timeout is not None else None,
            lock=_parent.budget.lock if _parent else Lock(),
            parent=_parent.budget if _parent else None,
        )
        observer = (
            _emit if _emit is not None else (_parent.observer if _parent else None)
        )
        streaming = _streaming or bool(_parent and _parent.streaming)
        registry = ToolRegistry(self.tools if tools is None else tools)
        if self.final_tool is not None and self.final_tool not in registry.list_names():
            raise ValueError(f"final_tool {self.final_tool!r} is not one of the agent's tools")

        def make_host(history):
            return _AgentHost(
                self,
                task,
                history,
                images,
                registry,
                turns,
                workers,
                budget,
                {**self.generation, **generation},
                observer,
                streaming,
                _parent,
                _parent.admission if _parent else _admission,
                context_policy,
                timeout,
            )

        try:
            with (
                conversation.turn(check=budget.check)
                if conversation is not None
                else nullcontext()
            ):
                host = make_host(conversation)
                host.emit("run_started", {"task": task})
                loop = RuntimeLoop(
                    host,
                    model_client=_RunClient(host),
                    force_final=self.final_tool is not None,
                    final_tool_name=self.final_tool or "final_answer",
                    no_final_error="max_turns",
                    no_final_finish_reason="max_turns",
                )
                loop.run_until_done()
                result = host.result()
                if conversation is not None:
                    conversation.extend(host.committed)
        except RunLimitError as exc:
            # The history lock was not acquired. Do not inspect or mutate it.
            host = make_host(None)
            if conversation is not None:
                host.trajectory.rollout_metadata.update(
                    conversation_id=conversation.id, history_not_acquired=True
                )
            host.emit("run_started", {"task": task})
            host.mark_termination(Termination("hard_limit", str(exc), "runtime"))
            result = host.result()
        if _parent:
            with _parent.children_lock:
                _parent.children.append(result)
        host.emit("run_completed", result)
        return result

    def run_many(
        self,
        tasks: Sequence[str],
        *,
        concurrency="auto",
        request_concurrency="auto",
        **kwargs,
    ) -> list[RunResult]:
        if isinstance(tasks, (str, bytes)) or not isinstance(tasks, Sequence):
            raise TypeError("tasks must be a sequence of non-empty strings")
        if any(not isinstance(task, str) or not task.strip() for task in tasks):
            raise ValueError("tasks must be non-empty strings")
        if kwargs.get("conversation") is not None:
            raise ValueError(
                "run_many uses independent conversations; fork explicitly for shared prefixes"
            )
        if concurrency != "auto":
            _positive("concurrency", concurrency)
        if request_concurrency != "auto":
            _positive("request_concurrency", request_concurrency)
        if not tasks:
            return []
        capacity = (
            self.client.capacity()
            if request_concurrency == "auto"
            else {"recommended_concurrency": request_concurrency, "source": "explicit"}
        )
        admission = RequestAdmission(request_concurrency)
        request_limit = admission.register(self.client, capacity)
        selected = min(256, 2 * request_limit) if concurrency == "auto" else concurrency
        selected = min(len(tasks), selected)
        results = bounded_map(
            lambda task: self.run(task, _admission=admission, **kwargs), tasks, selected
        )
        for result in results:
            result.trajectory.rollout_metadata.update(
                concurrency=selected,
                concurrency_source=capacity["source"]
                if concurrency == "auto"
                else "explicit",
                request_concurrency=request_limit,
                request_admission=admission.snapshot(),
            )
        return results

    def as_tool(self, *, name=None, description=None):
        def delegate(task: str, runtime_context):
            result = self.run(task, _parent=runtime_context)
            return ToolResult(
                status="success" if result.ok else "error",
                content=result.text,
                error=None if result.ok else result.termination.reason,
            )

        return FunctionTool(
            name=name or self.name,
            description=description
            or self.instructions
            or f"Delegate a task to {self.name}",
            parameters_schema={
                "type": "object",
                "properties": {"task": {"type": "string"}},
                "required": ["task"],
                "additionalProperties": False,
            },
            func=delegate,
            inject_runtime_context=True,
        )

    def run_stream(self, task: str, **kwargs):
        """Yield model/tool/run events; the final event contains the RunResult.

        Closing the iterator requests cooperative cancellation. In-flight tools
        finish normally; model streaming stops at the next received chunk.
        """
        events = Queue(maxsize=128)
        cancel = kwargs.pop("cancel", None) or Event()
        done = Event()
        consumer_closed = Event()
        errors = []

        def emit(event):
            while not consumer_closed.is_set():
                try:
                    events.put(event, timeout=0.1)
                    return
                except Full:
                    continue

        def work():
            try:
                self.run(task, cancel=cancel, _emit=emit, _streaming=True, **kwargs)
            except Exception as exc:  # noqa: BLE001 - transfer worker exceptions to the consumer.
                errors.append(exc)
            finally:
                done.set()

        worker = Thread(target=work, name=f"agent-stream:{self.name}", daemon=True)
        worker.start()
        try:
            while not done.is_set() or not events.empty():
                try:
                    yield events.get(timeout=0.1)
                except Empty:
                    continue
            if errors:
                raise errors[0]
        finally:
            consumer_closed.set()
            if not done.is_set():
                cancel.set()
            worker.join(timeout=1)


def _spec_name(spec):
    return (spec.get("function") or spec).get("name")


def _forced(name):
    return {"type": "function", "function": {"name": name}}


class _AgentHost:
    def __init__(
        self,
        agent,
        task,
        conversation,
        images,
        registry,
        max_turns,
        tool_concurrency,
        budget,
        options,
        observer,
        streaming,
        parent,
        admission,
        context_policy,
        timeout,
    ):
        self.agent, self.registry, self.max_turns = agent, registry, max_turns
        self.tool_concurrency, self.budget, self.options = (
            tool_concurrency,
            budget,
            options,
        )
        self.observer, self.streaming = observer, streaming
        self.admission = admission
        self.context_policy = context_policy
        self.context_reports = []
        self.last_request_messages = []
        self.turn_count = 0
        self.finished = False
        self.error = None
        self.termination = None
        self.final = None
        self.final_arguments = None
        self.responses = []
        self.children = []
        self.children_lock = Lock()
        self.model_calls = 0
        self.started = perf_counter()
        self.committed = []
        self.initial = conversation.to_list() if conversation is not None else []
        config = getattr(agent.client, "cfg", getattr(agent.client, "_config", None))
        self.image_url_mode = (
            conversation.image_url_mode
            if conversation is not None
            else getattr(config, "image_url_mode", "file")
        )
        self.user = msg.user(task, images=images)
        self.conversation_id = (
            conversation.id if conversation is not None else uuid4().hex
        )
        self.trajectory = TrajectoryRecord(
            rollout_id=uuid4().hex,
            task_id=uuid4().hex,
            domain="agent",
            initial_messages=deepcopy(self.initial),
            input={"task": task},
            parent_trajectory_id=parent.trajectory.trajectory_id if parent else None,
            rollout_metadata={
                "agent": agent.name,
                "conversation_id": self.conversation_id,
                "max_turns": max_turns,
                "max_calls": budget.max_calls,
                "tool_concurrency": tool_concurrency,
                "timeout": timeout,
                "max_total_tokens": budget.max_total_tokens,
            },
        )

    def emit(self, kind, data):
        if self.observer is not None:
            self.observer(
                RunEvent(
                    kind,
                    self.agent.name,
                    self.trajectory.trajectory_id,
                    data=data,
                    result=data if isinstance(data, RunResult) else None,
                )
            )

    def prepare_turn(self, turn_index):
        messages = [*self.initial, *self.committed]
        if self.agent.instructions:
            from agent_core.context import message_text

            instructions = "\n\n".join(
                [message_text(m) for m in messages if m["role"] == "system"]
                + [self.agent.instructions]
            )
            messages = [
                msg.system(instructions),
                *[m for m in messages if m["role"] != "system"],
            ]
        delta = [self.user] if not self.committed else []
        names = tuple(self.registry.list_names())
        specs = self.registry.specs()
        tool_choice = self.options.get("tool_choice", "auto")
        policy = None
        if self.agent.turn_policy is not None:
            calls = tuple(
                (transition.tool_action or {}).get("function", {}).get("name", "")
                for transition in self.trajectory.transitions
                if transition.kind == "tool"
            )
            policy = self.agent.turn_policy(
                TurnView(turn_index, self.max_turns, names, calls)
            )
        if policy is not None and policy.mode != "auto":
            unknown = set(policy.allowed_tools) - set(names)
            if unknown:
                raise ValueError(f"turn policy names unknown tools: {sorted(unknown)}")
            names = policy.allowed_tools
            specs = [spec for spec in specs if _spec_name(spec) in names]
            if policy.mode == "force":
                tool_choice = _forced(names[0])
        return PreparedTurn(
            turn_index,
            names,
            [*messages, *delta],
            delta,
            specs,
            tool_choice=tool_choice,
            policy=policy,
        )

    def prepare_forced_final_turn(self, turn_index, *, reason):
        prepared = self.prepare_turn(turn_index)
        prepared.available_tools = (self.agent.final_tool,)
        prepared.tool_specs = [
            spec for spec in self.registry.specs() if _spec_name(spec) == self.agent.final_tool
        ]
        prepared.tool_choice = _forced(self.agent.final_tool)
        prepared.policy = ToolPolicy("force", (self.agent.final_tool,), reason)
        return prepared

    def completion_kwargs(self, prepared):
        options = dict(self.options)
        options.pop("tool_choice", None)
        return options

    def assistant_tool_call_payloads(self, assistant):
        return message_tool_calls(assistant)

    def validate_tool_call(self, calls, prepared):
        finish_reason = self.responses[-1].finish_reason
        if calls and finish_reason not in (None, "stop", "tool_calls"):
            raise ValueError(
                f"tool calls require a complete model response; finish_reason={finish_reason!r}"
            )
        ids = [call.get("id") for call in calls]
        if any(not value for value in ids) or len(ids) != len(set(ids)):
            raise ValueError("tool calls require distinct non-empty IDs")
        names = [(call.get("function") or {}).get("name") for call in calls]
        if prepared.policy is not None and prepared.policy.mode != "auto":
            for name in names:
                if name not in prepared.available_tools:
                    raise ValueError(f"tool {name!r} is not allowed this turn")

        choice = prepared.tool_choice
        if isinstance(choice, dict) and choice.get("type") == "function":
            forced_name = (choice.get("function") or {}).get("name")
            if not names or any(name != forced_name for name in names):
                raise ValueError(f"this turn requires a call to tool {forced_name!r}")
        elif choice == "required" and not calls:
            raise ValueError("this turn requires a tool call")
        elif choice == "none" and calls:
            raise ValueError("tool calls are disabled this turn")
        return list(calls)

    def normalize_assistant(self, assistant, parsed):
        assistant.setdefault("metadata", {})["agent"] = self.agent.name
        return assistant

    def build_correction(self, error, prepared):
        return None

    def apply_tool_call(
        self,
        *,
        prepared,
        assistant_message,
        parsed,
        attempts,
        prefix_messages,
        origin="model",
    ):
        response = self.responses[-1]
        response.message = deepcopy(assistant_message)
        delta = [*prefix_messages, assistant_message]
        self.committed.extend(deepcopy(delta))
        transition = self.trajectory.add_transition(
            TransitionRecord(
                kind="model",
                agent_id=self.agent.name,
                model=response.model,
                endpoint=response.endpoint,
                latency_sec=response.latency_sec,
                ok=True,
                message_delta=deepcopy(delta),
                model_action={"response": response.to_dict()},
                context={"request_messages": deepcopy(self.last_request_messages)}
                if self.context_policy
                else {},
            )
        )
        self.turn_count += 1
        if not parsed and self.agent.final_tool is not None:
            # Only the final tool ends this run; a plain reply uses up the turn.
            violation = self.budget.violation()
            if violation:
                self.mark_termination(Termination("hard_limit", violation, "runtime"))
            return transition
        if not parsed:
            self.final = response
            violation = self.budget.violation()
            if violation:
                self.mark_termination(Termination("hard_limit", violation, "runtime"))
                return transition
            success = response.finish_reason in (None, "stop")
            self.mark_termination(
                Termination(
                    "semantic_success" if success else "semantic_failure",
                    "completed" if success else str(response.finish_reason),
                    "model",
                )
            )
            return transition

        def execute(call):
            try:
                self.budget.check()
            except RunLimitError as exc:
                skipped_reason = str(exc)
                # Keep the history protocol-valid even when cancellation skips tools.
                execution = None
            else:
                skipped_reason = None
                self.emit("tool_started", deepcopy(call))
                execution = self.registry.execute_tool_call(call, context=self)
            # Report completion immediately; history is still committed in call order.
            self.emit(
                "tool_completed",
                {
                    "tool_call_id": call["id"],
                    "name": call["function"]["name"],
                    "content": execution.content
                    if execution
                    else f"Tool skipped: {skipped_reason}",
                    "ok": bool(execution and execution.ok),
                },
            )
            return call, execution, skipped_reason

        workers = min(self.tool_concurrency, len(parsed))
        executions = (
            [execute(call) for call in parsed]
            if workers == 1
            else bounded_map(execute, parsed, workers)
        )
        for call, execution, skipped_reason in executions:
            name = call["function"]["name"]
            content = (
                execution.content
                if execution is not None
                else f"Tool skipped: {skipped_reason}"
            )
            tool_message = msg.tool(tool_call_id=call["id"], name=name, content=content)
            messages = [
                tool_message,
                *(execution.result.messages_to_append if execution else []),
            ]
            self.committed.extend(deepcopy(messages))
            self.trajectory.add_transition(
                TransitionRecord(
                    kind="tool",
                    agent_id=self.agent.name,
                    tool_action=deepcopy(call),
                    ok=execution.ok if execution else False,
                    error=execution.error if execution else skipped_reason,
                    message_delta=messages,
                    latency_sec=execution.latency_sec if execution else 0,
                )
            )
            if execution is None:
                self.mark_termination(
                    Termination("hard_limit", skipped_reason, "runtime")
                )
            elif name == self.agent.final_tool and execution.ok:
                self.final = response
                self.final_arguments = deepcopy(execution.arguments)
                self.mark_termination(Termination("semantic_success", "final_tool", "tool"))
            elif execution.result.status == "error":
                category = (
                    "hard_limit"
                    if execution.error
                    in {
                        "max_calls",
                        "cancelled",
                        "deadline",
                        "max_total_tokens",
                        "usage_unavailable",
                        "context_limit",
                    }
                    else "runtime_error"
                )
                self.mark_termination(
                    Termination(category, execution.error or "tool_error", "tool")
                )
        violation = self.budget.violation()
        if violation:
            self.mark_termination(Termination("hard_limit", violation, "runtime"))
        # A forced tool is a first-turn constraint, otherwise it can loop forever.
        self.options["tool_choice"] = "auto"
        return transition

    def record_terminal_error(
        self, *, prepared, attempts, message_delta, error, finish_reason
    ):
        self.error = error
        reasons = {
            "max_calls",
            "cancelled",
            "deadline",
            "max_total_tokens",
            "usage_unavailable",
        }
        reason = next(
            (value for value in reasons if f"RunLimitError: {value}" in error),
            finish_reason,
        )
        if "ContextLimitError:" in error:
            reason = "context_limit"
        reason = self.budget.violation() or reason
        self.trajectory.add_transition(
            TransitionRecord(
                kind="error",
                agent_id=self.agent.name,
                ok=False,
                error=error,
                context={"request_messages": prepared.messages},
            )
        )
        self.mark_termination(
            Termination(
                "hard_limit"
                if reason
                in {
                    "max_calls",
                    "cancelled",
                    "deadline",
                    "max_total_tokens",
                    "usage_unavailable",
                    "context_limit",
                }
                else "runtime_error",
                reason,
                "runtime",
                raw_error=error,
            )
        )

    def mark_termination(self, termination):
        self.termination = termination
        self.finished = termination.category == "semantic_success"

    def result(self):
        usage = Counter()
        for response in self.responses:
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                usage[key] += (response.usage or {}).get(key, 0)
        for child in self.children:
            usage.update(child.usage)
        self.trajectory.finished = self.finished
        self.trajectory.finish_reason = self.termination.reason
        self.trajectory.termination = self.termination.to_dict()
        self.trajectory.error = self.error
        self.trajectory.final_output = (
            self.final_arguments
            if self.final_arguments is not None
            else self.final.to_dict() if self.final else None
        )
        self.trajectory.latency_sec = perf_counter() - self.started
        self.trajectory.rollout_metadata["usage"] = dict(usage)
        self.trajectory.rollout_metadata["budget"] = self.budget.snapshot()
        self.trajectory.rollout_metadata["context"] = self.context_reports
        self.trajectory.rollout_metadata["missing_usage_calls"] = (
            self.model_calls
            - sum(response.usage is not None for response in self.responses)
            + sum(
                child.trajectory.rollout_metadata["missing_usage_calls"]
                for child in self.children
            )
        )
        return RunResult(
            self.final,
            self.trajectory,
            self.termination,
            dict(usage),
            self.model_calls + sum(child.model_calls for child in self.children),
            list(self.children),
        )


def _positive(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


__all__ = ["Agent", "RunEvent", "RunResult"]
