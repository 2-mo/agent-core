"""Explicit HTTP adapter for a Chat Completions compatible endpoint."""

from collections.abc import Sequence
from contextlib import nullcontext
from copy import deepcopy
from time import perf_counter
from types import SimpleNamespace
import json

import httpx

from agent_core.context import Conversation, msg, render_openai, validate_messages
from .response import CompletionMetadata, Response, StreamEvent, response_to_message

_OPTIONS = {
    "temperature", "max_tokens", "max_completion_tokens", "top_p", "stop",
    "tools", "tool_choice", "parallel_tool_calls", "response_format", "seed",
    "frequency_penalty", "presence_penalty", "logprobs", "top_logprobs",
}


def _messages(value, images=()):
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("text must be non-empty")
        return [msg.user(value, images=images)]
    if images:
        raise ValueError("use image blocks when passing messages or a Conversation")
    if isinstance(value, Conversation):
        return value.to_list()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return validate_messages(value)
    raise TypeError("expected text, messages, or a Conversation")


class ChatModel:
    """One endpoint and model, with no hidden conversation or automatic retries.

    A supplied http_client stays owned by the caller. Credentials are sent as
    headers and are never placed in response or trajectory request metadata.
    """

    def __init__(
        self, *, model: str, base_url: str, api_key: str | None = None,
        timeout: float = 60.0, concurrency: int = 4,
        http_client: httpx.Client | None = None, **generation,
    ):
        if not model.strip():
            raise ValueError("model must be non-empty")
        url = httpx.URL(base_url)
        if url.scheme not in {"http", "https"} or not url.host:
            raise ValueError("base_url must be an HTTP(S) endpoint")
        if url.userinfo or url.query or url.fragment:
            raise ValueError("pass credentials with api_key, not in base_url")
        if type(concurrency) is not int or concurrency < 1:
            raise ValueError("concurrency must be a positive integer")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout < float("inf"):
            raise ValueError("timeout must be finite and positive")
        if unknown := set(generation) - _OPTIONS:
            raise TypeError(f"unsupported generation options: {sorted(unknown)}")
        self.model = model
        self.endpoint = str(url).rstrip("/")
        self.cfg = SimpleNamespace(timeout=timeout, image_url_mode="data")
        self._concurrency = concurrency
        self._generation = deepcopy(generation)
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = http_client if http_client is not None else httpx.Client(timeout=timeout)
        self._owns_http = http_client is None

    def close(self):
        if self._owns_http:
            self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def capacity(self):
        return {"recommended_concurrency": self._concurrency, "source": "configured"}

    def ask(self, text, *, images=(), **options) -> Response:
        return Response.from_completion(
            self._complete_with_metadata(_messages(text, images), **options)
        )

    def chat(self, conversation: Conversation, text: str, *, images=(), **options):
        if not isinstance(conversation, Conversation):
            raise TypeError("conversation must be a Conversation")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be non-empty")
        with conversation.turn():
            user = msg.user(text, images=images)
            response = self.ask([*conversation.to_list(), user], **options)
            conversation.extend([user, response.message])
            response.execution["conversation_id"] = conversation.id
            return response

    def _request(self, messages, options, *, stream=False):
        options = dict(options)
        limiter = options.pop("request_limiter", None)
        timeout = options.pop("time_limit", None)
        if options.pop("n", 1) != 1:
            raise ValueError("ChatModel returns one response per request")
        if unknown := set(options) - _OPTIONS:
            raise TypeError(f"unsupported generation options: {sorted(unknown)}")
        generation = {**self._generation, **options}
        payload = {
            "model": self.model,
            "messages": render_openai(_messages(messages), image_url_mode="data"),
            **{k: deepcopy(v) for k, v in generation.items() if v is not None},
        }
        if not payload.get("tools"):
            if payload.get("tool_choice") not in (None, "auto", "none"):
                raise ValueError("tool_choice requires a non-empty tool list")
            for key in ("tools", "tool_choice", "parallel_tool_calls"):
                payload.pop(key, None)
        if stream:
            payload.update(stream=True, stream_options={"include_usage": True})
        return payload, limiter, self.cfg.timeout if timeout is None else timeout

    def _complete_with_metadata(self, messages, **options):
        payload, limiter, timeout = self._request(messages, options)
        started = perf_counter()
        with limiter if limiter is not None else nullcontext():
            response = self._http.post(
                self.endpoint + "/chat/completions", json=payload,
                headers=self._headers, timeout=timeout,
            )
            response.raise_for_status()
            raw = response.json()
        choices = raw.get("choices") or []
        if len(choices) != 1 or not isinstance(choices[0].get("message"), dict):
            raise ValueError("endpoint must return exactly one message choice")
        choice = {**choices[0], "id": raw.get("id"), "model": raw.get("model", self.model)}
        return CompletionMetadata(
            endpoint=self.endpoint,
            responses=[response_to_message(choice["message"])],
            latency_sec=perf_counter() - started,
            model=raw.get("model", self.model), usage=raw.get("usage"),
            choices=[choice], request={k: v for k, v in payload.items() if k != "messages"},
        )

    def ask_stream(self, text, *, images=(), **options):
        payload, limiter, timeout = self._request(_messages(text, images), options, stream=True)
        started = perf_counter()
        content, calls = [], {}
        usage, finish_reason, response_id = None, None, None
        model = self.model
        with limiter if limiter is not None else nullcontext():
            with self._http.stream(
                "POST", self.endpoint + "/chat/completions", json=payload,
                headers=self._headers, timeout=timeout,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    if not data:
                        continue
                    chunk = json.loads(data)
                    if chunk.get("error"):
                        raise RuntimeError("endpoint returned a streaming error")
                    response_id = chunk.get("id") or response_id
                    model = chunk.get("model") or model
                    usage = chunk.get("usage") or usage
                    for choice in chunk.get("choices", []):
                        if choice.get("index", 0) != 0:
                            raise ValueError("only one streamed choice is supported")
                        finish_reason = choice.get("finish_reason") or finish_reason
                        delta = choice.get("delta") or {}
                        if delta.get("content"):
                            content.append(delta["content"])
                            yield StreamEvent("text_delta", text=delta["content"])
                        for part in delta.get("tool_calls") or []:
                            call = calls.setdefault(part.get("index", 0), {
                                "id": "", "type": "function",
                                "function": {"name": "", "arguments": ""},
                            })
                            if part.get("id"):
                                call["id"] = part["id"]
                            for key in ("name", "arguments"):
                                call["function"][key] += (part.get("function") or {}).get(key) or ""
                            yield StreamEvent("tool_call_delta", tool_call=deepcopy(part))
        if finish_reason is None:
            raise RuntimeError("model stream ended without a finish reason")
        message = msg.assistant("".join(content), tool_calls=[calls[i] for i in sorted(calls)])
        yield StreamEvent("response", response=Response(
            message=message, model=model, endpoint=self.endpoint,
            id=response_id, finish_reason=finish_reason, usage=usage,
            latency_sec=perf_counter() - started,
            request={k: v for k, v in payload.items() if k != "messages"},
        ))
