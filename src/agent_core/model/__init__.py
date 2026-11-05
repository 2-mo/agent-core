"""Chat Completions adapter and provider-independent response records."""

from .client import ChatModel
from .response import CompletionMetadata, Response, StreamEvent

__all__ = ["ChatModel", "CompletionMetadata", "Response", "StreamEvent"]
