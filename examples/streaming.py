"""Read model text as it arrives, followed by the final structured run result."""

import os
from types import SimpleNamespace

from agent_core import Agent, ChatModel

cfg = SimpleNamespace(
    base_url=os.environ["MODEL_BASE_URL"],
    model=os.environ["MODEL_NAME"],
    api_key=os.environ.get("MODEL_API_KEY"),
)


def main():
    with ChatModel(model=cfg.model, base_url=cfg.base_url, api_key=cfg.api_key) as model:
        for event in Agent(model).run_stream("Explain a queue in two sentences."):
            if event.type == "text_delta":
                print(event.data.text, end="", flush=True)
            elif event.result is not None:
                print(f"\nSuccess: {event.result.ok}")


if __name__ == "__main__":
    main()
