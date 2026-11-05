"""Connect a tool-using agent to your configured Chat Completions endpoint."""

import os
from types import SimpleNamespace

from agent_core import Agent, ChatModel

cfg = SimpleNamespace(
    base_url=os.environ["MODEL_BASE_URL"],
    model=os.environ["MODEL_NAME"],
    api_key=os.environ.get("MODEL_API_KEY"),
    max_turns=6,
)


def multiply(a: int, b: int) -> int:
    """Multiply two integers exactly."""
    return a * b


def main():
    with ChatModel(model=cfg.model, base_url=cfg.base_url, api_key=cfg.api_key) as model:
        agent = Agent(model, tools=[multiply], max_turns=cfg.max_turns,
                      instructions="Use the provided tool for multiplication.")
        result = agent.run("What is 17 times 23?")
        print(result.text if result.ok else result.termination.reason)


if __name__ == "__main__":
    main()
