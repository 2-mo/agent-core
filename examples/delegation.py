"""Expose one agent as another agent's tool, with a shared request budget."""

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
        editor = Agent(model, name="editor", instructions="Rewrite text clearly and concisely.")
        coordinator = Agent(
            model, tools=[editor.as_tool()], max_calls=6,
            instructions="Ask the editor to improve the text, then present the result.",
        )
        result = coordinator.run("Improve: We are writing to let you know the meeting moved to Friday.")
        print(result.text if result.ok else result.termination.reason)


if __name__ == "__main__":
    main()
