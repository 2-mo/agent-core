"""Draft and review an announcement through agents with a shared request budget."""

import os
from types import SimpleNamespace

from agent_core import Agent, ChatModel
from agent_core.context import message_text

cfg = SimpleNamespace(
    base_url=os.environ["MODEL_BASE_URL"],
    model=os.environ["MODEL_NAME"],
    api_key=os.environ.get("MODEL_API_KEY"),
    task="Announce: The meeting has moved to Friday. No time or location has been confirmed.",
)


def main():
    with ChatModel(
        model=cfg.model, base_url=cfg.base_url, api_key=cfg.api_key,
        timeout=120, temperature=0, max_tokens=512,
    ) as model:
        writer = Agent(
            model, name="writer", max_turns=3, max_calls=3,
            instructions="Write a short announcement from the supplied facts. Do not invent details.",
        )
        reviewer = Agent(
            model, name="reviewer", max_turns=3, max_calls=3,
            instructions="Check the draft against the supplied facts. Flag unsupported details and suggest corrections.",
        )
        coordinator = Agent(
            model, name="coordinator",
            tools=[writer.as_tool(), reviewer.as_tool()],
            max_turns=6, max_calls=10,
            instructions=(
                "First send the facts to writer. Then send the original facts and writer's draft "
                "to reviewer. Use the review to return the final announcement."
            ),
        )
        result = coordinator.run(cfg.task)

    # Inspect the actual exchanges recorded during this run.
    for step in result.trajectory.transitions:
        if step.kind == "tool":
            call = step.tool_action["function"]
            print(f"\ncoordinator -> {call['name']}: {call['arguments']}")
            print(f"{call['name']} -> coordinator: {message_text(step.message_delta[0])}")
    print("\nFinal:", result.text if result.ok else result.termination.reason)
    print(f"Success: {result.ok}; total model calls: {result.model_calls}")
    for child in result.children:
        name = child.trajectory.rollout_metadata["agent"]
        print(f"Child {name}: success={child.ok}; model calls={child.model_calls}")


if __name__ == "__main__":
    main()
