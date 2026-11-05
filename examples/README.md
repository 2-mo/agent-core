# Examples

Run these commands from the repository root after `uv sync`.

| Example | Command | Requirements |
| --- | --- | --- |
| Offline tool loop | `uv run python examples/offline_tools.py` | No endpoint; deterministic local stub |
| Tool-using agent | `uv run python examples/tool_agent.py` | A Chat Completions endpoint with tool support |
| Agent delegation | `uv run python examples/delegation.py` | A Chat Completions endpoint with tool support |
| Streaming | `uv run python examples/streaming.py` | A Chat Completions endpoint with streaming support |

Online examples read `MODEL_BASE_URL`, `MODEL_NAME`, and optional `MODEL_API_KEY` from the environment. Their `cfg` object keeps the settings in one place. Model clients live in a `with` block, so connections close after use.

The offline example exercises the same HTTP adapter and Agent runtime through `httpx.MockTransport`. It illustrates the execution mechanics; its scripted answers are not model-generated results.
