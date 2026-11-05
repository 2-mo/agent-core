# Contributing

Use Python 3.12 or later and uv:

```bash
uv sync --locked
uv run python examples/offline_tools.py
```

Keep changes focused on the relevant contract: messages in `context`, tool definitions and validation in `tools`, endpoint adaptation in `model`, and execution or state transitions in `runtime`.

For behavior changes, run the affected test files. For example:

```bash
uv run pytest -q tests/test_tool_executor.py
uv run pytest -q tests/test_agent.py
```

Use deterministic fixtures and HTTP mock transports in tests. Document new public parameters and include a small runnable example when it helps explain their purpose. Contributions are provided under the Apache-2.0 license.
