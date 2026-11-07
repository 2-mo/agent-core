# Examples

Run these commands from the repository root after `uv sync`.

| Example | Command | Requirements |
| --- | --- | --- |
| Offline tool loop | `uv run python examples/offline_tools.py` | No endpoint; deterministic local stub |
| Local document catalog | `uv run python examples/local_catalog.py` | No endpoint; bundled Markdown documents and a local stub |
| Tool-using agent | `uv run python examples/tool_agent.py` | A Chat Completions endpoint with tool support |
| Multi-agent writing and review | `uv run python examples/delegation.py` | A Chat Completions endpoint with tool support |
| Streaming | `uv run python examples/streaming.py` | A Chat Completions endpoint with streaming support |

Online examples read `MODEL_BASE_URL`, `MODEL_NAME`, and optional `MODEL_API_KEY` from the environment. Their `cfg` object keeps the settings in one place. Model clients live in a `with` block, so connections close after use.

The offline examples exercise the same HTTP adapter and Agent runtime through `httpx.MockTransport`. They illustrate the execution mechanics; their scripted answers are not model-generated results.

## Coordinate writing and review

`delegation.py` gives a coordinator two agent tools: `writer` drafts an announcement, and `reviewer` checks the draft against the original facts. The coordinator is instructed to pass the facts and draft explicitly, then apply the review to produce the final announcement. Each child has its own conversation; the agents share a `ChatModel` and the parent's overall request budget of 10 model calls.

```bash
uv run python examples/delegation.py
```

After the run, the console prints the actual `coordinator -> writer/reviewer` task arguments and replies recorded in the parent trajectory, the final answer or termination reason, the total model request count, and each child's status. The model chooses the calls, so wording, order, and counts may vary. Inspect `result.children` for each child's full trajectory; the parent's `result.model_calls` already includes their requests.

For local serving commands and configuration advice, see [Model backend deployment](../README.md#model-backend-deployment) or [模型后端部署建议](../README.zh-CN.md#模型后端部署建议).

## Search a local document catalog

`local_catalog.py` reads the small documents in [catalog/](catalog/). The first scripted tool call supplies `limit="2"`, which fails schema validation before the search function runs. The next call supplies the integer `2`, performs the actual file search, and returns matching document contents. The final scripted reply lists the files returned by the tool.

```bash
uv run python examples/local_catalog.py
```

The default query finds `handoff.md` and `review.md`. The console shows a failed tool call followed by a successful one, with three model calls in total. Settings live in `cfg`; change its directory or query to search your own Markdown documents.

Each run appends a trajectory to `outputs/catalog-runs.jsonl`. Inspect the latest run without printing the full record:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

record = json.loads(Path("outputs/catalog-runs.jsonl").read_text().splitlines()[-1])
for step in record["transitions"]:
    print(step["kind"], step["ok"], step["error"])
PY
```

To connect a real model, reuse `search_tool` with a configured `ChatModel` as shown in [tool_agent.py](tool_agent.py). The correction sequence above is scripted for the offline demonstration; a real model chooses its own calls.
