# Agent Core

**显式对话、工具契约与有边界的 Agent 执行。**

[English](README.md) · [架构说明](docs/architecture.md) · [API 指南](docs/api.md) · [示例](examples/README.md)

Agent Core 是一个通用 Python Agent 库。将兼容 Chat Completions 的模型与 Python 工具组合，即可运行多轮任务，并获得结构化的执行轨迹。

## 主要能力

| 能力 | 入口 |
| --- | --- |
| 工具调用与参数校验 | 普通 Python 函数、`ToolRegistry`、JSON Schema |
| 显式对话与分叉 | `Conversation`、`fork()`、`snapshot()` |
| 多轮 Agent 执行 | `Agent.run()` |
| Agent 委派 | `Agent.as_tool()` |
| 独立任务并发 | `Agent.run_many()` |
| 流式文本与工具事件 | `Agent.run_stream()` |
| 预算与终止原因 | 轮数、请求数、时间、服务端报告的 token 用量 |
| 轨迹记录 | `RunResult`、`TrajectoryRecord`、JSONL 保存 |

## 快速开始

安装 Python 3.12 及以上版本与 [uv](https://docs.astral.sh/uv/)，然后运行：

```bash
git clone https://github.com/2-mo/agent-core.git
cd agent-core
uv sync
uv run python examples/offline_tools.py
```

离线示例使用明确编写的模型回复，实际执行工具注册、参数校验、调用、预算与轨迹记录，不需要模型账号或网络请求。

连接自己的模型服务时，设置：

```bash
export MODEL_BASE_URL="https://your-provider.example/v1"
export MODEL_NAME="your-model"
export MODEL_API_KEY="your-api-key"
uv run python examples/tool_agent.py
```

环境变量由 shell 管理，程序不会自动加载 `.env`。服务端需要支持使用到的功能，例如工具调用和流式响应。

## 使用方式

`model.ask(...)` 生成一次回答；`model.chat(conversation, ...)` 同时更新显式历史；`agent.run(...)` 执行完整的模型与工具循环。工具主要负责具体业务操作，运行时统一管理消息、预算、调度和轨迹。

```python
from agent_core import Agent


def multiply(a: int, b: int) -> int:
    """精确计算两个整数的乘积。"""
    return a * b


# model 使用 ChatModel 配置，完整代码见 examples/tool_agent.py。
agent = Agent(model, tools=[multiply], max_turns=6, max_calls=8)
result = agent.run("17 乘以 23 是多少？")
print(result.text if result.ok else result.termination.reason)
```

新增工具可先从普通函数开始；需要复杂约束时提供显式 schema，需要可恢复的业务拒绝时返回 `ToolResult(status="failed", ...)`。工具执行异常与预算耗尽都保留明确终止信息。

## 进一步阅读

- [示例说明](examples/README.md)：离线运行、模型接入、委派与流式输出。
- [API 指南](docs/api.md)：工具、历史、策略、预算、并发和轨迹。
- [架构说明](docs/architecture.md)：模块边界与扩展接口。
- [贡献说明](CONTRIBUTING.md)：开发环境与验证方式。

许可证为 [Apache-2.0](LICENSE)。
