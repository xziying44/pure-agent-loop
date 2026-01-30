# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

**pure-agent-loop** — 轻量级 ReAct 模式 Agentic Loop 框架。基于 OpenAI Function Calling 协议，支持任意 OpenAI 兼容 API（DeepSeek、Qwen 等）。Python >=3.10。

## 常用命令

```bash
# 激活虚拟环境（所有命令前必须确认）
source venv/bin/activate

# 安装开发依赖
pip install -e ".[dev]"

# 运行全部测试
pytest

# 运行单个测试文件
pytest tests/test_tool.py

# 运行单个测试函数
pytest tests/test_tool.py::test_tool_decorator_basic -v

# 运行测试（含覆盖率）
pytest --cov=pure_agent_loop --cov-report=term-missing

# 构建包
python -m build
```

## 架构

### ReAct 循环核心流程

```
User Task → Agent → ReactLoop → LLM 调用
                                  ↓
                        [有 tool_calls?]
                        ├─ 是 → 执行工具 → Observation → 回到 LLM
                        └─ 否 → 最终回答 → AgentResult
```

### 模块职责

| 模块 | 职责 |
|------|------|
| `agent.py` | 唯一用户入口。提供 `run()`/`arun()`（阻塞）和 `run_stream()`/`arun_stream()`（流式）四种调用方式 |
| `loop.py` | ReAct 引擎。管理 Thought → Action → Observation 循环，协调 LLM 调用、工具执行和限制检查 |
| `tool.py` | 工具系统。`@tool` 装饰器自动从类型注解和 Google-style docstring 提取 JSON Schema；`ToolRegistry` 管理工具集合 |
| `events.py` | 事件系统。7 种 `EventType`（LOOP_START/THOUGHT/ACTION/OBSERVATION/SOFT_LIMIT/ERROR/LOOP_END），循环每一步产出结构化事件 |
| `limits.py` | 终止控制。`LoopLimits` 定义软限制（max_steps=10, timeout=300s）和硬限制（max_tokens=100000）；`LimitChecker` 在每步检查 |
| `retry.py` | 重试机制。`RetryHandler` 实现指数退避，可配置重试次数和可重试异常类型 |
| `renderer.py` | 事件渲染。装饰器注册自定义渲染器，支持工具级/事件类型级/默认级优先级 |
| `llm/` | LLM 抽象层。`BaseLLMClient`（ABC）→ `OpenAIClient`（AsyncOpenAI 实现）；`LLMResponse`/`ToolCall`/`TokenUsage` 统一类型 |
| `errors.py` | 异常体系。`PureAgentLoopError` 基类 → `ToolExecutionError`/`LLMError`/`LimitExceededError` |

### 关键设计模式

- **依赖注入**：`Agent` 接受自定义 `BaseLLMClient` 实例，测试通过 MockLLM 注入
- **事件驱动**：循环过程通过 `AsyncIterator[Event]` 流式产出，支持实时监控
- **同步/异步双模式**：`Agent` 提供 sync wrapper（内部处理事件循环创建和线程池），async 为原生实现
- **工具错误不中断**：`Tool.execute()` 捕获所有异常，返回格式化错误字符串而非抛出

## 代码规范

- 所有注释和 docstring 使用**中文**
- Docstring 使用 Google 风格（`@tool` 装饰器自动解析 `Args:` 段）
- 类型注解使用 Python 3.10+ 语法（`X | None` 而非 `Optional[X]`，`list[T]` 而非 `List[T]`）
- 测试使用 `pytest` + `pytest-asyncio`（asyncio_mode = "auto"，无需手动标记 `@pytest.mark.asyncio`）

## 测试规范

- 测试文件位于 `tests/`，与源码模块一一对应（`test_agent.py` ↔ `agent.py`）
- LLM 调用通过 `MockLLM`（实现 `BaseLLMClient`）模拟，构造可控的响应序列
- 异步测试直接用 `async def test_xxx()` 定义，pytest-asyncio 自动处理
