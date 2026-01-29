# pure-agent-loop 设计文档

> 轻量级 ReAct 模式 Agentic Loop 框架

## 1. 项目概述

### 目标

构建一个面向**应用开发者**的轻量级 Python 库，利用 ReAct（Reasoning + Acting）模式驱动大模型，实现稳定运行的 Agentic Loop。发布到 PyPI，包名 `pure-agent-loop`。

### 核心特性

- **轻量级**：仅依赖 `openai` SDK，无其他运行时依赖
- **易用性**：`model` + `tools` 两个参数即可启动，零配置门槛
- **稳定性**：软/硬限制双层终止控制，完善的错误处理与重试机制
- **可扩展**：支持自定义 LLM 客户端、渲染器、工具定义方式
- **同步/异步双支持**：`run()` / `arun()` / `run_stream()` / `arun_stream()`

### 技术栈

- Python >= 3.10
- 运行时依赖：`openai>=1.0.0`
- 构建工具：`hatchling`

---

## 2. 整体架构

```
┌─────────────────────────────────────────────┐
│              pure-agent-loop                │
├─────────────────────────────────────────────┤
│                                             │
│  ┌─────────┐    ┌──────────┐    ┌────────┐ │
│  │  Agent   │───▶│  Loop    │───▶│ Events │ │
│  │ (入口)   │    │ (核心引擎)│    │ (事件流)│ │
│  └─────────┘    └──────────┘    └────────┘ │
│       │              │              │       │
│       ▼              ▼              ▼       │
│  ┌─────────┐    ┌──────────┐    ┌────────┐ │
│  │  Tools  │    │  LLM     │    │Renderer│ │
│  │ (工具层) │    │ (模型层)  │    │(渲染层) │ │
│  └─────────┘    └──────────┘    └────────┘ │
│                      │                      │
│                      ▼                      │
│               ┌──────────┐                  │
│               │  Limits  │                  │
│               │(终止控制) │                  │
│               └──────────┘                  │
└─────────────────────────────────────────────┘
```

### 模块职责

| 模块 | 职责 | 文件 |
|------|------|------|
| **Agent** | 用户唯一入口，组装所有配置 | `agent.py` |
| **Loop** | ReAct 循环引擎 | `loop.py` |
| **Tools** | 工具注册与管理 | `tool.py` |
| **LLM** | 模型调用抽象层 | `llm/` |
| **Events** | 结构化 JSON 事件定义 | `events.py` |
| **Renderer** | 事件渲染层 | `renderer.py` |
| **Limits** | 终止条件管理 | `limits.py` |
| **Retry** | 重试与退避 | `retry.py` |

---

## 3. ReAct 循环核心流程

```
用户输入
   │
   ▼
┌──────────────────────────────────────────────────┐
│                  ReAct Loop                       │
│                                                   │
│  ┌─────────┐    ┌─────────┐    ┌──────────────┐  │
│  │ Thought │───▶│ Action  │───▶│ Observation  │  │
│  │ (思考)   │    │ (行动)   │    │ (观察结果)    │  │
│  └─────────┘    └─────────┘    └──────────────┘  │
│       ▲                              │            │
│       │         ┌─────────┐          │            │
│       └─────────│ Limits  │◀─────────┘            │
│                 │  Check  │                       │
│                 └─────────┘                       │
│                      │                            │
│              ┌───────┴───────┐                    │
│              ▼               ▼                    │
│         [软限制触发]     [硬限制/结束]              │
│         注入提示继续       终止循环                 │
└──────────────────────────────────────────────────┘
                       │
                       ▼
                  最终结果输出
```

### 单步执行流程

1. **Thought** - 将历史消息发送给 LLM，LLM 返回思考过程和决策
2. **Action** - 若 LLM 决定调用工具，解析 tool_calls 并执行对应函数
3. **Observation** - 将工具返回值作为 observation 追加到消息历史
4. **Limits Check** - 检查终止条件

### 消息历史结构（OpenAI Chat 格式）

```python
messages = [
    {"role": "system", "content": "你是一个..."},
    {"role": "user", "content": "用户任务"},
    {"role": "assistant", "content": "...", "tool_calls": [...]},
    {"role": "tool", "content": "...", "tool_call_id": "..."},
    {"role": "system", "content": "⚠️ 已超过最大步数..."},  # 软限制注入
]
```

---

## 4. 工具系统

### 装饰器定义（推荐）

```python
from pure_agent_loop import tool

@tool
def search(query: str, max_results: int = 5) -> str:
    """搜索网页内容

    Args:
        query: 搜索关键词
        max_results: 最大返回结果数
    """
    return requests.get(f"...{query}").text
```

装饰器自动完成：
- 从类型注解提取参数 JSON Schema
- 从 docstring 提取工具描述和参数描述（支持 Google / NumPy 风格）
- 自动处理 `Optional`、`Literal`、`Enum`、默认值
- 支持同步和异步函数

### 字典格式（兼容）

```python
tools = [{
    "name": "search",
    "description": "搜索网页内容",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词"}
        },
        "required": ["query"]
    },
    "function": search_fn
}]
```

### ToolRegistry

```python
class ToolRegistry:
    def register(self, tool)          # 注册单个工具
    def register_many(self, tools)    # 批量注册
    def get(self, name) -> Tool       # 按名称获取
    def to_openai_schema() -> list    # 转换为 OpenAI tools 格式
    def execute(self, name, args)     # 执行工具，包装错误处理
```

---

## 5. LLM 抽象层

### 抽象接口

```python
from abc import ABC, abstractmethod

class BaseLLMClient(ABC):
    @abstractmethod
    async def chat(self, messages: list, tools: list | None = None, **kwargs) -> LLMResponse:
        ...
```

### 内置 OpenAI 兼容实现

```python
class OpenAIClient(BaseLLMClient):
    def __init__(self, model: str, api_key: str = None, base_url: str = None, **kwargs):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, **kwargs)

    async def chat(self, messages, tools=None, **kwargs) -> LLMResponse:
        response = await self.client.chat.completions.create(
            model=self.model, messages=messages, tools=tools, **kwargs
        )
        return LLMResponse.from_openai(response)
```

### 统一响应模型

```python
@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall]
    usage: TokenUsage
    raw: Any

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict

@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
```

---

## 6. 事件系统与渲染层

### 事件类型

```python
class EventType(Enum):
    LOOP_START = "loop_start"
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    SOFT_LIMIT = "soft_limit"
    ERROR = "error"
    LOOP_END = "loop_end"

@dataclass
class Event:
    type: EventType
    step: int
    timestamp: float
    data: dict

    def to_dict(self) -> dict: ...
```

### 事件流使用

```python
# 同步流式
for event in agent.run_stream("任务"):
    print(event.to_dict())

# 异步流式
async for event in agent.arun_stream("任务"):
    print(event.to_dict())

# 事后查看
result = agent.run("任务")
for event in result.events:
    print(event.to_dict())
```

### 渲染装饰器

```python
renderer = Renderer()

@renderer.on_tool("search")
def render_search(event: Event) -> str:
    return f"🔍 搜索: {event.data['args']['query']}"

@renderer.on_event(EventType.SOFT_LIMIT)
def render_limit(event: Event) -> str:
    return f"⚠️ {event.data['reason']}, AI 正在调整策略..."

# 使用渲染器
for event in agent.run_stream("任务"):
    print(renderer.render(event))
```

### 渲染器匹配优先级

1. 工具专用规则（`@renderer.on_tool("search")`）
2. 事件类型规则（`@renderer.on_event(EventType.ACTION)`）
3. 内置默认规则

---

## 7. 终止控制（Limits）

### 限制类型与行为

| 限制类型 | 触发后行为 |
|---------|-----------|
| **软限制** - `max_steps` / `timeout` | 通知 AI 调整策略，重置周期计数器，继续循环 |
| **硬限制** - `max_tokens` / 模型自主结束 | 强制终止循环 |

### 配置

```python
@dataclass
class LoopLimits:
    max_steps: int = 10
    timeout: float = 300.0
    max_tokens: int = 100_000

    step_limit_prompt: str = (
        "⚠️ 你已完成第 {checkpoint_steps} 步检查点（总步数: {current_steps}）。\n"
        "请评估当前进度:\n"
        "- 如果接近完成，请尽快总结答案\n"
        "- 如果陷入循环，请调整策略或尝试其他方法\n"
        "- 如果需要更多步骤，请继续但保持高效"
    )
    timeout_prompt: str = (
        "执行已超时（已用 {elapsed:.1f}s / 上限 {timeout}s），"
        "请立即总结当前已获得的信息并给出最终答案。"
    )
```

### 周期性检查点机制

```python
class LimitChecker:
    def __init__(self, limits: LoopLimits):
        self._step_checkpoint = 0
        self._timeout_checkpoint = 0

    def check(self) -> LimitResult:
        # 硬限制优先
        if self.total_tokens >= self.limits.max_tokens:
            return LimitResult(action="stop", reason="token_limit")

        # 软限制：每 max_steps 步触发一次
        steps_since_checkpoint = self.current_step - self._step_checkpoint
        if steps_since_checkpoint >= self.limits.max_steps:
            self._step_checkpoint = self.current_step  # 重置
            return LimitResult(action="warn", ...)

        # 软限制：每 timeout 秒触发一次
        elapsed = time.time() - self.start_time
        if elapsed - self._timeout_checkpoint >= self.limits.timeout:
            self._timeout_checkpoint = elapsed  # 重置
            return LimitResult(action="warn", ...)

        return LimitResult(action="continue")
```

---

## 8. 错误处理与重试

### 两层策略

| 错误类型 | 处理方式 |
|---------|---------|
| **LLM API 错误**（网络/限流） | 自动重试，指数退避 |
| **工具执行错误**（业务逻辑/运行时） | 格式化后反馈给 AI |
| **致命错误**（重试耗尽） | 产出 ERROR + LOOP_END 事件，优雅退出 |

### 重试配置

```python
@dataclass
class RetryConfig:
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
```

### 工具错误反馈

```python
# 工具执行失败时，返回给 AI 的格式：
"⚠️ 工具 '{name}' 执行失败:\n"
"错误类型: {type(e).__name__}\n"
"错误信息: {str(e)}\n"
"请尝试调整参数或使用其他方法。"
```

---

## 9. Agent 入口 API

### 构造参数

```python
class Agent:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        llm: BaseLLMClient | None = None,     # 自定义客户端（二选一）
        tools: list = None,
        system_prompt: str = "You are a helpful assistant.",
        limits: LoopLimits | None = None,
        retry: RetryConfig | None = None,
        temperature: float = 0.7,
        **llm_kwargs,
    ):
```

### 执行方式

```python
# 同步
result: AgentResult = agent.run("任务")

# 异步
result: AgentResult = await agent.arun("任务")

# 同步流式
for event in agent.run_stream("任务"):
    ...

# 异步流式
async for event in agent.arun_stream("任务"):
    ...
```

### 返回结果

```python
@dataclass
class AgentResult:
    content: str
    steps: int
    total_tokens: TokenUsage
    events: list[Event]
    stop_reason: str             # "completed" | "token_limit"
    messages: list[dict]         # 可用于续接对话
```

### 多轮对话

```python
r1 = agent.run("今天天气怎么样？")
r2 = agent.run("那明天呢？", messages=r1.messages)
```

---

## 10. 项目结构

```
pure-agent-loop/
├── src/
│   └── pure_agent_loop/
│       ├── __init__.py
│       ├── agent.py
│       ├── loop.py
│       ├── tool.py
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── openai_client.py
│       │   └── types.py
│       ├── events.py
│       ├── renderer.py
│       ├── limits.py
│       ├── retry.py
│       └── errors.py
├── tests/
│   ├── test_agent.py
│   ├── test_loop.py
│   ├── test_tool.py
│   ├── test_llm.py
│   ├── test_events.py
│   ├── test_renderer.py
│   ├── test_limits.py
│   └── test_retry.py
├── examples/
│   ├── basic.py
│   ├── streaming.py
│   ├── custom_renderer.py
│   └── custom_llm.py
├── pyproject.toml
├── README.md
└── LICENSE
```

### 依赖

```toml
[project]
name = "pure-agent-loop"
version = "0.1.0"
description = "轻量级 ReAct 模式 Agentic Loop 框架"
requires-python = ">=3.10"
dependencies = ["openai>=1.0.0"]
```

---

## 11. TODO（后续迭代）

- [ ] 上下文管理机制：token 接近上限时自动压缩/摘要历史消息
- [ ] 更多内置渲染器模板
- [ ] 工具并行执行支持
- [ ] 会话持久化（JSONL）
