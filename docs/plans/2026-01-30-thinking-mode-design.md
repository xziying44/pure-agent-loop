# 思考模式支持设计方案

> 创建日期: 2026-01-30
> 状态: 已确认

## 概述

为 pure-agent-loop 框架添加对推理模型思考模式的支持，兼容 OpenAI o1/o3 系列和 DeepSeek/智谱等 OpenAI 兼容 API。

### 核心目标

1. **统一抽象**：通过 `thinking_level` 参数控制思考深度，内部自动映射到不同 API
2. **事件支持**：新增 `REASONING` 事件类型，用于调试观察模型推理过程
3. **完整统计**：扩展 `TokenUsage` 记录 `reasoning_tokens`、`cached_tokens` 等详细信息

---

## 设计详情

### 1. Agent 接口设计

**新增参数：**

```python
class Agent:
    def __init__(
        self,
        # ... 现有参数 ...

        # 思考模式配置（新增）
        thinking_level: Literal["off", "low", "medium", "high"] = "off",
        emit_reasoning_events: bool = False,
    ):
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `thinking_level` | `Literal["off", "low", "medium", "high"]` | `"off"` | 控制思考深度 |
| `emit_reasoning_events` | `bool` | `False` | 是否推送 REASONING 事件（调试用） |

**thinking_level 取值：**

- `"off"`: 禁用思考模式（默认）
- `"low"`: 低深度思考，快速响应
- `"medium"`: 中等深度思考（推荐）
- `"high"`: 深度思考，适合复杂问题

---

### 2. 事件系统扩展

**新增事件类型：**

```python
class EventType(Enum):
    LOOP_START = "loop_start"
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    SOFT_LIMIT = "soft_limit"
    ERROR = "error"
    LOOP_END = "loop_end"

    # 新增
    REASONING = "reasoning"  # 模型内部推理过程
```

**事件产出逻辑：**

```python
# 在 loop.py 中，解析 LLM 响应后
if emit_reasoning_events and response.reasoning_content:
    yield Event.reasoning(step=step, content=response.reasoning_content)
```

**关键约束：**

- `REASONING` 事件仅在 `emit_reasoning_events=True` 时产出
- `reasoning_content` **不会**追加到消息历史中（符合 DeepSeek/OpenAI 规范）
- 与 `THOUGHT` 事件区分：
  - `THOUGHT`: 模型的显式文本回复（会进入上下文）
  - `REASONING`: 内部推理链（不进入上下文，仅用于调试）

---

### 3. LLM 响应类型扩展

**TokenUsage 扩展：**

```python
@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    # 新增详细统计（可选）
    reasoning_tokens: int | None = None      # 思考消耗的 token
    cached_tokens: int | None = None         # 缓存命中的输入 token
```

**LLMResponse 扩展：**

```python
@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall]
    usage: TokenUsage
    raw: Any

    # 新增
    reasoning_content: str | None = None  # 模型推理过程
```

**解析逻辑（在 `OpenAIClient._parse_response` 中）：**

```python
# 解析 reasoning_content（从 message 或 model_extra 中获取）
reasoning_content = getattr(message, "reasoning_content", None)

# 解析详细 token 统计
reasoning_tokens = None
cached_tokens = None
if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
    reasoning_tokens = usage.completion_tokens_details.reasoning_tokens
if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
    cached_tokens = usage.prompt_tokens_details.cached_tokens
```

---

### 4. thinking_level 映射逻辑

**在 `OpenAIClient` 中实现：**

```python
class OpenAIClient(BaseLLMClient):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        thinking_level: Literal["off", "low", "medium", "high"] = "off",
        **kwargs: Any,
    ):
        self.model = model
        self.thinking_level = thinking_level
        # ...

    async def chat(self, messages, tools, **kwargs) -> LLMResponse:
        request_kwargs = {"model": self.model, "messages": messages, **kwargs}

        # 根据 thinking_level 注入参数
        if self.thinking_level != "off":
            if self._is_openai_reasoning_model():
                # OpenAI o1/o3: 使用 reasoning_effort
                request_kwargs["reasoning_effort"] = self.thinking_level
            else:
                # DeepSeek/智谱等: 使用 extra_body.thinking
                request_kwargs["extra_body"] = {
                    "thinking": {"type": "enabled"}
                }

        response = await self._client.chat.completions.create(**request_kwargs)
        return self._parse_response(response)

    def _is_openai_reasoning_model(self) -> bool:
        """检测是否为 OpenAI 原生推理模型"""
        reasoning_prefixes = ("o1", "o3", "o4")
        return any(self.model.startswith(p) for p in reasoning_prefixes)
```

**映射关系：**

| thinking_level | OpenAI o1/o3 | DeepSeek/智谱 |
|----------------|--------------|---------------|
| `"off"` | 不传参数 | 不传参数 |
| `"low"` | `reasoning_effort="low"` | `thinking={"type":"enabled"}` |
| `"medium"` | `reasoning_effort="medium"` | `thinking={"type":"enabled"}` |
| `"high"` | `reasoning_effort="high"` | `thinking={"type":"enabled"}` |

> 注：DeepSeek/智谱暂不支持精细的思考深度控制，统一启用思考模式。

---

### 5. 使用示例

**基础使用：**

```python
from pure_agent_loop import Agent

# 启用中等深度思考
agent = Agent(
    model="glm-4.7",
    base_url="https://open.bigmodel.cn/api/coding/paas/v4",
    thinking_level="medium",
)

result = agent.run("解释量子纠缠")
print(f"回答: {result.content}")
print(f"总 token: {result.usage.total_tokens}")
print(f"思考 token: {result.usage.reasoning_tokens}")
```

**调试模式（观察推理过程）：**

```python
from pure_agent_loop import Agent, EventType

agent = Agent(
    model="glm-4.7",
    base_url="https://open.bigmodel.cn/api/coding/paas/v4",
    thinking_level="high",
    emit_reasoning_events=True,  # 开启调试
)

async for event in agent.arun_stream("分析这段代码的性能问题"):
    if event.type == EventType.REASONING:
        print(f"[思考] {event.content[:100]}...")
    elif event.type == EventType.THOUGHT:
        print(f"[回复] {event.content}")
```

**查看详细 token 统计：**

```python
result = agent.run("复杂数学问题")
print(f"输入 token: {result.usage.prompt_tokens}")
print(f"输出 token: {result.usage.completion_tokens}")
print(f"思考 token: {result.usage.reasoning_tokens}")
print(f"缓存命中: {result.usage.cached_tokens}")
```

---

## 实现清单

- [ ] 扩展 `TokenUsage`：添加 `reasoning_tokens`、`cached_tokens` 字段
- [ ] 扩展 `LLMResponse`：添加 `reasoning_content` 字段
- [ ] 扩展 `EventType`：添加 `REASONING` 类型
- [ ] 扩展 `Event`：添加 `Event.reasoning()` 工厂方法
- [ ] 修改 `OpenAIClient`：
  - 添加 `thinking_level` 参数
  - 实现 `_is_openai_reasoning_model()` 检测
  - 在 `chat()` 中注入思考参数
  - 在 `_parse_response()` 中解析 `reasoning_content` 和详细 token
- [ ] 修改 `Agent`：添加 `thinking_level`、`emit_reasoning_events` 参数
- [ ] 修改 `ReactLoop`：
  - 传递 `emit_reasoning_events` 配置
  - 在响应处理中产出 `REASONING` 事件
- [ ] 添加测试用例
- [ ] 更新文档和示例

---

## API 响应结构参考

基于智谱 glm-4.7 实际测试结果：

**Message 结构：**
```json
{
  "content": "最终回答",
  "reasoning_content": "推理过程..."
}
```

**Usage 结构：**
```json
{
  "completion_tokens": 422,
  "prompt_tokens": 17,
  "total_tokens": 439,
  "completion_tokens_details": {
    "reasoning_tokens": 412
  },
  "prompt_tokens_details": {
    "cached_tokens": 2
  }
}
```
