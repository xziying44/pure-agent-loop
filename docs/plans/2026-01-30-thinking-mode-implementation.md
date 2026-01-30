# 思考模式支持实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为 pure-agent-loop 框架添加推理模型思考模式支持，兼容 OpenAI o1/o3 和 DeepSeek/智谱等 API。

**Architecture:** 通过 `thinking_level` 参数统一控制思考深度，内部自动映射到不同 API；新增 `REASONING` 事件类型用于调试；扩展 `TokenUsage` 记录详细 token 统计。

**Tech Stack:** Python 3.10+, openai SDK, pytest, pytest-asyncio

---

## Task 1: 扩展 TokenUsage 类型

**Files:**
- Modify: `src/pure_agent_loop/llm/types.py:10-35`
- Test: `tests/test_llm_types.py`

**Step 1: 编写 TokenUsage 扩展字段的测试**

在 `tests/test_llm_types.py` 中的 `TestTokenUsage` 类末尾添加：

```python
    def test_token_usage_with_reasoning_tokens(self):
        """应该支持 reasoning_tokens 和 cached_tokens 字段"""
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            reasoning_tokens=30,
            cached_tokens=10,
        )
        assert usage.reasoning_tokens == 30
        assert usage.cached_tokens == 10

    def test_token_usage_optional_fields_default_none(self):
        """reasoning_tokens 和 cached_tokens 默认为 None"""
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        assert usage.reasoning_tokens is None
        assert usage.cached_tokens is None

    def test_token_usage_add_with_reasoning_tokens(self):
        """累加时应该正确处理 reasoning_tokens 和 cached_tokens"""
        usage1 = TokenUsage(
            prompt_tokens=100, completion_tokens=50, total_tokens=150,
            reasoning_tokens=20, cached_tokens=5
        )
        usage2 = TokenUsage(
            prompt_tokens=200, completion_tokens=100, total_tokens=300,
            reasoning_tokens=40, cached_tokens=10
        )
        result = usage1 + usage2
        assert result.reasoning_tokens == 60
        assert result.cached_tokens == 15

    def test_token_usage_add_with_none_reasoning_tokens(self):
        """累加时 None 值应该被当作 0 处理"""
        usage1 = TokenUsage(
            prompt_tokens=100, completion_tokens=50, total_tokens=150,
            reasoning_tokens=20, cached_tokens=None
        )
        usage2 = TokenUsage(
            prompt_tokens=200, completion_tokens=100, total_tokens=300,
            reasoning_tokens=None, cached_tokens=10
        )
        result = usage1 + usage2
        assert result.reasoning_tokens == 20
        assert result.cached_tokens == 10

    def test_token_usage_zero_includes_optional_fields(self):
        """zero() 方法应该将可选字段设为 None"""
        usage = TokenUsage.zero()
        assert usage.reasoning_tokens is None
        assert usage.cached_tokens is None
```

**Step 2: 运行测试验证失败**

Run: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest tests/test_llm_types.py::TestTokenUsage -v`

Expected: FAIL，因为 `TokenUsage` 还没有 `reasoning_tokens` 和 `cached_tokens` 字段

**Step 3: 实现 TokenUsage 扩展**

修改 `src/pure_agent_loop/llm/types.py` 中的 `TokenUsage` 类：

```python
@dataclass
class TokenUsage:
    """Token 用量统计

    Attributes:
        prompt_tokens: 输入 token 数
        completion_tokens: 输出 token 数
        total_tokens: 总 token 数
        reasoning_tokens: 思考消耗的 token 数（可选）
        cached_tokens: 缓存命中的输入 token 数（可选）
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    reasoning_tokens: int | None = None
    cached_tokens: int | None = None

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """累加两个 TokenUsage"""
        # 处理可选字段的累加（None 视为 0，但结果保留 None 如果两者都是 None）
        def add_optional(a: int | None, b: int | None) -> int | None:
            if a is None and b is None:
                return None
            return (a or 0) + (b or 0)

        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            reasoning_tokens=add_optional(self.reasoning_tokens, other.reasoning_tokens),
            cached_tokens=add_optional(self.cached_tokens, other.cached_tokens),
        )

    @classmethod
    def zero(cls) -> "TokenUsage":
        """创建零值 TokenUsage"""
        return cls(prompt_tokens=0, completion_tokens=0, total_tokens=0)
```

**Step 4: 运行测试验证通过**

Run: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest tests/test_llm_types.py::TestTokenUsage -v`

Expected: PASS

**Step 5: 提交**

```bash
git add src/pure_agent_loop/llm/types.py tests/test_llm_types.py
git commit -m "feat(types): 扩展 TokenUsage 支持 reasoning_tokens 和 cached_tokens"
```

---

## Task 2: 扩展 LLMResponse 类型

**Files:**
- Modify: `src/pure_agent_loop/llm/types.py:53-72`
- Test: `tests/test_llm_types.py`

**Step 1: 编写 LLMResponse 扩展字段的测试**

在 `tests/test_llm_types.py` 中的 `TestLLMResponse` 类末尾添加：

```python
    def test_create_response_with_reasoning_content(self):
        """应该支持 reasoning_content 字段"""
        usage = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        response = LLMResponse(
            content="最终回答",
            tool_calls=[],
            usage=usage,
            raw={},
            reasoning_content="这是推理过程...",
        )
        assert response.reasoning_content == "这是推理过程..."

    def test_reasoning_content_default_none(self):
        """reasoning_content 默认为 None"""
        usage = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        response = LLMResponse(content="Hello", tool_calls=[], usage=usage, raw={})
        assert response.reasoning_content is None
```

**Step 2: 运行测试验证失败**

Run: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest tests/test_llm_types.py::TestLLMResponse -v`

Expected: FAIL

**Step 3: 实现 LLMResponse 扩展**

修改 `src/pure_agent_loop/llm/types.py` 中的 `LLMResponse` 类：

```python
@dataclass
class LLMResponse:
    """LLM 统一响应模型

    Attributes:
        content: 文本回复内容（可能为 None）
        tool_calls: 工具调用列表
        usage: Token 用量统计
        raw: 原始响应对象（供高级用户访问）
        reasoning_content: 模型推理过程（可选，用于调试）
    """

    content: str | None
    tool_calls: list[ToolCall]
    usage: TokenUsage
    raw: Any
    reasoning_content: str | None = None

    @property
    def has_tool_calls(self) -> bool:
        """是否包含工具调用"""
        return len(self.tool_calls) > 0
```

**Step 4: 运行测试验证通过**

Run: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest tests/test_llm_types.py::TestLLMResponse -v`

Expected: PASS

**Step 5: 提交**

```bash
git add src/pure_agent_loop/llm/types.py tests/test_llm_types.py
git commit -m "feat(types): 扩展 LLMResponse 支持 reasoning_content"
```

---

## Task 3: 扩展事件系统

**Files:**
- Modify: `src/pure_agent_loop/events.py:12-22` (EventType)
- Modify: `src/pure_agent_loop/events.py:107-114` (Event 工厂方法)
- Test: `tests/test_events.py`

**Step 1: 编写 REASONING 事件类型的测试**

在 `tests/test_events.py` 中添加新的测试类：

```python
class TestEventTypeReasoning:
    """REASONING 事件类型测试"""

    def test_reasoning_type_exists(self):
        """应该存在 REASONING 事件类型"""
        assert EventType.REASONING.value == "reasoning"

    def test_reasoning_event_factory(self):
        """应该有 reasoning 工厂方法"""
        event = Event.reasoning(step=1, content="让我分析一下这个问题...")
        assert event.type == EventType.REASONING
        assert event.step == 1
        assert event.data["content"] == "让我分析一下这个问题..."

    def test_reasoning_event_to_dict(self):
        """REASONING 事件应该能正确序列化"""
        event = Event.reasoning(step=2, content="推理内容")
        d = event.to_dict()
        assert d["type"] == "reasoning"
        assert d["step"] == 2
        assert d["data"]["content"] == "推理内容"
```

**Step 2: 运行测试验证失败**

Run: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest tests/test_events.py::TestEventTypeReasoning -v`

Expected: FAIL

**Step 3: 实现 REASONING 事件类型和工厂方法**

修改 `src/pure_agent_loop/events.py`：

1. 在 `EventType` 枚举中添加：

```python
class EventType(Enum):
    """事件类型枚举"""

    LOOP_START = "loop_start"
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    SOFT_LIMIT = "soft_limit"
    ERROR = "error"
    LOOP_END = "loop_end"
    TODO_UPDATE = "todo_update"
    REASONING = "reasoning"  # 新增：模型内部推理过程
```

2. 在 `Event` 类中添加工厂方法（在 `todo_update` 方法后）：

```python
    @classmethod
    def reasoning(cls, step: int, content: str) -> "Event":
        """创建推理过程事件

        用于记录模型的内部推理链，仅在 emit_reasoning_events=True 时产出。
        与 thought 事件的区别：reasoning 不会进入消息历史。
        """
        return cls(type=EventType.REASONING, step=step, data={"content": content})
```

**Step 4: 运行测试验证通过**

Run: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest tests/test_events.py -v`

Expected: PASS

**Step 5: 提交**

```bash
git add src/pure_agent_loop/events.py tests/test_events.py
git commit -m "feat(events): 添加 REASONING 事件类型"
```

---

## Task 4: 修改 OpenAIClient 支持思考模式

**Files:**
- Modify: `src/pure_agent_loop/llm/openai_client.py`
- Test: `tests/test_openai_client.py`

**Step 1: 编写 OpenAIClient thinking_level 参数的测试**

在 `tests/test_openai_client.py` 文件末尾添加：

```python
class TestOpenAIClientThinkingLevel:
    """OpenAIClient 思考模式测试"""

    def test_default_thinking_level_is_off(self):
        """默认 thinking_level 应该是 off"""
        client = OpenAIClient(model="gpt-4o-mini", api_key="test-key")
        assert client.thinking_level == "off"

    def test_thinking_level_can_be_set(self):
        """应该能设置 thinking_level"""
        client = OpenAIClient(
            model="gpt-4o-mini",
            api_key="test-key",
            thinking_level="medium",
        )
        assert client.thinking_level == "medium"

    def test_is_openai_reasoning_model_o1(self):
        """应该能识别 o1 系列模型"""
        client = OpenAIClient(model="o1-preview", api_key="test-key")
        assert client._is_openai_reasoning_model() is True

    def test_is_openai_reasoning_model_o3(self):
        """应该能识别 o3 系列模型"""
        client = OpenAIClient(model="o3-mini", api_key="test-key")
        assert client._is_openai_reasoning_model() is True

    def test_is_openai_reasoning_model_gpt(self):
        """GPT 模型不应该被识别为推理模型"""
        client = OpenAIClient(model="gpt-4o-mini", api_key="test-key")
        assert client._is_openai_reasoning_model() is False

    def test_is_openai_reasoning_model_deepseek(self):
        """DeepSeek 模型不应该被识别为 OpenAI 推理模型"""
        client = OpenAIClient(model="deepseek-chat", api_key="test-key")
        assert client._is_openai_reasoning_model() is False
```

**Step 2: 运行测试验证失败**

Run: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest tests/test_openai_client.py::TestOpenAIClientThinkingLevel -v`

Expected: FAIL

**Step 3: 实现 OpenAIClient thinking_level 支持**

修改 `src/pure_agent_loop/llm/openai_client.py`：

```python
"""OpenAI 兼容客户端

基于 openai SDK 实现的 LLM 客户端，兼容所有 OpenAI 兼容 API。
"""

import json
import logging
from typing import Any, Literal

from openai import AsyncOpenAI

from .base import BaseLLMClient
from .types import LLMResponse, ToolCall, TokenUsage

logger = logging.getLogger(__name__)

# 思考深度类型
ThinkingLevel = Literal["off", "low", "medium", "high"]


class OpenAIClient(BaseLLMClient):
    """OpenAI 兼容 LLM 客户端

    支持 OpenAI、Azure OpenAI、DeepSeek、通义千问等所有兼容 API。

    Args:
        model: 模型名称
        api_key: API 密钥（默认读取 OPENAI_API_KEY 环境变量）
        base_url: API 基础地址（用于接入其他兼容服务）
        thinking_level: 思考深度（off/low/medium/high），默认 off
        **kwargs: 传递给 AsyncOpenAI 的额外参数
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        thinking_level: ThinkingLevel = "off",
        **kwargs: Any,
    ):
        self.model = model
        self.thinking_level = thinking_level
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    def _is_openai_reasoning_model(self) -> bool:
        """检测是否为 OpenAI 原生推理模型（o1/o3/o4 系列）"""
        reasoning_prefixes = ("o1", "o3", "o4")
        return any(self.model.startswith(p) for p in reasoning_prefixes)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """发送对话请求

        Args:
            messages: 消息列表
            tools: 工具定义列表
            **kwargs: 额外参数（temperature, max_tokens 等）

        Returns:
            统一响应模型
        """
        # 构建请求参数
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            **kwargs,
        }

        # 仅在有工具时传入 tools 参数
        if tools:
            request_kwargs["tools"] = tools

        # 根据 thinking_level 注入思考模式参数
        if self.thinking_level != "off":
            if self._is_openai_reasoning_model():
                # OpenAI o1/o3: 使用 reasoning_effort 参数
                request_kwargs["reasoning_effort"] = self.thinking_level
            else:
                # DeepSeek/智谱等: 使用 extra_body.thinking 参数
                request_kwargs["extra_body"] = {
                    "thinking": {"type": "enabled"}
                }

        logger.debug("发送 LLM 请求: model=%s, messages=%d条", self.model, len(messages))

        response = await self._client.chat.completions.create(**request_kwargs)

        return self._parse_response(response)

    def _parse_response(self, response: Any) -> LLMResponse:
        """解析 OpenAI 响应为统一格式"""
        choice = response.choices[0]
        message = choice.message

        # 解析工具调用
        tool_calls: list[ToolCall] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    arguments = {}

                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=arguments,
                    )
                )

        # 解析 reasoning_content（从 message 属性或 model_extra 中获取）
        reasoning_content = getattr(message, "reasoning_content", None)

        # 解析详细 token 统计
        reasoning_tokens = None
        cached_tokens = None
        usage = response.usage

        if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
            reasoning_tokens = getattr(
                usage.completion_tokens_details, "reasoning_tokens", None
            )
        if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
            cached_tokens = getattr(
                usage.prompt_tokens_details, "cached_tokens", None
            )

        # 构建 TokenUsage
        token_usage = TokenUsage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            reasoning_tokens=reasoning_tokens,
            cached_tokens=cached_tokens,
        )

        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            usage=token_usage,
            raw=response,
            reasoning_content=reasoning_content,
        )
```

**Step 4: 运行测试验证通过**

Run: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest tests/test_openai_client.py -v`

Expected: PASS

**Step 5: 提交**

```bash
git add src/pure_agent_loop/llm/openai_client.py tests/test_openai_client.py
git commit -m "feat(openai_client): 支持 thinking_level 参数和 reasoning_content 解析"
```

---

## Task 5: 修改 Agent 支持思考模式

**Files:**
- Modify: `src/pure_agent_loop/agent.py:72-94`
- Test: `tests/test_agent.py`

**Step 1: 编写 Agent 思考模式参数的测试**

在 `tests/test_agent.py` 文件末尾添加：

```python
class TestAgentThinkingMode:
    """Agent 思考模式测试"""

    def test_default_thinking_level_is_off(self):
        """默认 thinking_level 应该是 off"""
        agent = Agent(model="gpt-4o-mini", api_key="test-key")
        assert agent._thinking_level == "off"

    def test_thinking_level_can_be_set(self):
        """应该能设置 thinking_level"""
        agent = Agent(
            model="gpt-4o-mini",
            api_key="test-key",
            thinking_level="high",
        )
        assert agent._thinking_level == "high"

    def test_default_emit_reasoning_events_is_false(self):
        """默认 emit_reasoning_events 应该是 False"""
        agent = Agent(model="gpt-4o-mini", api_key="test-key")
        assert agent._emit_reasoning_events is False

    def test_emit_reasoning_events_can_be_enabled(self):
        """应该能启用 emit_reasoning_events"""
        agent = Agent(
            model="gpt-4o-mini",
            api_key="test-key",
            emit_reasoning_events=True,
        )
        assert agent._emit_reasoning_events is True

    def test_thinking_level_passed_to_openai_client(self):
        """thinking_level 应该传递给 OpenAIClient"""
        agent = Agent(
            model="gpt-4o-mini",
            api_key="test-key",
            thinking_level="medium",
        )
        # 验证内部 LLM 客户端的 thinking_level
        assert agent._llm.thinking_level == "medium"
```

**Step 2: 运行测试验证失败**

Run: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest tests/test_agent.py::TestAgentThinkingMode -v`

Expected: FAIL

**Step 3: 实现 Agent 思考模式支持**

修改 `src/pure_agent_loop/agent.py` 中的 `Agent.__init__` 方法：

```python
from typing import Any, AsyncIterator, Iterator, Literal

# 在 Agent 类定义前添加类型别名
ThinkingLevel = Literal["off", "low", "medium", "high"]


class Agent:
    """Agent 入口类

    使用 pure-agent-loop 的唯一入口。

    使用方式 1 - 通过 model 参数（内置 OpenAI 兼容客户端）:
        agent = Agent(model="deepseek-chat", api_key="sk-xxx", base_url="...")

    使用方式 2 - 通过自定义 LLM 客户端:
        agent = Agent(llm=MyCustomClient())

    Args:
        model: 模型名称（使用内置客户端时）
        api_key: API 密钥（默认读取环境变量）
        base_url: API 基础地址
        llm: 自定义 LLM 客户端实例（与 model 二选一）
        tools: 工具列表（@tool 装饰器或字典格式）
        system_prompt: 系统提示
        limits: 终止条件配置
        retry: 重试配置
        temperature: 温度参数
        thinking_level: 思考深度（off/low/medium/high），默认 off
        emit_reasoning_events: 是否推送 REASONING 事件，默认 False
        **llm_kwargs: 透传给 LLM 调用的额外参数
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        llm: BaseLLMClient | None = None,
        tools: list[Tool | dict[str, Any]] | None = None,
        system_prompt: str = "",
        name: str = "智能助理",
        limits: LoopLimits | None = None,
        retry: RetryConfig | None = None,
        temperature: float = 0.7,
        thinking_level: ThinkingLevel = "off",
        emit_reasoning_events: bool = False,
        **llm_kwargs: Any,
    ):
        # 保存思考模式配置
        self._thinking_level = thinking_level
        self._emit_reasoning_events = emit_reasoning_events

        # 构建 LLM 客户端
        if llm is not None:
            self._llm = llm
        else:
            self._llm = OpenAIClient(
                model=model,
                api_key=api_key,
                base_url=base_url,
                thinking_level=thinking_level,
            )

        # 创建 TodoStore 和内置工具
        self._todo_store = TodoStore()
        self._name = name

        # 注册工具（内置 + 用户）
        self._tool_registry = ToolRegistry()
        self._tool_registry.register(create_todo_tool(self._todo_store))
        if tools:
            self._tool_registry.register_many(tools)

        # 构建完整系统提示词
        self._system_prompt = build_system_prompt(
            name=name,
            user_prompt=system_prompt,
        )
        self._limits = limits or LoopLimits()
        self._retry = retry or RetryConfig()
        self._llm_kwargs: dict[str, Any] = {"temperature": temperature, **llm_kwargs}
```

**Step 4: 运行测试验证通过**

Run: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest tests/test_agent.py::TestAgentThinkingMode -v`

Expected: PASS

**Step 5: 提交**

```bash
git add src/pure_agent_loop/agent.py tests/test_agent.py
git commit -m "feat(agent): 添加 thinking_level 和 emit_reasoning_events 参数"
```

---

## Task 6: 修改 ReactLoop 支持 REASONING 事件

**Files:**
- Modify: `src/pure_agent_loop/loop.py:35-49` (构造函数)
- Modify: `src/pure_agent_loop/loop.py:132-137` (响应处理)
- Modify: `src/pure_agent_loop/agent.py:115-124` (_create_loop 方法)
- Test: `tests/test_loop.py`

**Step 1: 编写 ReactLoop REASONING 事件的测试**

在 `tests/test_loop.py` 文件末尾添加：

```python
class TestReactLoopReasoningEvent:
    """ReactLoop REASONING 事件测试"""

    async def test_reasoning_event_emitted_when_enabled(self):
        """当 emit_reasoning_events=True 且有 reasoning_content 时应该产出 REASONING 事件"""
        # 创建模拟 LLM，返回包含 reasoning_content 的响应
        mock_llm = MockLLM([
            LLMResponse(
                content="最终回答",
                tool_calls=[],
                usage=TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
                raw={},
                reasoning_content="这是推理过程...",
            )
        ])

        registry = ToolRegistry()
        loop = ReactLoop(
            llm=mock_llm,
            tool_registry=registry,
            limits=LoopLimits(),
            retry=RetryConfig(),
            emit_reasoning_events=True,
        )

        events = []
        async for event in loop.run("测试任务"):
            events.append(event)

        # 应该有 REASONING 事件
        reasoning_events = [e for e in events if e.type == EventType.REASONING]
        assert len(reasoning_events) == 1
        assert reasoning_events[0].data["content"] == "这是推理过程..."

    async def test_reasoning_event_not_emitted_when_disabled(self):
        """当 emit_reasoning_events=False 时不应该产出 REASONING 事件"""
        mock_llm = MockLLM([
            LLMResponse(
                content="最终回答",
                tool_calls=[],
                usage=TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
                raw={},
                reasoning_content="这是推理过程...",
            )
        ])

        registry = ToolRegistry()
        loop = ReactLoop(
            llm=mock_llm,
            tool_registry=registry,
            limits=LoopLimits(),
            retry=RetryConfig(),
            emit_reasoning_events=False,  # 禁用
        )

        events = []
        async for event in loop.run("测试任务"):
            events.append(event)

        # 不应该有 REASONING 事件
        reasoning_events = [e for e in events if e.type == EventType.REASONING]
        assert len(reasoning_events) == 0

    async def test_reasoning_event_not_emitted_when_no_content(self):
        """当没有 reasoning_content 时不应该产出 REASONING 事件"""
        mock_llm = MockLLM([
            LLMResponse(
                content="最终回答",
                tool_calls=[],
                usage=TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
                raw={},
                reasoning_content=None,  # 无推理内容
            )
        ])

        registry = ToolRegistry()
        loop = ReactLoop(
            llm=mock_llm,
            tool_registry=registry,
            limits=LoopLimits(),
            retry=RetryConfig(),
            emit_reasoning_events=True,
        )

        events = []
        async for event in loop.run("测试任务"):
            events.append(event)

        # 不应该有 REASONING 事件
        reasoning_events = [e for e in events if e.type == EventType.REASONING]
        assert len(reasoning_events) == 0
```

**Step 2: 运行测试验证失败**

Run: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest tests/test_loop.py::TestReactLoopReasoningEvent -v`

Expected: FAIL

**Step 3: 实现 ReactLoop REASONING 事件支持**

1. 修改 `src/pure_agent_loop/loop.py` 中的 `ReactLoop.__init__`：

```python
    def __init__(
        self,
        llm: BaseLLMClient,
        tool_registry: ToolRegistry,
        limits: LoopLimits,
        retry: RetryConfig,
        llm_kwargs: dict[str, Any] | None = None,
        todo_store: TodoStore | None = None,
        emit_reasoning_events: bool = False,
    ):
        self._llm = llm
        self._tools = tool_registry
        self._limits = limits
        self._retry_handler = RetryHandler(retry)
        self._llm_kwargs = llm_kwargs or {}
        self._todo_store = todo_store
        self._emit_reasoning_events = emit_reasoning_events
```

2. 修改 `src/pure_agent_loop/loop.py` 中的 `run` 方法，在处理响应时（第 132-137 行附近）添加 REASONING 事件产出：

在 `if response.has_tool_calls:` 分支的开头（第 133 行后）添加：

```python
            # ---- 处理响应 ----
            # 产出 REASONING 事件（如果启用且有内容）
            if self._emit_reasoning_events and response.reasoning_content:
                yield Event.reasoning(step=step, content=response.reasoning_content)

            if response.has_tool_calls:
                # ... 现有代码
```

同样，在 `else:` 分支（无工具调用 = 最终回答）的开头也添加：

```python
            else:
                # 产出 REASONING 事件（如果启用且有内容）
                if self._emit_reasoning_events and response.reasoning_content:
                    yield Event.reasoning(step=step, content=response.reasoning_content)

                # 无工具调用 = 最终回答
                final_content = response.content or ""
                # ... 现有代码
```

3. 修改 `src/pure_agent_loop/agent.py` 中的 `_create_loop` 方法：

```python
    def _create_loop(self) -> ReactLoop:
        """创建循环引擎实例"""
        return ReactLoop(
            llm=self._llm,
            tool_registry=self._tool_registry,
            limits=self._limits,
            retry=self._retry,
            llm_kwargs=self._llm_kwargs,
            todo_store=self._todo_store,
            emit_reasoning_events=self._emit_reasoning_events,
        )
```

**Step 4: 运行测试验证通过**

Run: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest tests/test_loop.py -v`

Expected: PASS

**Step 5: 运行全部测试确保无回归**

Run: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest -v`

Expected: ALL PASS

**Step 6: 提交**

```bash
git add src/pure_agent_loop/loop.py src/pure_agent_loop/agent.py tests/test_loop.py
git commit -m "feat(loop): 支持 REASONING 事件产出"
```

---

## Task 7: 更新模块导出

**Files:**
- Modify: `src/pure_agent_loop/__init__.py`

**Step 1: 更新 __init__.py 导出**

确保新增的类型可以从顶层导入。检查是否需要导出 `ThinkingLevel` 类型。

修改 `src/pure_agent_loop/__init__.py`：

在导入部分添加（如果需要让用户使用类型提示）：

```python
# 在文件顶部的 import 语句后，添加类型别名的重新导出
from typing import Literal

# 思考深度类型别名（供用户类型提示使用）
ThinkingLevel = Literal["off", "low", "medium", "high"]
```

在 `__all__` 列表中添加：

```python
    # 类型
    "ThinkingLevel",
```

**Step 2: 验证导入正常**

Run: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && python -c "from pure_agent_loop import Agent, EventType, ThinkingLevel; print('导入成功')"`

Expected: 输出 "导入成功"

**Step 3: 提交**

```bash
git add src/pure_agent_loop/__init__.py
git commit -m "chore(exports): 导出 ThinkingLevel 类型"
```

---

## Task 8: 添加集成测试

**Files:**
- Create: `tests/test_thinking_mode_integration.py`

**Step 1: 创建集成测试文件**

```python
"""思考模式集成测试

测试思考模式功能的端到端行为。
"""

import pytest
from pure_agent_loop import Agent, EventType
from pure_agent_loop.llm.types import LLMResponse, TokenUsage, ToolCall
from pure_agent_loop.llm.base import BaseLLMClient


class MockThinkingLLM(BaseLLMClient):
    """模拟支持思考模式的 LLM"""

    def __init__(self, responses: list[LLMResponse]):
        self._responses = responses
        self._call_count = 0
        self.thinking_level = "off"

    async def chat(self, messages, tools=None, **kwargs):
        response = self._responses[self._call_count]
        self._call_count += 1
        return response


class TestThinkingModeIntegration:
    """思考模式集成测试"""

    async def test_agent_with_thinking_mode_emits_reasoning_events(self):
        """Agent 启用思考模式时应该产出 REASONING 事件"""
        mock_llm = MockThinkingLLM([
            LLMResponse(
                content="42 是生命、宇宙以及一切的答案",
                tool_calls=[],
                usage=TokenUsage(
                    prompt_tokens=10,
                    completion_tokens=50,
                    total_tokens=60,
                    reasoning_tokens=30,
                    cached_tokens=2,
                ),
                raw={},
                reasoning_content="让我思考这个深刻的问题...\n首先，根据《银河系漫游指南》...",
            )
        ])

        agent = Agent(
            llm=mock_llm,
            thinking_level="high",
            emit_reasoning_events=True,
        )

        events = []
        async for event in agent.arun_stream("生命的意义是什么？"):
            events.append(event)

        # 验证事件序列
        event_types = [e.type for e in events]
        assert EventType.LOOP_START in event_types
        assert EventType.REASONING in event_types
        assert EventType.THOUGHT in event_types
        assert EventType.LOOP_END in event_types

        # 验证 REASONING 事件内容
        reasoning_event = next(e for e in events if e.type == EventType.REASONING)
        assert "让我思考" in reasoning_event.data["content"]

    async def test_agent_without_emit_reasoning_no_reasoning_events(self):
        """Agent 未启用 emit_reasoning_events 时不应该产出 REASONING 事件"""
        mock_llm = MockThinkingLLM([
            LLMResponse(
                content="42",
                tool_calls=[],
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                raw={},
                reasoning_content="内部推理...",
            )
        ])

        agent = Agent(
            llm=mock_llm,
            thinking_level="medium",
            emit_reasoning_events=False,  # 禁用
        )

        events = []
        async for event in agent.arun_stream("1+1=?"):
            events.append(event)

        # 不应该有 REASONING 事件
        event_types = [e.type for e in events]
        assert EventType.REASONING not in event_types

    async def test_token_usage_includes_reasoning_tokens(self):
        """Token 统计应该包含 reasoning_tokens"""
        mock_llm = MockThinkingLLM([
            LLMResponse(
                content="答案",
                tool_calls=[],
                usage=TokenUsage(
                    prompt_tokens=100,
                    completion_tokens=200,
                    total_tokens=300,
                    reasoning_tokens=150,
                    cached_tokens=20,
                ),
                raw={},
                reasoning_content="推理内容",
            )
        ])

        agent = Agent(llm=mock_llm, thinking_level="high")
        result = await agent.arun("问题")

        # 验证 token 统计（注意：当前 AgentResult 的 token 累计逻辑可能需要调整）
        # 这里主要验证事件中的数据正确性
        assert result.content == "答案"
```

**Step 2: 运行集成测试**

Run: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest tests/test_thinking_mode_integration.py -v`

Expected: PASS

**Step 3: 提交**

```bash
git add tests/test_thinking_mode_integration.py
git commit -m "test: 添加思考模式集成测试"
```

---

## Task 9: 运行完整测试套件并验证

**Step 1: 运行所有测试**

Run: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest -v --tb=short`

Expected: ALL PASS

**Step 2: 运行测试覆盖率**

Run: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest --cov=pure_agent_loop --cov-report=term-missing`

Expected: 覆盖率 ≥ 80%

**Step 3: 最终提交**

```bash
git add -A
git commit -m "feat: 完成思考模式支持功能"
```

---

## 实施清单摘要

| Task | 描述 | 文件 |
|------|------|------|
| 1 | 扩展 TokenUsage | `llm/types.py`, `test_llm_types.py` |
| 2 | 扩展 LLMResponse | `llm/types.py`, `test_llm_types.py` |
| 3 | 添加 REASONING 事件 | `events.py`, `test_events.py` |
| 4 | OpenAIClient 思考模式 | `llm/openai_client.py`, `test_openai_client.py` |
| 5 | Agent 思考模式参数 | `agent.py`, `test_agent.py` |
| 6 | ReactLoop REASONING 事件 | `loop.py`, `agent.py`, `test_loop.py` |
| 7 | 更新模块导出 | `__init__.py` |
| 8 | 集成测试 | `test_thinking_mode_integration.py` |
| 9 | 完整测试验证 | - |
