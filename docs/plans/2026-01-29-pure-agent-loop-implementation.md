# pure-agent-loop 实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 构建一个轻量级的 ReAct 模式 Agentic Loop 框架，发布到 PyPI。

**Architecture:** 采用分层架构，Agent 作为用户入口，Loop 作为核心引擎，通过 Events 事件流输出执行过程。支持软/硬限制双层终止控制，工具系统支持装饰器和字典两种定义方式。

**Tech Stack:** Python 3.10+, openai SDK, hatchling (构建工具), pytest (测试)

---

## Task 1: 项目骨架

**Files:**
- Create: `pyproject.toml`
- Create: `src/pure_agent_loop/__init__.py`
- Create: `src/pure_agent_loop/llm/__init__.py`
- Create: `tests/__init__.py`
- Create: `README.md`
- Create: `LICENSE`

**Step 1: 创建目录结构**

```bash
mkdir -p src/pure_agent_loop/llm tests examples docs/plans
```

**Step 2: 创建 pyproject.toml**

```toml
[project]
name = "pure-agent-loop"
version = "0.1.0"
description = "轻量级 ReAct 模式 Agentic Loop 框架"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.10"
authors = [
    {name = "Your Name", email = "your@email.com"}
]
keywords = ["agent", "llm", "react", "agentic", "loop"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
dependencies = [
    "openai>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-cov>=5.0",
]

[project.urls]
Homepage = "https://github.com/yourusername/pure-agent-loop"
Repository = "https://github.com/yourusername/pure-agent-loop"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/pure_agent_loop"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
```

**Step 3: 创建空的 __init__.py 文件**

`src/pure_agent_loop/__init__.py`:
```python
"""pure-agent-loop: 轻量级 ReAct 模式 Agentic Loop 框架"""

__version__ = "0.1.0"
```

`src/pure_agent_loop/llm/__init__.py`:
```python
"""LLM 抽象层模块"""
```

`tests/__init__.py`:
```python
"""测试模块"""
```

**Step 4: 创建 README.md**

```markdown
# pure-agent-loop

轻量级 ReAct 模式 Agentic Loop 框架。

## 安装

```bash
pip install pure-agent-loop
```

## 快速开始

```python
from pure_agent_loop import Agent, tool

@tool
def search(query: str) -> str:
    """搜索网页"""
    return f"搜索结果: {query}"

agent = Agent(model="gpt-4o-mini", tools=[search])
result = agent.run("帮我搜索 Python 教程")
print(result.content)
```

## 许可证

MIT
```

**Step 5: 创建 LICENSE**

```text
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

**Step 6: 验证项目结构**

```bash
ls -la src/pure_agent_loop/
```

Expected: 显示 `__init__.py` 和 `llm/` 目录

**Step 7: 创建虚拟环境并安装开发依赖**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

**Step 8: 验证 pytest 可运行**

```bash
pytest --version
```

Expected: 显示 pytest 版本号

**Step 9: Commit**

```bash
git init
git add .
git commit -m "chore: 初始化项目骨架"
```

---

## Task 2: 自定义异常 (errors.py)

**Files:**
- Create: `src/pure_agent_loop/errors.py`
- Create: `tests/test_errors.py`

**Step 1: 编写失败的测试**

`tests/test_errors.py`:
```python
"""错误类型测试"""

import pytest
from pure_agent_loop.errors import (
    PureAgentLoopError,
    ToolExecutionError,
    LLMError,
    LimitExceededError,
)


class TestErrors:
    """异常类测试"""

    def test_base_error_is_exception(self):
        """基础异常应该继承自 Exception"""
        error = PureAgentLoopError("test error")
        assert isinstance(error, Exception)
        assert str(error) == "test error"

    def test_tool_execution_error(self):
        """工具执行异常应该包含工具名称"""
        error = ToolExecutionError("search", "connection timeout")
        assert error.tool_name == "search"
        assert "search" in str(error)
        assert "connection timeout" in str(error)

    def test_llm_error(self):
        """LLM 异常应该包含原始异常"""
        original = ValueError("rate limit")
        error = LLMError("API 调用失败", original)
        assert error.original_error == original
        assert "API 调用失败" in str(error)

    def test_limit_exceeded_error(self):
        """限制超出异常应该包含限制类型"""
        error = LimitExceededError("token_limit", 100000, 150000)
        assert error.limit_type == "token_limit"
        assert error.limit_value == 100000
        assert error.current_value == 150000
```

**Step 2: 运行测试验证失败**

```bash
pytest tests/test_errors.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'pure_agent_loop.errors'"

**Step 3: 实现 errors.py**

`src/pure_agent_loop/errors.py`:
```python
"""自定义异常类型

定义 pure-agent-loop 框架使用的所有异常类型。
"""

from typing import Any


class PureAgentLoopError(Exception):
    """pure-agent-loop 框架的基础异常类"""

    pass


class ToolExecutionError(PureAgentLoopError):
    """工具执行失败异常

    当工具函数执行过程中发生错误时抛出。
    """

    def __init__(self, tool_name: str, message: str):
        self.tool_name = tool_name
        self.message = message
        super().__init__(f"工具 '{tool_name}' 执行失败: {message}")


class LLMError(PureAgentLoopError):
    """LLM 调用异常

    当 LLM API 调用失败且重试耗尽时抛出。
    """

    def __init__(self, message: str, original_error: Exception | None = None):
        self.original_error = original_error
        super().__init__(message)


class LimitExceededError(PureAgentLoopError):
    """限制超出异常

    当硬限制被触发时抛出。
    """

    def __init__(self, limit_type: str, limit_value: Any, current_value: Any):
        self.limit_type = limit_type
        self.limit_value = limit_value
        self.current_value = current_value
        super().__init__(
            f"超出 {limit_type} 限制: 当前 {current_value}, 上限 {limit_value}"
        )
```

**Step 4: 运行测试验证通过**

```bash
pytest tests/test_errors.py -v
```

Expected: 4 passed

**Step 5: Commit**

```bash
git add src/pure_agent_loop/errors.py tests/test_errors.py
git commit -m "feat: 添加自定义异常类型"
```

---

## Task 3: LLM 类型定义 (llm/types.py)

**Files:**
- Create: `src/pure_agent_loop/llm/types.py`
- Create: `tests/test_llm_types.py`

**Step 1: 编写失败的测试**

`tests/test_llm_types.py`:
```python
"""LLM 类型定义测试"""

import pytest
from pure_agent_loop.llm.types import TokenUsage, ToolCall, LLMResponse


class TestTokenUsage:
    """TokenUsage 测试"""

    def test_create_token_usage(self):
        """应该能创建 TokenUsage 实例"""
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150

    def test_token_usage_add(self):
        """应该能累加两个 TokenUsage"""
        usage1 = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        usage2 = TokenUsage(prompt_tokens=200, completion_tokens=100, total_tokens=300)
        result = usage1 + usage2
        assert result.prompt_tokens == 300
        assert result.completion_tokens == 150
        assert result.total_tokens == 450

    def test_token_usage_zero(self):
        """应该有零值工厂方法"""
        usage = TokenUsage.zero()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0


class TestToolCall:
    """ToolCall 测试"""

    def test_create_tool_call(self):
        """应该能创建 ToolCall 实例"""
        call = ToolCall(id="call_123", name="search", arguments={"query": "python"})
        assert call.id == "call_123"
        assert call.name == "search"
        assert call.arguments == {"query": "python"}


class TestLLMResponse:
    """LLMResponse 测试"""

    def test_create_response_with_content(self):
        """应该能创建包含文本内容的响应"""
        usage = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        response = LLMResponse(content="Hello", tool_calls=[], usage=usage, raw={})
        assert response.content == "Hello"
        assert response.tool_calls == []
        assert response.has_tool_calls is False

    def test_create_response_with_tool_calls(self):
        """应该能创建包含工具调用的响应"""
        usage = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        tool_call = ToolCall(id="call_1", name="search", arguments={"q": "test"})
        response = LLMResponse(
            content=None, tool_calls=[tool_call], usage=usage, raw={}
        )
        assert response.content is None
        assert len(response.tool_calls) == 1
        assert response.has_tool_calls is True
```

**Step 2: 运行测试验证失败**

```bash
pytest tests/test_llm_types.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: 实现 llm/types.py**

`src/pure_agent_loop/llm/types.py`:
```python
"""LLM 相关类型定义

定义 LLM 调用的统一响应模型，屏蔽不同提供商的差异。
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TokenUsage:
    """Token 用量统计

    Attributes:
        prompt_tokens: 输入 token 数
        completion_tokens: 输出 token 数
        total_tokens: 总 token 数
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """累加两个 TokenUsage"""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )

    @classmethod
    def zero(cls) -> "TokenUsage":
        """创建零值 TokenUsage"""
        return cls(prompt_tokens=0, completion_tokens=0, total_tokens=0)


@dataclass
class ToolCall:
    """工具调用信息

    Attributes:
        id: 调用 ID（用于匹配 tool 消息）
        name: 工具名称
        arguments: 解析后的参数字典
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """LLM 统一响应模型

    Attributes:
        content: 文本回复内容（可能为 None）
        tool_calls: 工具调用列表
        usage: Token 用量统计
        raw: 原始响应对象（供高级用户访问）
    """

    content: str | None
    tool_calls: list[ToolCall]
    usage: TokenUsage
    raw: Any

    @property
    def has_tool_calls(self) -> bool:
        """是否包含工具调用"""
        return len(self.tool_calls) > 0
```

**Step 4: 运行测试验证通过**

```bash
pytest tests/test_llm_types.py -v
```

Expected: 6 passed

**Step 5: Commit**

```bash
git add src/pure_agent_loop/llm/types.py tests/test_llm_types.py
git commit -m "feat: 添加 LLM 类型定义 (TokenUsage, ToolCall, LLMResponse)"
```

---

## Task 4: 事件系统 (events.py)

**Files:**
- Create: `src/pure_agent_loop/events.py`
- Create: `tests/test_events.py`

**Step 1: 编写失败的测试**

`tests/test_events.py`:
```python
"""事件系统测试"""

import time
import pytest
from pure_agent_loop.events import Event, EventType


class TestEventType:
    """事件类型枚举测试"""

    def test_all_event_types_exist(self):
        """应该包含所有预定义事件类型"""
        assert EventType.LOOP_START.value == "loop_start"
        assert EventType.THOUGHT.value == "thought"
        assert EventType.ACTION.value == "action"
        assert EventType.OBSERVATION.value == "observation"
        assert EventType.SOFT_LIMIT.value == "soft_limit"
        assert EventType.ERROR.value == "error"
        assert EventType.LOOP_END.value == "loop_end"


class TestEvent:
    """事件对象测试"""

    def test_create_event(self):
        """应该能创建事件实例"""
        event = Event(
            type=EventType.THOUGHT,
            step=1,
            data={"content": "我需要搜索"},
        )
        assert event.type == EventType.THOUGHT
        assert event.step == 1
        assert event.data["content"] == "我需要搜索"
        assert isinstance(event.timestamp, float)

    def test_event_to_dict(self):
        """应该能转换为字典"""
        event = Event(
            type=EventType.ACTION,
            step=2,
            data={"tool": "search", "args": {"query": "python"}},
        )
        d = event.to_dict()
        assert d["type"] == "action"
        assert d["step"] == 2
        assert d["data"]["tool"] == "search"
        assert "timestamp" in d

    def test_event_factory_methods(self):
        """应该有便捷的工厂方法"""
        event = Event.thought(step=1, content="思考中")
        assert event.type == EventType.THOUGHT
        assert event.data["content"] == "思考中"

        event = Event.action(step=2, tool="search", args={"q": "test"})
        assert event.type == EventType.ACTION
        assert event.data["tool"] == "search"
        assert event.data["args"] == {"q": "test"}

        event = Event.observation(step=2, tool="search", result="结果", duration=1.5)
        assert event.type == EventType.OBSERVATION
        assert event.data["tool"] == "search"
        assert event.data["result"] == "结果"
        assert event.data["duration"] == 1.5

        event = Event.error(step=3, error="出错了", fatal=False)
        assert event.type == EventType.ERROR
        assert event.data["error"] == "出错了"
        assert event.data["fatal"] is False

        event = Event.loop_start(task="测试任务")
        assert event.type == EventType.LOOP_START
        assert event.data["task"] == "测试任务"

        event = Event.loop_end(step=5, stop_reason="completed", content="最终结果")
        assert event.type == EventType.LOOP_END
        assert event.data["stop_reason"] == "completed"

        event = Event.soft_limit(step=10, reason="step_limit", prompt="请调整")
        assert event.type == EventType.SOFT_LIMIT
        assert event.data["reason"] == "step_limit"
```

**Step 2: 运行测试验证失败**

```bash
pytest tests/test_events.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: 实现 events.py**

`src/pure_agent_loop/events.py`:
```python
"""事件系统

定义 Agentic Loop 执行过程中产出的结构化事件。
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(Enum):
    """事件类型枚举"""

    LOOP_START = "loop_start"
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    SOFT_LIMIT = "soft_limit"
    ERROR = "error"
    LOOP_END = "loop_end"


@dataclass
class Event:
    """结构化事件

    Attributes:
        type: 事件类型
        step: 当前步数
        data: 事件数据
        timestamp: 事件产生时间戳
    """

    type: EventType
    step: int
    data: dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 可序列化字典"""
        return {
            "type": self.type.value,
            "step": self.step,
            "data": self.data,
            "timestamp": self.timestamp,
        }

    # ---- 工厂方法 ----

    @classmethod
    def thought(cls, step: int, content: str) -> "Event":
        """创建思考事件"""
        return cls(type=EventType.THOUGHT, step=step, data={"content": content})

    @classmethod
    def action(cls, step: int, tool: str, args: dict[str, Any]) -> "Event":
        """创建工具调用事件"""
        return cls(type=EventType.ACTION, step=step, data={"tool": tool, "args": args})

    @classmethod
    def observation(
        cls, step: int, tool: str, result: str, duration: float
    ) -> "Event":
        """创建观察结果事件"""
        return cls(
            type=EventType.OBSERVATION,
            step=step,
            data={"tool": tool, "result": result, "duration": duration},
        )

    @classmethod
    def error(cls, step: int, error: str, fatal: bool = False) -> "Event":
        """创建错误事件"""
        return cls(
            type=EventType.ERROR,
            step=step,
            data={"error": error, "fatal": fatal},
        )

    @classmethod
    def loop_start(cls, task: str) -> "Event":
        """创建循环启动事件"""
        return cls(type=EventType.LOOP_START, step=0, data={"task": task})

    @classmethod
    def loop_end(cls, step: int, stop_reason: str, content: str = "") -> "Event":
        """创建循环结束事件"""
        return cls(
            type=EventType.LOOP_END,
            step=step,
            data={"stop_reason": stop_reason, "content": content},
        )

    @classmethod
    def soft_limit(cls, step: int, reason: str, prompt: str) -> "Event":
        """创建软限制事件"""
        return cls(
            type=EventType.SOFT_LIMIT,
            step=step,
            data={"reason": reason, "prompt": prompt},
        )
```

**Step 4: 运行测试验证通过**

```bash
pytest tests/test_events.py -v
```

Expected: 4 passed

**Step 5: Commit**

```bash
git add src/pure_agent_loop/events.py tests/test_events.py
git commit -m "feat: 添加事件系统 (Event, EventType)"
```

---

## Task 5: 终止控制 (limits.py)

**Files:**
- Create: `src/pure_agent_loop/limits.py`
- Create: `tests/test_limits.py`

**Step 1: 编写失败的测试**

`tests/test_limits.py`:
```python
"""终止控制测试"""

import time
import pytest
from pure_agent_loop.limits import LoopLimits, LimitChecker, LimitResult


class TestLoopLimits:
    """LoopLimits 配置测试"""

    def test_default_values(self):
        """应该有合理的默认值"""
        limits = LoopLimits()
        assert limits.max_steps == 10
        assert limits.timeout == 300.0
        assert limits.max_tokens == 100_000

    def test_custom_values(self):
        """应该支持自定义值"""
        limits = LoopLimits(max_steps=5, timeout=60.0, max_tokens=50_000)
        assert limits.max_steps == 5
        assert limits.timeout == 60.0
        assert limits.max_tokens == 50_000


class TestLimitResult:
    """LimitResult 测试"""

    def test_continue_result(self):
        """继续执行的结果"""
        result = LimitResult.continue_running()
        assert result.action == "continue"
        assert result.reason == ""
        assert result.prompt == ""

    def test_warn_result(self):
        """软限制警告结果"""
        result = LimitResult.warn("step_limit", "请调整策略")
        assert result.action == "warn"
        assert result.reason == "step_limit"
        assert result.prompt == "请调整策略"

    def test_stop_result(self):
        """硬限制停止结果"""
        result = LimitResult.stop("token_limit")
        assert result.action == "stop"
        assert result.reason == "token_limit"


class TestLimitChecker:
    """LimitChecker 测试"""

    def test_continue_within_limits(self):
        """在限制范围内应该返回 continue"""
        checker = LimitChecker(LoopLimits(max_steps=10))
        checker.current_step = 5
        result = checker.check()
        assert result.action == "continue"

    def test_step_soft_limit_triggers(self):
        """达到步数限制应触发软限制"""
        checker = LimitChecker(LoopLimits(max_steps=3))
        checker.current_step = 3
        result = checker.check()
        assert result.action == "warn"
        assert result.reason == "step_limit"

    def test_step_soft_limit_resets(self):
        """软限制触发后应重置检查点，下一个周期再触发"""
        checker = LimitChecker(LoopLimits(max_steps=3))

        # 第 3 步触发
        checker.current_step = 3
        result = checker.check()
        assert result.action == "warn"

        # 第 4、5 步不触发
        checker.current_step = 4
        assert checker.check().action == "continue"
        checker.current_step = 5
        assert checker.check().action == "continue"

        # 第 6 步再次触发
        checker.current_step = 6
        result = checker.check()
        assert result.action == "warn"
        assert result.reason == "step_limit"

    def test_token_hard_limit(self):
        """token 超限应触发硬限制"""
        checker = LimitChecker(LoopLimits(max_tokens=1000))
        checker.total_tokens = 1500
        result = checker.check()
        assert result.action == "stop"
        assert result.reason == "token_limit"

    def test_hard_limit_takes_priority(self):
        """硬限制应该优先于软限制"""
        checker = LimitChecker(LoopLimits(max_steps=3, max_tokens=1000))
        checker.current_step = 3
        checker.total_tokens = 1500
        result = checker.check()
        assert result.action == "stop"
        assert result.reason == "token_limit"

    def test_timeout_soft_limit(self):
        """超时应触发软限制"""
        checker = LimitChecker(LoopLimits(timeout=0.01))
        time.sleep(0.02)
        checker.current_step = 1
        result = checker.check()
        assert result.action == "warn"
        assert result.reason == "timeout"

    def test_add_tokens(self):
        """应该能累加 token"""
        checker = LimitChecker(LoopLimits())
        checker.add_tokens(100)
        assert checker.total_tokens == 100
        checker.add_tokens(200)
        assert checker.total_tokens == 300

    def test_step_limit_prompt_formatted(self):
        """软限制提示应包含格式化的步数"""
        checker = LimitChecker(LoopLimits(max_steps=5))
        checker.current_step = 5
        result = checker.check()
        assert "5" in result.prompt
```

**Step 2: 运行测试验证失败**

```bash
pytest tests/test_limits.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: 实现 limits.py**

`src/pure_agent_loop/limits.py`:
```python
"""终止控制

管理 Agentic Loop 的终止条件，区分软限制（通知 AI 调整）和硬限制（强制终止）。
"""

import time
from dataclasses import dataclass, field


@dataclass
class LoopLimits:
    """循环限制配置

    Attributes:
        max_steps: 软限制 - 每 N 步触发一次检查点（默认 10）
        timeout: 软限制 - 每 N 秒触发一次检查点（默认 300 秒）
        max_tokens: 硬限制 - 累计 token 上限（默认 100,000）
        step_limit_prompt: 步数软限制的提示模板
        timeout_prompt: 超时软限制的提示模板
    """

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
        "⚠️ 执行已超时（已用 {elapsed:.1f}s / 每轮上限 {timeout}s），"
        "请评估当前进度，尽快总结已获得的信息或调整策略。"
    )


@dataclass
class LimitResult:
    """限制检查结果

    Attributes:
        action: 行动指令 - "continue" | "warn" | "stop"
        reason: 触发原因
        prompt: 注入给 AI 的提示（仅 warn 时有值）
    """

    action: str
    reason: str = ""
    prompt: str = ""

    @classmethod
    def continue_running(cls) -> "LimitResult":
        """创建继续执行的结果"""
        return cls(action="continue")

    @classmethod
    def warn(cls, reason: str, prompt: str) -> "LimitResult":
        """创建软限制警告结果"""
        return cls(action="warn", reason=reason, prompt=prompt)

    @classmethod
    def stop(cls, reason: str) -> "LimitResult":
        """创建硬限制停止结果"""
        return cls(action="stop", reason=reason)


class LimitChecker:
    """限制检查器

    每步结束后调用 check() 判断是否触发限制。
    软限制使用周期性检查点机制，每 N 步/秒触发一次。

    Args:
        limits: 限制配置
    """

    def __init__(self, limits: LoopLimits):
        self.limits = limits
        self.start_time = time.time()
        self.current_step: int = 0
        self.total_tokens: int = 0
        self._step_checkpoint: int = 0
        self._timeout_checkpoint: float = 0.0

    def add_tokens(self, tokens: int) -> None:
        """累加 token 用量"""
        self.total_tokens += tokens

    @property
    def elapsed(self) -> float:
        """已用时间（秒）"""
        return time.time() - self.start_time

    def check(self) -> LimitResult:
        """检查终止条件

        Returns:
            LimitResult: 检查结果，包含行动指令和提示
        """
        # 硬限制优先 - token 上限
        if self.total_tokens >= self.limits.max_tokens:
            return LimitResult.stop("token_limit")

        # 软限制：步数周期检查
        steps_since_checkpoint = self.current_step - self._step_checkpoint
        if steps_since_checkpoint >= self.limits.max_steps:
            self._step_checkpoint = self.current_step
            prompt = self.limits.step_limit_prompt.format(
                checkpoint_steps=self.limits.max_steps,
                current_steps=self.current_step,
            )
            return LimitResult.warn("step_limit", prompt)

        # 软限制：超时周期检查
        elapsed = self.elapsed
        if elapsed - self._timeout_checkpoint >= self.limits.timeout:
            self._timeout_checkpoint = elapsed
            prompt = self.limits.timeout_prompt.format(
                elapsed=elapsed,
                timeout=self.limits.timeout,
            )
            return LimitResult.warn("timeout", prompt)

        return LimitResult.continue_running()
```

**Step 4: 运行测试验证通过**

```bash
pytest tests/test_limits.py -v
```

Expected: 9 passed

**Step 5: Commit**

```bash
git add src/pure_agent_loop/limits.py tests/test_limits.py
git commit -m "feat: 添加终止控制 (LoopLimits, LimitChecker)"
```

---

## Task 6: 重试机制 (retry.py)

**Files:**
- Create: `src/pure_agent_loop/retry.py`
- Create: `tests/test_retry.py`

**Step 1: 编写失败的测试**

`tests/test_retry.py`:
```python
"""重试机制测试"""

import pytest
from pure_agent_loop.retry import RetryConfig, RetryHandler


class TestRetryConfig:
    """RetryConfig 测试"""

    def test_default_values(self):
        """应该有合理的默认值"""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 30.0

    def test_custom_values(self):
        """应该支持自定义值"""
        config = RetryConfig(max_retries=5, base_delay=0.5, max_delay=10.0)
        assert config.max_retries == 5


class TestRetryHandler:
    """RetryHandler 测试"""

    @pytest.mark.asyncio
    async def test_success_no_retry(self):
        """成功调用不应重试"""
        call_count = 0

        async def success_fn():
            nonlocal call_count
            call_count += 1
            return "ok"

        handler = RetryHandler(RetryConfig())
        result = await handler.execute(success_fn)
        assert result == "ok"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """失败后应该重试"""
        call_count = 0

        async def flaky_fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("连接超时")
            return "ok"

        handler = RetryHandler(RetryConfig(base_delay=0.01))
        result = await handler.execute(flaky_fn)
        assert result == "ok"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """超过最大重试次数应该抛出异常"""

        async def always_fail():
            raise ConnectionError("连接超时")

        handler = RetryHandler(RetryConfig(max_retries=2, base_delay=0.01))
        with pytest.raises(ConnectionError):
            await handler.execute(always_fail)

    @pytest.mark.asyncio
    async def test_non_retryable_error(self):
        """不可重试的错误应该立即抛出"""
        call_count = 0

        async def bad_fn():
            nonlocal call_count
            call_count += 1
            raise ValueError("参数错误")

        handler = RetryHandler(RetryConfig(base_delay=0.01))
        with pytest.raises(ValueError):
            await handler.execute(bad_fn)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_on_retry_callback(self):
        """应该在每次重试时调用回调"""
        retry_events = []
        call_count = 0

        async def flaky_fn():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("连接超时")
            return "ok"

        def on_retry(attempt, error, delay):
            retry_events.append((attempt, str(error), delay))

        handler = RetryHandler(RetryConfig(base_delay=0.01))
        result = await handler.execute(flaky_fn, on_retry=on_retry)
        assert result == "ok"
        assert len(retry_events) == 1
        assert retry_events[0][0] == 1

    def test_calculate_delay(self):
        """延迟应该使用指数退避"""
        handler = RetryHandler(RetryConfig(base_delay=1.0, max_delay=30.0))
        assert handler._calculate_delay(0) == 1.0
        assert handler._calculate_delay(1) == 2.0
        assert handler._calculate_delay(2) == 4.0
        assert handler._calculate_delay(10) == 30.0  # 不超过 max_delay
```

**Step 2: 运行测试验证失败**

```bash
pytest tests/test_retry.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: 实现 retry.py**

`src/pure_agent_loop/retry.py`:
```python
"""重试机制

为 LLM API 调用提供自动重试与指数退避能力。
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable


# 默认可重试的异常类型
DEFAULT_RETRYABLE_ERRORS = (
    ConnectionError,
    TimeoutError,
    OSError,
)


@dataclass
class RetryConfig:
    """重试配置

    Attributes:
        max_retries: 最大重试次数（默认 3）
        base_delay: 基础延迟秒数（默认 1.0）
        max_delay: 最大延迟秒数（默认 30.0）
        retryable_errors: 可重试的异常类型元组
    """

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    retryable_errors: tuple[type[Exception], ...] = DEFAULT_RETRYABLE_ERRORS


class RetryHandler:
    """重试处理器

    使用指数退避策略自动重试失败的异步调用。

    Args:
        config: 重试配置
    """

    def __init__(self, config: RetryConfig | None = None):
        self.config = config or RetryConfig()

    def _calculate_delay(self, attempt: int) -> float:
        """计算退避延迟时间

        Args:
            attempt: 当前重试次数（0-based）

        Returns:
            延迟秒数
        """
        delay = self.config.base_delay * (2 ** attempt)
        return min(delay, self.config.max_delay)

    def _is_retryable(self, error: Exception) -> bool:
        """判断异常是否可重试"""
        return isinstance(error, self.config.retryable_errors)

    async def execute(
        self,
        fn: Callable[..., Awaitable[Any]],
        *args: Any,
        on_retry: Callable[[int, Exception, float], None] | None = None,
        **kwargs: Any,
    ) -> Any:
        """执行异步函数，失败时自动重试

        Args:
            fn: 要执行的异步函数
            on_retry: 重试时的回调函数 (attempt, error, delay)
            *args: 传递给 fn 的位置参数
            **kwargs: 传递给 fn 的关键字参数

        Returns:
            fn 的返回值

        Raises:
            Exception: 重试耗尽或不可重试的异常
        """
        last_error: Exception | None = None

        for attempt in range(self.config.max_retries + 1):
            try:
                return await fn(*args, **kwargs)
            except Exception as e:
                last_error = e

                # 不可重试的错误立即抛出
                if not self._is_retryable(e):
                    raise

                # 已耗尽重试次数
                if attempt >= self.config.max_retries:
                    raise

                # 计算延迟并等待
                delay = self._calculate_delay(attempt)
                if on_retry:
                    on_retry(attempt + 1, e, delay)
                await asyncio.sleep(delay)

        raise last_error  # type: ignore[misc]
```

**Step 4: 运行测试验证通过**

```bash
pytest tests/test_retry.py -v
```

Expected: 7 passed

**Step 5: Commit**

```bash
git add src/pure_agent_loop/retry.py tests/test_retry.py
git commit -m "feat: 添加重试机制 (RetryConfig, RetryHandler)"
```

---

## Task 7: 工具系统 (tool.py)

**Files:**
- Create: `src/pure_agent_loop/tool.py`
- Create: `tests/test_tool.py`

**Step 1: 编写失败的测试**

`tests/test_tool.py`:
```python
"""工具系统测试"""

import pytest
from pure_agent_loop.tool import tool, Tool, ToolRegistry


class TestToolDecorator:
    """@tool 装饰器测试"""

    def test_decorate_sync_function(self):
        """应该能装饰同步函数"""

        @tool
        def search(query: str) -> str:
            """搜索网页内容"""
            return f"结果: {query}"

        assert isinstance(search, Tool)
        assert search.name == "search"
        assert search.description == "搜索网页内容"

    def test_decorate_async_function(self):
        """应该能装饰异步函数"""

        @tool
        async def search(query: str) -> str:
            """搜索网页内容"""
            return f"结果: {query}"

        assert isinstance(search, Tool)
        assert search.name == "search"

    def test_extract_parameters_schema(self):
        """应该从类型注解提取参数 schema"""

        @tool
        def search(query: str, max_results: int = 5) -> str:
            """搜索网页内容

            Args:
                query: 搜索关键词
                max_results: 最大返回结果数
            """
            return ""

        schema = search.to_openai_schema()
        params = schema["function"]["parameters"]
        assert params["type"] == "object"
        assert "query" in params["properties"]
        assert params["properties"]["query"]["type"] == "string"
        assert params["properties"]["max_results"]["type"] == "integer"
        assert "query" in params["required"]
        assert "max_results" not in params["required"]

    def test_extract_description_from_docstring(self):
        """应该从 docstring 提取描述"""

        @tool
        def search(query: str) -> str:
            """搜索网页内容

            Args:
                query: 搜索关键词
            """
            return ""

        schema = search.to_openai_schema()
        assert schema["function"]["description"] == "搜索网页内容"
        assert schema["function"]["parameters"]["properties"]["query"].get("description") == "搜索关键词"

    def test_optional_parameter(self):
        """应该处理 Optional 参数"""

        @tool
        def search(query: str, lang: str | None = None) -> str:
            """搜索"""
            return ""

        schema = search.to_openai_schema()
        assert "lang" in schema["function"]["parameters"]["properties"]
        assert "lang" not in schema["function"]["parameters"]["required"]

    def test_bool_parameter(self):
        """应该处理布尔参数"""

        @tool
        def search(query: str, verbose: bool = False) -> str:
            """搜索"""
            return ""

        schema = search.to_openai_schema()
        assert schema["function"]["parameters"]["properties"]["verbose"]["type"] == "boolean"

    def test_float_parameter(self):
        """应该处理浮点参数"""

        @tool
        def calc(value: float) -> str:
            """计算"""
            return ""

        schema = calc.to_openai_schema()
        assert schema["function"]["parameters"]["properties"]["value"]["type"] == "number"


class TestTool:
    """Tool 对象测试"""

    @pytest.mark.asyncio
    async def test_execute_sync_tool(self):
        """应该能执行同步工具"""

        @tool
        def add(a: int, b: int) -> str:
            """加法"""
            return str(a + b)

        result = await add.execute({"a": 1, "b": 2})
        assert result == "3"

    @pytest.mark.asyncio
    async def test_execute_async_tool(self):
        """应该能执行异步工具"""

        @tool
        async def add(a: int, b: int) -> str:
            """加法"""
            return str(a + b)

        result = await add.execute({"a": 1, "b": 2})
        assert result == "3"

    @pytest.mark.asyncio
    async def test_execute_error_returns_message(self):
        """工具执行失败应返回错误信息字符串"""

        @tool
        def bad_tool(x: str) -> str:
            """坏工具"""
            raise ValueError("参数无效")

        result = await bad_tool.execute({"x": "test"})
        assert "执行失败" in result
        assert "ValueError" in result
        assert "参数无效" in result


class TestToolRegistry:
    """ToolRegistry 测试"""

    def test_register_tool_decorator(self):
        """应该能注册 @tool 装饰的函数"""

        @tool
        def search(query: str) -> str:
            """搜索"""
            return ""

        registry = ToolRegistry()
        registry.register(search)
        assert registry.get("search") is search

    def test_register_dict_format(self):
        """应该能注册字典格式的工具"""

        def search_fn(query: str) -> str:
            return ""

        tool_dict = {
            "name": "search",
            "description": "搜索网页",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "关键词"}
                },
                "required": ["query"],
            },
            "function": search_fn,
        }

        registry = ToolRegistry()
        registry.register(tool_dict)
        assert registry.get("search") is not None
        assert registry.get("search").name == "search"

    def test_register_many(self):
        """应该能批量注册"""

        @tool
        def tool_a(x: str) -> str:
            """工具A"""
            return ""

        @tool
        def tool_b(x: str) -> str:
            """工具B"""
            return ""

        registry = ToolRegistry()
        registry.register_many([tool_a, tool_b])
        assert registry.get("tool_a") is not None
        assert registry.get("tool_b") is not None

    def test_to_openai_schemas(self):
        """应该能转换为 OpenAI tools 格式"""

        @tool
        def search(query: str) -> str:
            """搜索"""
            return ""

        registry = ToolRegistry()
        registry.register(search)
        schemas = registry.to_openai_schemas()
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "search"

    @pytest.mark.asyncio
    async def test_execute_tool(self):
        """应该能按名称执行工具"""

        @tool
        def add(a: int, b: int) -> str:
            """加法"""
            return str(a + b)

        registry = ToolRegistry()
        registry.register(add)
        result = await registry.execute("add", {"a": 3, "b": 4})
        assert result == "7"

    def test_get_unknown_tool_returns_none(self):
        """获取不存在的工具应返回 None"""
        registry = ToolRegistry()
        assert registry.get("unknown") is None
```

**Step 2: 运行测试验证失败**

```bash
pytest tests/test_tool.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: 实现 tool.py**

`src/pure_agent_loop/tool.py`:
```python
"""工具系统

支持 @tool 装饰器和字典格式两种工具定义方式。
"""

import asyncio
import inspect
import re
from dataclasses import dataclass, field
from typing import Any, Callable, get_type_hints


# Python 类型到 JSON Schema 类型的映射
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _parse_google_docstring(docstring: str) -> tuple[str, dict[str, str]]:
    """解析 Google 风格的 docstring

    Returns:
        (描述, {参数名: 参数描述})
    """
    if not docstring:
        return "", {}

    lines = docstring.strip().split("\n")
    description_lines = []
    param_descriptions: dict[str, str] = {}
    in_args_section = False

    for line in lines:
        stripped = line.strip()

        # 检测 Args: 段落
        if stripped.lower().startswith("args:"):
            in_args_section = True
            continue

        if in_args_section:
            # 新段落开始（如 Returns:, Raises:）则结束 Args 解析
            if stripped and not stripped.startswith("-") and ":" in stripped:
                # 检查是否是参数行（格式: param_name: description）
                match = re.match(r"(\w+)\s*(?:\(.*?\))?\s*:\s*(.+)", stripped)
                if match:
                    param_descriptions[match.group(1)] = match.group(2).strip()
                    continue
            if stripped.lower() in ("returns:", "raises:", "yields:", "examples:", "note:", "notes:"):
                in_args_section = False
                continue
            # 可能是参数描述的续行，忽略
            continue

        if stripped:
            description_lines.append(stripped)

    description = " ".join(description_lines) if description_lines else ""
    return description, param_descriptions


def _get_json_type(python_type: type) -> str:
    """将 Python 类型转换为 JSON Schema 类型"""
    # 处理 Optional (X | None)
    origin = getattr(python_type, "__origin__", None)
    if origin is type(int | str):  # types.UnionType (Python 3.10+)
        args = python_type.__args__
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            return _get_json_type(non_none[0])

    return _TYPE_MAP.get(python_type, "string")


class Tool:
    """工具对象

    封装工具的元信息和执行逻辑。

    Attributes:
        name: 工具名称
        description: 工具描述
        parameters: JSON Schema 格式的参数描述
        function: 实际执行的函数
        is_async: 是否为异步函数
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        function: Callable,
        is_async: bool = False,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.function = function
        self.is_async = is_async

    def to_openai_schema(self) -> dict[str, Any]:
        """转换为 OpenAI Function Calling 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> str:
        """执行工具函数

        出错时不抛出异常，而是返回格式化的错误信息。

        Args:
            arguments: 工具参数字典

        Returns:
            工具执行结果（字符串）
        """
        try:
            if self.is_async:
                result = await self.function(**arguments)
            else:
                result = self.function(**arguments)
            return str(result)
        except Exception as e:
            return (
                f"⚠️ 工具 '{self.name}' 执行失败:\n"
                f"错误类型: {type(e).__name__}\n"
                f"错误信息: {str(e)}\n"
                f"请尝试调整参数或使用其他方法。"
            )


def tool(fn: Callable) -> Tool:
    """工具装饰器

    将普通函数转换为 Tool 对象，自动提取参数 schema 和描述。

    Args:
        fn: 要装饰的函数（同步或异步）

    Returns:
        Tool 对象
    """
    name = fn.__name__
    is_async = asyncio.iscoroutinefunction(fn)

    # 解析 docstring
    description, param_descriptions = _parse_google_docstring(fn.__doc__ or "")

    # 获取类型注解
    hints = get_type_hints(fn)
    sig = inspect.signature(fn)

    # 构建 JSON Schema
    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name == "return":
            continue

        param_type = hints.get(param_name, str)
        json_type = _get_json_type(param_type)

        prop: dict[str, Any] = {"type": json_type}

        # 添加参数描述
        if param_name in param_descriptions:
            prop["description"] = param_descriptions[param_name]

        properties[param_name] = prop

        # 无默认值 => 必填参数
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    parameters = {
        "type": "object",
        "properties": properties,
        "required": required,
    }

    return Tool(
        name=name,
        description=description,
        parameters=parameters,
        function=fn,
        is_async=is_async,
    )


class ToolRegistry:
    """工具注册表

    管理所有注册的工具，提供查询和执行能力。
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool_or_dict: Tool | dict[str, Any]) -> None:
        """注册单个工具

        支持 Tool 对象或字典格式。

        Args:
            tool_or_dict: Tool 对象或字典格式工具定义
        """
        if isinstance(tool_or_dict, Tool):
            self._tools[tool_or_dict.name] = tool_or_dict
        elif isinstance(tool_or_dict, dict):
            # 从字典格式创建 Tool
            t = Tool(
                name=tool_or_dict["name"],
                description=tool_or_dict.get("description", ""),
                parameters=tool_or_dict.get("parameters", {"type": "object", "properties": {}}),
                function=tool_or_dict["function"],
                is_async=asyncio.iscoroutinefunction(tool_or_dict["function"]),
            )
            self._tools[t.name] = t
        else:
            raise TypeError(f"不支持的工具类型: {type(tool_or_dict)}")

    def register_many(self, tools: list[Tool | dict[str, Any]]) -> None:
        """批量注册工具"""
        for t in tools:
            self.register(t)

    def get(self, name: str) -> Tool | None:
        """按名称获取工具"""
        return self._tools.get(name)

    def to_openai_schemas(self) -> list[dict[str, Any]]:
        """转换为 OpenAI tools 格式列表"""
        return [t.to_openai_schema() for t in self._tools.values()]

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """按名称执行工具

        Args:
            name: 工具名称
            arguments: 工具参数

        Returns:
            执行结果字符串
        """
        t = self.get(name)
        if t is None:
            return f"⚠️ 未知工具 '{name}'，请检查工具名称。"
        return await t.execute(arguments)
```

**Step 4: 运行测试验证通过**

```bash
pytest tests/test_tool.py -v
```

Expected: 13 passed

**Step 5: Commit**

```bash
git add src/pure_agent_loop/tool.py tests/test_tool.py
git commit -m "feat: 添加工具系统 (@tool 装饰器, ToolRegistry)"
```

---

## Task 8: LLM 抽象层 (llm/base.py)

**Files:**
- Create: `src/pure_agent_loop/llm/base.py`
- Create: `tests/test_llm_base.py`

**Step 1: 编写失败的测试**

`tests/test_llm_base.py`:
```python
"""LLM 抽象层测试"""

import pytest
from pure_agent_loop.llm.base import BaseLLMClient
from pure_agent_loop.llm.types import LLMResponse, TokenUsage


class MockLLMClient(BaseLLMClient):
    """测试用 Mock 客户端"""

    async def chat(self, messages, tools=None, **kwargs):
        return LLMResponse(
            content="Mock 回复",
            tool_calls=[],
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            raw={},
        )


class TestBaseLLMClient:
    """BaseLLMClient 抽象接口测试"""

    def test_cannot_instantiate_directly(self):
        """不能直接实例化抽象类"""
        with pytest.raises(TypeError):
            BaseLLMClient()

    @pytest.mark.asyncio
    async def test_mock_client_implements_interface(self):
        """实现子类应能正常工作"""
        client = MockLLMClient()
        response = await client.chat([{"role": "user", "content": "hello"}])
        assert response.content == "Mock 回复"
        assert response.usage.total_tokens == 15
```

**Step 2: 运行测试验证失败**

```bash
pytest tests/test_llm_base.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: 实现 llm/base.py**

`src/pure_agent_loop/llm/base.py`:
```python
"""LLM 客户端抽象接口

定义所有 LLM 客户端必须实现的接口。
"""

from abc import ABC, abstractmethod
from typing import Any

from .types import LLMResponse


class BaseLLMClient(ABC):
    """LLM 客户端抽象基类

    所有 LLM 客户端（内置和自定义）必须继承此类并实现 chat 方法。
    """

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """发送对话请求

        Args:
            messages: 消息列表（OpenAI Chat 格式）
            tools: 工具定义列表（OpenAI Function Calling 格式）
            **kwargs: 额外参数（如 temperature 等）

        Returns:
            统一响应模型
        """
        ...
```

**Step 4: 运行测试验证通过**

```bash
pytest tests/test_llm_base.py -v
```

Expected: 2 passed

**Step 5: Commit**

```bash
git add src/pure_agent_loop/llm/base.py tests/test_llm_base.py
git commit -m "feat: 添加 LLM 抽象接口 (BaseLLMClient)"
```

---

## Task 9: OpenAI 兼容客户端 (llm/openai_client.py)

**Files:**
- Create: `src/pure_agent_loop/llm/openai_client.py`
- Modify: `src/pure_agent_loop/llm/__init__.py`
- Create: `tests/test_openai_client.py`

**Step 1: 编写失败的测试**

`tests/test_openai_client.py`:
```python
"""OpenAI 兼容客户端测试"""

import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from pure_agent_loop.llm.openai_client import OpenAIClient
from pure_agent_loop.llm.types import LLMResponse


def _make_mock_response(content=None, tool_calls=None):
    """构造 Mock 的 OpenAI 响应对象"""
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls or []

    choice = MagicMock()
    choice.message = message

    usage = MagicMock()
    usage.prompt_tokens = 100
    usage.completion_tokens = 50
    usage.total_tokens = 150

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


def _make_mock_tool_call(call_id, name, arguments):
    """构造 Mock 的 tool_call 对象"""
    function = MagicMock()
    function.name = name
    function.arguments = json.dumps(arguments)

    tc = MagicMock()
    tc.id = call_id
    tc.function = function
    return tc


class TestOpenAIClient:
    """OpenAI 客户端测试"""

    @pytest.mark.asyncio
    async def test_chat_with_text_response(self):
        """应该正确解析文本响应"""
        mock_response = _make_mock_response(content="你好")

        client = OpenAIClient(model="gpt-4o-mini", api_key="test-key")
        client._client = AsyncMock()
        client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        response = await client.chat([{"role": "user", "content": "hi"}])
        assert isinstance(response, LLMResponse)
        assert response.content == "你好"
        assert response.has_tool_calls is False
        assert response.usage.total_tokens == 150

    @pytest.mark.asyncio
    async def test_chat_with_tool_calls(self):
        """应该正确解析工具调用响应"""
        mock_tc = _make_mock_tool_call("call_1", "search", {"query": "python"})
        mock_response = _make_mock_response(tool_calls=[mock_tc])

        client = OpenAIClient(model="gpt-4o-mini", api_key="test-key")
        client._client = AsyncMock()
        client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        response = await client.chat(
            [{"role": "user", "content": "搜索"}],
            tools=[{"type": "function", "function": {"name": "search"}}],
        )
        assert response.has_tool_calls is True
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "search"
        assert response.tool_calls[0].arguments == {"query": "python"}

    def test_constructor_with_base_url(self):
        """应该支持自定义 base_url"""
        client = OpenAIClient(
            model="deepseek-chat",
            api_key="sk-xxx",
            base_url="https://api.deepseek.com/v1",
        )
        assert client.model == "deepseek-chat"

    @pytest.mark.asyncio
    async def test_extra_kwargs_passed_through(self):
        """额外参数应该透传给 API 调用"""
        mock_response = _make_mock_response(content="ok")

        client = OpenAIClient(model="gpt-4o-mini", api_key="test-key")
        client._client = AsyncMock()
        client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        await client.chat(
            [{"role": "user", "content": "hi"}],
            temperature=0.5,
            max_tokens=100,
        )

        call_kwargs = client._client.chat.completions.create.call_args
        assert call_kwargs.kwargs.get("temperature") == 0.5
        assert call_kwargs.kwargs.get("max_tokens") == 100
```

**Step 2: 运行测试验证失败**

```bash
pytest tests/test_openai_client.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: 实现 llm/openai_client.py**

`src/pure_agent_loop/llm/openai_client.py`:
```python
"""OpenAI 兼容客户端

基于 openai SDK 实现的 LLM 客户端，兼容所有 OpenAI 兼容 API。
"""

import json
import logging
from typing import Any

from openai import AsyncOpenAI

from .base import BaseLLMClient
from .types import LLMResponse, ToolCall, TokenUsage

logger = logging.getLogger(__name__)


class OpenAIClient(BaseLLMClient):
    """OpenAI 兼容 LLM 客户端

    支持 OpenAI、Azure OpenAI、DeepSeek、通义千问等所有兼容 API。

    Args:
        model: 模型名称
        api_key: API 密钥（默认读取 OPENAI_API_KEY 环境变量）
        base_url: API 基础地址（用于接入其他兼容服务）
        **kwargs: 传递给 AsyncOpenAI 的额外参数
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ):
        self.model = model
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

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

        # 解析 token 用量
        usage = TokenUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )

        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            usage=usage,
            raw=response,
        )
```

**Step 4: 更新 llm/__init__.py**

`src/pure_agent_loop/llm/__init__.py`:
```python
"""LLM 抽象层模块"""

from .base import BaseLLMClient
from .openai_client import OpenAIClient
from .types import LLMResponse, ToolCall, TokenUsage

__all__ = [
    "BaseLLMClient",
    "OpenAIClient",
    "LLMResponse",
    "ToolCall",
    "TokenUsage",
]
```

**Step 5: 运行测试验证通过**

```bash
pytest tests/test_openai_client.py -v
```

Expected: 4 passed

**Step 6: Commit**

```bash
git add src/pure_agent_loop/llm/ tests/test_openai_client.py
git commit -m "feat: 添加 OpenAI 兼容客户端 (OpenAIClient)"
```

---

## Task 10: 渲染器 (renderer.py)

**Files:**
- Create: `src/pure_agent_loop/renderer.py`
- Create: `tests/test_renderer.py`

**Step 1: 编写失败的测试**

`tests/test_renderer.py`:
```python
"""渲染器测试"""

import pytest
from pure_agent_loop.renderer import Renderer
from pure_agent_loop.events import Event, EventType


class TestRenderer:
    """Renderer 测试"""

    def test_default_render_action(self):
        """默认应渲染 action 事件"""
        renderer = Renderer()
        event = Event.action(step=1, tool="search", args={"query": "python"})
        result = renderer.render(event)
        assert "search" in result
        assert result is not None

    def test_default_render_observation(self):
        """默认应渲染 observation 事件"""
        renderer = Renderer()
        event = Event.observation(step=1, tool="search", result="找到结果", duration=1.2)
        result = renderer.render(event)
        assert result is not None

    def test_default_render_thought(self):
        """默认应渲染 thought 事件"""
        renderer = Renderer()
        event = Event.thought(step=1, content="让我思考一下")
        result = renderer.render(event)
        assert "让我思考一下" in result

    def test_default_render_error(self):
        """默认应渲染 error 事件"""
        renderer = Renderer()
        event = Event.error(step=1, error="出错了")
        result = renderer.render(event)
        assert "出错了" in result

    def test_custom_tool_renderer(self):
        """应该支持自定义工具渲染器"""
        renderer = Renderer()

        @renderer.on_tool("search")
        def render_search(event: Event) -> str:
            return f"🔍 搜索: {event.data['args']['query']}"

        event = Event.action(step=1, tool="search", args={"query": "python"})
        result = renderer.render(event)
        assert result == "🔍 搜索: python"

    def test_custom_event_renderer(self):
        """应该支持自定义事件类型渲染器"""
        renderer = Renderer()

        @renderer.on_event(EventType.SOFT_LIMIT)
        def render_limit(event: Event) -> str:
            return f"⚠️ 限制: {event.data['reason']}"

        event = Event.soft_limit(step=10, reason="step_limit", prompt="请调整")
        result = renderer.render(event)
        assert result == "⚠️ 限制: step_limit"

    def test_tool_renderer_priority(self):
        """工具专用渲染器应优先于事件类型渲染器"""
        renderer = Renderer()

        @renderer.on_tool("search")
        def render_search(event: Event) -> str:
            return "工具专用"

        @renderer.on_event(EventType.ACTION)
        def render_action(event: Event) -> str:
            return "类型通用"

        event = Event.action(step=1, tool="search", args={})
        result = renderer.render(event)
        assert result == "工具专用"

    def test_event_renderer_fallback(self):
        """无工具专用渲染器时应回退到事件类型渲染器"""
        renderer = Renderer()

        @renderer.on_event(EventType.ACTION)
        def render_action(event: Event) -> str:
            return "类型通用"

        event = Event.action(step=1, tool="unknown_tool", args={})
        result = renderer.render(event)
        assert result == "类型通用"

    def test_render_returns_none_for_unhandled(self):
        """无渲染器的事件类型返回默认渲染"""
        renderer = Renderer()
        event = Event.loop_start(task="测试")
        result = renderer.render(event)
        # 默认渲染器应该返回某些内容
        assert isinstance(result, str)
```

**Step 2: 运行测试验证失败**

```bash
pytest tests/test_renderer.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: 实现 renderer.py**

`src/pure_agent_loop/renderer.py`:
```python
"""事件渲染器

将结构化事件转换为用户友好的展示格式。
支持内置默认规则和自定义渲染装饰器。
"""

from typing import Any, Callable

from .events import Event, EventType


class Renderer:
    """事件渲染器

    渲染优先级:
    1. 工具专用渲染器 (@renderer.on_tool("search"))
    2. 事件类型渲染器 (@renderer.on_event(EventType.ACTION))
    3. 内置默认渲染器
    """

    def __init__(self):
        self._tool_renderers: dict[str, Callable[[Event], str]] = {}
        self._event_renderers: dict[EventType, Callable[[Event], str]] = {}

    def on_tool(self, tool_name: str) -> Callable:
        """注册工具专用渲染器

        Args:
            tool_name: 工具名称

        Returns:
            装饰器函数
        """

        def decorator(fn: Callable[[Event], str]) -> Callable[[Event], str]:
            self._tool_renderers[tool_name] = fn
            return fn

        return decorator

    def on_event(self, event_type: EventType) -> Callable:
        """注册事件类型渲染器

        Args:
            event_type: 事件类型

        Returns:
            装饰器函数
        """

        def decorator(fn: Callable[[Event], str]) -> Callable[[Event], str]:
            self._event_renderers[event_type] = fn
            return fn

        return decorator

    def render(self, event: Event) -> str:
        """渲染事件

        按优先级匹配渲染器: 工具专用 > 事件类型 > 默认。

        Args:
            event: 要渲染的事件

        Returns:
            渲染后的字符串
        """
        # 1. 工具专用渲染器（仅 ACTION 和 OBSERVATION 事件）
        tool_name = event.data.get("tool")
        if tool_name and tool_name in self._tool_renderers:
            return self._tool_renderers[tool_name](event)

        # 2. 事件类型渲染器
        if event.type in self._event_renderers:
            return self._event_renderers[event.type](event)

        # 3. 内置默认渲染器
        return self._default_render(event)

    def _default_render(self, event: Event) -> str:
        """内置默认渲染器"""
        match event.type:
            case EventType.LOOP_START:
                return f"🚀 开始任务: {event.data.get('task', '')}"
            case EventType.THOUGHT:
                return f"💭 思考: {event.data.get('content', '')}"
            case EventType.ACTION:
                tool = event.data.get("tool", "")
                args = event.data.get("args", {})
                args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
                return f"🔧 调用工具: {tool}({args_str})"
            case EventType.OBSERVATION:
                tool = event.data.get("tool", "")
                result = event.data.get("result", "")
                duration = event.data.get("duration", 0)
                # 截断过长的结果
                preview = result[:200] + "..." if len(str(result)) > 200 else result
                return f"📋 [{tool}] 结果 ({duration:.1f}s): {preview}"
            case EventType.SOFT_LIMIT:
                return f"⚠️ 软限制触发: {event.data.get('reason', '')}"
            case EventType.ERROR:
                prefix = "❌ 致命错误" if event.data.get("fatal") else "⚠️ 错误"
                return f"{prefix}: {event.data.get('error', '')}"
            case EventType.LOOP_END:
                reason = event.data.get("stop_reason", "")
                return f"✅ 任务结束 (原因: {reason})"
            case _:
                return f"[{event.type.value}] {event.data}"
```

**Step 4: 运行测试验证通过**

```bash
pytest tests/test_renderer.py -v
```

Expected: 9 passed

**Step 5: Commit**

```bash
git add src/pure_agent_loop/renderer.py tests/test_renderer.py
git commit -m "feat: 添加渲染器 (Renderer)"
```

---

## Task 11: ReAct 循环引擎 (loop.py)

**Files:**
- Create: `src/pure_agent_loop/loop.py`
- Create: `tests/test_loop.py`

**Step 1: 编写失败的测试**

`tests/test_loop.py`:
```python
"""ReAct 循环引擎测试"""

import json
import pytest
from unittest.mock import AsyncMock

from pure_agent_loop.loop import ReactLoop
from pure_agent_loop.tool import tool, ToolRegistry
from pure_agent_loop.llm.base import BaseLLMClient
from pure_agent_loop.llm.types import LLMResponse, ToolCall, TokenUsage
from pure_agent_loop.events import Event, EventType
from pure_agent_loop.limits import LoopLimits
from pure_agent_loop.retry import RetryConfig


class MockLLM(BaseLLMClient):
    """可控的 Mock LLM 客户端"""

    def __init__(self, responses: list[LLMResponse]):
        self._responses = responses
        self._call_count = 0

    async def chat(self, messages, tools=None, **kwargs):
        response = self._responses[self._call_count]
        self._call_count += 1
        return response


def _text_response(content: str) -> LLMResponse:
    """创建纯文本响应"""
    return LLMResponse(
        content=content,
        tool_calls=[],
        usage=TokenUsage(prompt_tokens=50, completion_tokens=20, total_tokens=70),
        raw={},
    )


def _tool_call_response(tool_name: str, arguments: dict) -> LLMResponse:
    """创建工具调用响应"""
    return LLMResponse(
        content=None,
        tool_calls=[ToolCall(id=f"call_{tool_name}", name=tool_name, arguments=arguments)],
        usage=TokenUsage(prompt_tokens=50, completion_tokens=30, total_tokens=80),
        raw={},
    )


class TestReactLoop:
    """ReactLoop 测试"""

    @pytest.mark.asyncio
    async def test_simple_text_response(self):
        """无工具调用时应直接返回文本"""
        llm = MockLLM([_text_response("你好，世界！")])

        loop = ReactLoop(
            llm=llm,
            tool_registry=ToolRegistry(),
            limits=LoopLimits(),
            retry=RetryConfig(),
        )

        events = []
        async for event in loop.run("打个招呼", system_prompt="你是助手"):
            events.append(event)

        # 应该有: LOOP_START, THOUGHT, LOOP_END
        types = [e.type for e in events]
        assert EventType.LOOP_START in types
        assert EventType.LOOP_END in types

        # 最终结果
        end_event = next(e for e in events if e.type == EventType.LOOP_END)
        assert end_event.data["stop_reason"] == "completed"
        assert end_event.data["content"] == "你好，世界！"

    @pytest.mark.asyncio
    async def test_tool_call_then_response(self):
        """应该执行工具调用然后返回最终结果"""

        @tool
        def add(a: int, b: int) -> str:
            """加法"""
            return str(a + b)

        registry = ToolRegistry()
        registry.register(add)

        llm = MockLLM([
            _tool_call_response("add", {"a": 3, "b": 4}),
            _text_response("3 + 4 = 7"),
        ])

        loop = ReactLoop(
            llm=llm,
            tool_registry=registry,
            limits=LoopLimits(),
            retry=RetryConfig(),
        )

        events = []
        async for event in loop.run("计算 3+4"):
            events.append(event)

        types = [e.type for e in events]
        assert EventType.ACTION in types
        assert EventType.OBSERVATION in types
        assert EventType.LOOP_END in types

        # 检查工具执行结果
        obs_event = next(e for e in events if e.type == EventType.OBSERVATION)
        assert obs_event.data["result"] == "7"

    @pytest.mark.asyncio
    async def test_step_soft_limit_triggers(self):
        """步数软限制应触发并注入提示"""

        @tool
        def noop(x: str) -> str:
            """空操作"""
            return "ok"

        registry = ToolRegistry()
        registry.register(noop)

        # 创建 4 次工具调用 + 最终文本响应
        responses = []
        for i in range(4):
            responses.append(_tool_call_response("noop", {"x": str(i)}))
        responses.append(_text_response("完成"))

        llm = MockLLM(responses)

        loop = ReactLoop(
            llm=llm,
            tool_registry=registry,
            limits=LoopLimits(max_steps=3),
            retry=RetryConfig(),
        )

        events = []
        async for event in loop.run("测试软限制"):
            events.append(event)

        types = [e.type for e in events]
        assert EventType.SOFT_LIMIT in types

    @pytest.mark.asyncio
    async def test_token_hard_limit_stops(self):
        """token 硬限制应强制终止循环"""

        @tool
        def noop(x: str) -> str:
            """空操作"""
            return "ok"

        registry = ToolRegistry()
        registry.register(noop)

        # 创建多次工具调用
        responses = [_tool_call_response("noop", {"x": str(i)}) for i in range(20)]
        llm = MockLLM(responses)

        loop = ReactLoop(
            llm=llm,
            tool_registry=registry,
            limits=LoopLimits(max_tokens=200),  # 非常低的限制
            retry=RetryConfig(),
        )

        events = []
        async for event in loop.run("测试硬限制"):
            events.append(event)

        end_event = next(e for e in events if e.type == EventType.LOOP_END)
        assert end_event.data["stop_reason"] == "token_limit"

    @pytest.mark.asyncio
    async def test_messages_history_continuity(self):
        """应该返回完整的消息历史用于多轮对话"""
        llm = MockLLM([_text_response("你好")])

        loop = ReactLoop(
            llm=llm,
            tool_registry=ToolRegistry(),
            limits=LoopLimits(),
            retry=RetryConfig(),
        )

        events = []
        async for event in loop.run("hi", system_prompt="你是助手"):
            events.append(event)

        end_event = next(e for e in events if e.type == EventType.LOOP_END)
        messages = end_event.data.get("messages", [])
        assert len(messages) >= 2  # 至少有 system + user + assistant
```

**Step 2: 运行测试验证失败**

```bash
pytest tests/test_loop.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: 实现 loop.py**

`src/pure_agent_loop/loop.py`:
```python
"""ReAct 循环引擎

实现 Thought → Action → Observation 的循环流程。
"""

import logging
import time
from typing import Any, AsyncIterator

from .events import Event, EventType
from .limits import LoopLimits, LimitChecker
from .llm.base import BaseLLMClient
from .llm.types import LLMResponse
from .retry import RetryConfig, RetryHandler
from .tool import ToolRegistry

logger = logging.getLogger(__name__)


class ReactLoop:
    """ReAct 循环引擎

    执行 Thought → Action → Observation 循环，直到满足终止条件。

    Args:
        llm: LLM 客户端实例
        tool_registry: 工具注册表
        limits: 终止条件配置
        retry: 重试配置
    """

    def __init__(
        self,
        llm: BaseLLMClient,
        tool_registry: ToolRegistry,
        limits: LoopLimits,
        retry: RetryConfig,
        llm_kwargs: dict[str, Any] | None = None,
    ):
        self._llm = llm
        self._tools = tool_registry
        self._limits = limits
        self._retry_handler = RetryHandler(retry)
        self._llm_kwargs = llm_kwargs or {}

    async def run(
        self,
        task: str,
        system_prompt: str = "You are a helpful assistant.",
        messages: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[Event]:
        """执行 ReAct 循环

        Args:
            task: 用户任务描述
            system_prompt: 系统提示
            messages: 初始消息历史（用于多轮对话）

        Yields:
            Event: 执行过程中的结构化事件
        """
        # 初始化消息历史
        if messages:
            msg_history = list(messages)
            msg_history.append({"role": "user", "content": task})
        else:
            msg_history = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task},
            ]

        # 初始化限制检查器
        checker = LimitChecker(self._limits)

        # 获取工具 schema
        tool_schemas = self._tools.to_openai_schemas() or None

        # 产出循环启动事件
        yield Event.loop_start(task=task)

        step = 0
        final_content = ""

        while True:
            step += 1
            checker.current_step = step

            # ---- 调用 LLM ----
            try:
                response = await self._retry_handler.execute(
                    self._llm.chat,
                    messages=msg_history,
                    tools=tool_schemas,
                    **self._llm_kwargs,
                )
            except Exception as e:
                logger.error("LLM 调用失败 (重试耗尽): %s", e)
                yield Event.error(step=step, error=str(e), fatal=True)
                yield Event.loop_end(
                    step=step,
                    stop_reason="error",
                    content=f"LLM 调用失败: {e}",
                )
                return

            # 累加 token
            checker.add_tokens(response.usage.total_tokens)

            # ---- 处理响应 ----
            if response.has_tool_calls:
                # Thought（如果有文本内容）
                if response.content:
                    yield Event.thought(step=step, content=response.content)

                # 构建 assistant 消息
                assistant_msg: dict[str, Any] = {"role": "assistant"}
                if response.content:
                    assistant_msg["content"] = response.content
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": __import__("json").dumps(tc.arguments),
                        },
                    }
                    for tc in response.tool_calls
                ]
                msg_history.append(assistant_msg)

                # ---- 执行工具 ----
                for tc in response.tool_calls:
                    yield Event.action(step=step, tool=tc.name, args=tc.arguments)

                    start_time = time.time()
                    result = await self._tools.execute(tc.name, tc.arguments)
                    duration = time.time() - start_time

                    yield Event.observation(
                        step=step, tool=tc.name, result=result, duration=duration
                    )

                    # 将工具结果追加到消息历史
                    msg_history.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        }
                    )

            else:
                # 无工具调用 = 最终回答
                final_content = response.content or ""
                if final_content:
                    yield Event.thought(step=step, content=final_content)

                msg_history.append(
                    {"role": "assistant", "content": final_content}
                )

                yield Event.loop_end(
                    step=step,
                    stop_reason="completed",
                    content=final_content,
                )
                # 在 loop_end 事件中附带消息历史
                # （通过修改最后一个事件的 data）
                # 注意: 上面已经 yield 了，这里需要特殊处理
                # 改为在 loop_end 中直接包含 messages
                return

            # ---- 检查限制 ----
            limit_result = checker.check()

            if limit_result.action == "stop":
                yield Event.loop_end(
                    step=step,
                    stop_reason=limit_result.reason,
                    content=final_content,
                )
                return

            if limit_result.action == "warn":
                yield Event.soft_limit(
                    step=step,
                    reason=limit_result.reason,
                    prompt=limit_result.prompt,
                )
                # 注入系统提示
                msg_history.append(
                    {"role": "system", "content": limit_result.prompt}
                )
```

注意: 上面的实现中 loop_end 事件需要携带 messages 历史。需要调整一下 loop_end 事件数据结构，将 messages 放入 data 中:

将 `loop.py` 中所有 `Event.loop_end(...)` 调用修改为包含 `messages` 参数。具体修改：

在 `events.py` 的 `loop_end` 工厂方法中添加 `messages` 参数（默认空列表）：

```python
@classmethod
def loop_end(cls, step: int, stop_reason: str, content: str = "", messages: list | None = None) -> "Event":
    """创建循环结束事件"""
    return cls(
        type=EventType.LOOP_END,
        step=step,
        data={"stop_reason": stop_reason, "content": content, "messages": messages or []},
    )
```

对应修改 `loop.py` 中的 3 处 `loop_end` 调用，传入 `messages=msg_history`：

```python
yield Event.loop_end(step=step, stop_reason="completed", content=final_content, messages=msg_history)
yield Event.loop_end(step=step, stop_reason=limit_result.reason, content=final_content, messages=msg_history)
yield Event.loop_end(step=step, stop_reason="error", content=f"LLM 调用失败: {e}", messages=msg_history)
```

**Step 4: 运行测试验证通过**

```bash
pytest tests/test_loop.py -v
```

Expected: 5 passed

**Step 5: Commit**

```bash
git add src/pure_agent_loop/loop.py src/pure_agent_loop/events.py tests/test_loop.py
git commit -m "feat: 添加 ReAct 循环引擎 (ReactLoop)"
```

---

## Task 12: Agent 入口 (agent.py)

**Files:**
- Create: `src/pure_agent_loop/agent.py`
- Create: `tests/test_agent.py`

**Step 1: 编写失败的测试**

`tests/test_agent.py`:
```python
"""Agent 入口测试"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pure_agent_loop.agent import Agent, AgentResult
from pure_agent_loop.tool import tool
from pure_agent_loop.events import Event, EventType
from pure_agent_loop.llm.base import BaseLLMClient
from pure_agent_loop.llm.types import LLMResponse, ToolCall, TokenUsage
from pure_agent_loop.limits import LoopLimits
from pure_agent_loop.retry import RetryConfig


class MockLLM(BaseLLMClient):
    """测试用 Mock LLM"""

    def __init__(self, responses):
        self._responses = responses
        self._call_count = 0

    async def chat(self, messages, tools=None, **kwargs):
        response = self._responses[self._call_count]
        self._call_count += 1
        return response


def _text_response(content):
    return LLMResponse(
        content=content,
        tool_calls=[],
        usage=TokenUsage(prompt_tokens=50, completion_tokens=20, total_tokens=70),
        raw={},
    )


def _tool_call_response(name, args):
    return LLMResponse(
        content=None,
        tool_calls=[ToolCall(id=f"call_{name}", name=name, arguments=args)],
        usage=TokenUsage(prompt_tokens=50, completion_tokens=30, total_tokens=80),
        raw={},
    )


class TestAgent:
    """Agent 入口测试"""

    @pytest.mark.asyncio
    async def test_arun_simple(self):
        """异步执行应返回 AgentResult"""
        mock_llm = MockLLM([_text_response("你好")])
        agent = Agent(llm=mock_llm)
        result = await agent.arun("打个招呼")

        assert isinstance(result, AgentResult)
        assert result.content == "你好"
        assert result.stop_reason == "completed"
        assert result.steps >= 1
        assert len(result.events) > 0
        assert len(result.messages) > 0

    def test_run_simple(self):
        """同步执行应返回 AgentResult"""
        mock_llm = MockLLM([_text_response("你好")])
        agent = Agent(llm=mock_llm)
        result = agent.run("打个招呼")

        assert isinstance(result, AgentResult)
        assert result.content == "你好"

    @pytest.mark.asyncio
    async def test_arun_with_tools(self):
        """带工具执行应正确处理"""

        @tool
        def add(a: int, b: int) -> str:
            """加法"""
            return str(a + b)

        mock_llm = MockLLM([
            _tool_call_response("add", {"a": 1, "b": 2}),
            _text_response("1 + 2 = 3"),
        ])

        agent = Agent(llm=mock_llm, tools=[add])
        result = await agent.arun("计算 1+2")

        assert result.content == "1 + 2 = 3"
        assert result.stop_reason == "completed"
        # 应该有工具调用事件
        action_events = [e for e in result.events if e.type == EventType.ACTION]
        assert len(action_events) == 1

    @pytest.mark.asyncio
    async def test_arun_stream(self):
        """流式执行应逐步产出事件"""
        mock_llm = MockLLM([_text_response("你好")])
        agent = Agent(llm=mock_llm)

        events = []
        async for event in agent.arun_stream("打个招呼"):
            events.append(event)

        assert len(events) > 0
        types = [e.type for e in events]
        assert EventType.LOOP_START in types
        assert EventType.LOOP_END in types

    def test_run_stream(self):
        """同步流式执行应逐步产出事件"""
        mock_llm = MockLLM([_text_response("你好")])
        agent = Agent(llm=mock_llm)

        events = list(agent.run_stream("打个招呼"))
        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self):
        """多轮对话应传递消息历史"""
        mock_llm1 = MockLLM([_text_response("北京今天晴天")])
        agent1 = Agent(llm=mock_llm1)
        r1 = await agent1.arun("北京天气")

        mock_llm2 = MockLLM([_text_response("上海今天多云")])
        agent2 = Agent(llm=mock_llm2)
        r2 = await agent2.arun("那上海呢？", messages=r1.messages)

        assert r2.content == "上海今天多云"
        # 消息历史应该包含前一轮的内容
        assert len(r2.messages) > len(r1.messages)

    def test_constructor_with_model_params(self):
        """应该支持 model 参数构造"""
        agent = Agent(
            model="deepseek-chat",
            api_key="sk-test",
            base_url="https://api.deepseek.com/v1",
        )
        assert agent is not None

    def test_constructor_with_custom_limits(self):
        """应该支持自定义限制"""
        agent = Agent(
            model="gpt-4o-mini",
            api_key="test",
            limits=LoopLimits(max_steps=5, timeout=60.0),
        )
        assert agent is not None


class TestAgentResult:
    """AgentResult 测试"""

    def test_create_result(self):
        """应该能创建 AgentResult"""
        result = AgentResult(
            content="测试",
            steps=3,
            total_tokens=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            events=[],
            stop_reason="completed",
            messages=[],
        )
        assert result.content == "测试"
        assert result.steps == 3
```

**Step 2: 运行测试验证失败**

```bash
pytest tests/test_agent.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: 实现 agent.py**

`src/pure_agent_loop/agent.py`:
```python
"""Agent 入口

用户使用 pure-agent-loop 的唯一入口点。
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator

from .events import Event, EventType
from .limits import LoopLimits
from .llm.base import BaseLLMClient
from .llm.openai_client import OpenAIClient
from .llm.types import TokenUsage
from .loop import ReactLoop
from .retry import RetryConfig
from .tool import Tool, ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Agent 执行结果

    Attributes:
        content: 最终回答文本
        steps: 总执行步数
        total_tokens: 累计 token 用量
        events: 完整事件历史
        stop_reason: 终止原因 ("completed" | "token_limit" | "error")
        messages: 完整消息历史（可用于续接对话）
    """

    content: str
    steps: int
    total_tokens: TokenUsage
    events: list[Event]
    stop_reason: str
    messages: list[dict[str, Any]]


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
        **llm_kwargs: 透传给 LLM 调用的额外参数
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        llm: BaseLLMClient | None = None,
        tools: list[Tool | dict[str, Any]] | None = None,
        system_prompt: str = "You are a helpful assistant.",
        limits: LoopLimits | None = None,
        retry: RetryConfig | None = None,
        temperature: float = 0.7,
        **llm_kwargs: Any,
    ):
        # 构建 LLM 客户端
        if llm is not None:
            self._llm = llm
        else:
            self._llm = OpenAIClient(
                model=model,
                api_key=api_key,
                base_url=base_url,
            )

        # 注册工具
        self._tool_registry = ToolRegistry()
        if tools:
            self._tool_registry.register_many(tools)

        # 配置
        self._system_prompt = system_prompt
        self._limits = limits or LoopLimits()
        self._retry = retry or RetryConfig()
        self._llm_kwargs: dict[str, Any] = {"temperature": temperature, **llm_kwargs}

    def _create_loop(self) -> ReactLoop:
        """创建循环引擎实例"""
        return ReactLoop(
            llm=self._llm,
            tool_registry=self._tool_registry,
            limits=self._limits,
            retry=self._retry,
            llm_kwargs=self._llm_kwargs,
        )

    async def arun_stream(
        self,
        task: str,
        messages: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[Event]:
        """异步流式执行

        Args:
            task: 任务描述
            messages: 初始消息历史（多轮对话续接）

        Yields:
            Event: 执行过程中的结构化事件
        """
        loop = self._create_loop()
        async for event in loop.run(
            task=task,
            system_prompt=self._system_prompt,
            messages=messages,
        ):
            yield event

    async def arun(
        self,
        task: str,
        messages: list[dict[str, Any]] | None = None,
    ) -> AgentResult:
        """异步执行（阻塞等待最终结果）

        Args:
            task: 任务描述
            messages: 初始消息历史

        Returns:
            AgentResult: 执行结果
        """
        events: list[Event] = []
        async for event in self.arun_stream(task, messages=messages):
            events.append(event)

        return self._build_result(events)

    def run_stream(
        self,
        task: str,
        messages: list[dict[str, Any]] | None = None,
    ) -> Iterator[Event]:
        """同步流式执行

        Args:
            task: 任务描述
            messages: 初始消息历史

        Yields:
            Event: 执行过程中的结构化事件
        """
        loop = asyncio.get_event_loop() if asyncio._get_running_loop() else asyncio.new_event_loop()

        async def _collect():
            events = []
            async for event in self.arun_stream(task, messages=messages):
                events.append(event)
            return events

        try:
            events = loop.run_until_complete(_collect())
        except RuntimeError:
            # 如果已有事件循环在运行，创建新线程
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                new_loop = asyncio.new_event_loop()
                future = pool.submit(new_loop.run_until_complete, _collect())
                events = future.result()
                new_loop.close()

        yield from events

    def run(
        self,
        task: str,
        messages: list[dict[str, Any]] | None = None,
    ) -> AgentResult:
        """同步执行（阻塞等待最终结果）

        Args:
            task: 任务描述
            messages: 初始消息历史

        Returns:
            AgentResult: 执行结果
        """
        events = list(self.run_stream(task, messages=messages))
        return self._build_result(events)

    def _build_result(self, events: list[Event]) -> AgentResult:
        """从事件列表构建 AgentResult"""
        # 查找结束事件
        end_event = next(
            (e for e in events if e.type == EventType.LOOP_END),
            None,
        )

        content = ""
        stop_reason = "unknown"
        messages_history: list[dict[str, Any]] = []
        max_step = 0

        if end_event:
            content = end_event.data.get("content", "")
            stop_reason = end_event.data.get("stop_reason", "unknown")
            messages_history = end_event.data.get("messages", [])
            max_step = end_event.step

        # 累计 token
        total_tokens = TokenUsage.zero()
        for event in events:
            if event.type in (EventType.THOUGHT, EventType.ACTION):
                # token 信息在 loop 内部已通过 checker 累加
                pass

        # 从事件推断总步数
        steps = max_step

        return AgentResult(
            content=content,
            steps=steps,
            total_tokens=total_tokens,
            events=events,
            stop_reason=stop_reason,
            messages=messages_history,
        )
```

**Step 4: 运行测试验证通过**

```bash
pytest tests/test_agent.py -v
```

Expected: 8 passed

**Step 5: Commit**

```bash
git add src/pure_agent_loop/agent.py tests/test_agent.py
git commit -m "feat: 添加 Agent 入口 (Agent, AgentResult)"
```

---

## Task 13: 公开 API 导出 (__init__.py)

**Files:**
- Modify: `src/pure_agent_loop/__init__.py`

**Step 1: 更新 __init__.py**

`src/pure_agent_loop/__init__.py`:
```python
"""pure-agent-loop: 轻量级 ReAct 模式 Agentic Loop 框架"""

__version__ = "0.1.0"

from .agent import Agent, AgentResult
from .tool import tool, Tool, ToolRegistry
from .events import Event, EventType
from .renderer import Renderer
from .limits import LoopLimits
from .retry import RetryConfig
from .llm.base import BaseLLMClient
from .llm.openai_client import OpenAIClient
from .llm.types import LLMResponse, ToolCall, TokenUsage
from .errors import (
    PureAgentLoopError,
    ToolExecutionError,
    LLMError,
    LimitExceededError,
)

__all__ = [
    # 核心入口
    "Agent",
    "AgentResult",
    # 工具
    "tool",
    "Tool",
    "ToolRegistry",
    # 事件
    "Event",
    "EventType",
    # 渲染
    "Renderer",
    # 配置
    "LoopLimits",
    "RetryConfig",
    # LLM
    "BaseLLMClient",
    "OpenAIClient",
    "LLMResponse",
    "ToolCall",
    "TokenUsage",
    # 异常
    "PureAgentLoopError",
    "ToolExecutionError",
    "LLMError",
    "LimitExceededError",
]
```

**Step 2: 验证导入正确**

```bash
python -c "from pure_agent_loop import Agent, tool, Renderer, LoopLimits; print('导入成功')"
```

Expected: `导入成功`

**Step 3: Commit**

```bash
git add src/pure_agent_loop/__init__.py
git commit -m "feat: 完成公开 API 导出"
```

---

## Task 14: 示例代码 (examples/)

**Files:**
- Create: `examples/basic.py`
- Create: `examples/streaming.py`
- Create: `examples/custom_renderer.py`
- Create: `examples/custom_llm.py`

**Step 1: 创建基础示例**

`examples/basic.py`:
```python
"""基础示例: 最简 Agent 使用"""

from pure_agent_loop import Agent, tool


@tool
def search(query: str) -> str:
    """搜索网页内容

    Args:
        query: 搜索关键词
    """
    # 这里替换为实际的搜索实现
    return f"搜索 '{query}' 的结果: Python 是一种通用编程语言..."


@tool
def calculate(expression: str) -> str:
    """计算数学表达式

    Args:
        expression: 数学表达式
    """
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误: {e}"


def main():
    agent = Agent(
        model="deepseek-chat",
        api_key="your-api-key",
        base_url="https://api.deepseek.com/v1",
        tools=[search, calculate],
        system_prompt="你是一个有用的助手，可以搜索信息和计算数学表达式。",
    )

    result = agent.run("Python 语言是什么时候发布的？1991 年到 2026 年一共多少年？")
    print(f"回答: {result.content}")
    print(f"步数: {result.steps}")
    print(f"终止原因: {result.stop_reason}")


if __name__ == "__main__":
    main()
```

**Step 2: 创建流式输出示例**

`examples/streaming.py`:
```python
"""流式输出示例: 实时查看 Agent 执行过程"""

import asyncio
from pure_agent_loop import Agent, tool, Renderer


@tool
def search(query: str) -> str:
    """搜索网页

    Args:
        query: 搜索关键词
    """
    return f"搜索结果: {query} 相关信息..."


async def main():
    agent = Agent(
        model="deepseek-chat",
        api_key="your-api-key",
        base_url="https://api.deepseek.com/v1",
        tools=[search],
    )

    renderer = Renderer()

    # 异步流式执行
    async for event in agent.arun_stream("搜索 Python 最新版本信息"):
        output = renderer.render(event)
        if output:
            print(output)

    print("\n--- 同步方式 ---\n")

    # 同步流式执行
    for event in agent.run_stream("搜索 Python 最新版本信息"):
        print(event.to_dict())


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 3: 创建自定义渲染器示例**

`examples/custom_renderer.py`:
```python
"""自定义渲染器示例: 定制事件展示格式"""

from pure_agent_loop import Agent, tool, Renderer, Event, EventType


@tool
def search(query: str) -> str:
    """搜索网页

    Args:
        query: 搜索关键词
    """
    return f"找到 10 条关于 '{query}' 的结果"


@tool
def save_file(filename: str, content: str) -> str:
    """保存文件

    Args:
        filename: 文件名
        content: 文件内容
    """
    return f"已保存到 {filename}"


# 创建自定义渲染器
renderer = Renderer()


@renderer.on_tool("search")
def render_search(event: Event) -> str:
    """自定义搜索工具的渲染"""
    if event.type == EventType.ACTION:
        return f"🔍 正在搜索: {event.data['args']['query']}"
    return f"📋 搜索完成: {event.data.get('result', '')}"


@renderer.on_tool("save_file")
def render_save(event: Event) -> str:
    """自定义保存文件工具的渲染"""
    if event.type == EventType.ACTION:
        return f"💾 正在保存: {event.data['args']['filename']}"
    return f"✅ 保存完成"


@renderer.on_event(EventType.SOFT_LIMIT)
def render_limit(event: Event) -> str:
    return f"⏰ 提醒: AI 正在调整策略 ({event.data['reason']})"


def main():
    agent = Agent(
        model="deepseek-chat",
        api_key="your-api-key",
        base_url="https://api.deepseek.com/v1",
        tools=[search, save_file],
    )

    for event in agent.run_stream("搜索 Python 教程并保存到文件"):
        output = renderer.render(event)
        if output:
            print(output)


if __name__ == "__main__":
    main()
```

**Step 4: 创建自定义 LLM 客户端示例**

`examples/custom_llm.py`:
```python
"""自定义 LLM 客户端示例: 接入非 OpenAI 兼容的模型"""

from typing import Any
from pure_agent_loop import Agent, BaseLLMClient, LLMResponse, ToolCall, TokenUsage


class MyCustomLLM(BaseLLMClient):
    """自定义 LLM 客户端示例

    演示如何接入任意 LLM API。
    只需继承 BaseLLMClient 并实现 chat 方法。
    """

    def __init__(self, model: str):
        self.model = model

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        # 这里替换为你的 LLM API 调用逻辑
        # 示例: 返回一个简单的响应
        last_message = messages[-1]["content"]

        return LLMResponse(
            content=f"[{self.model}] 收到消息: {last_message}",
            tool_calls=[],
            usage=TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            raw={},
        )


def main():
    # 使用自定义客户端
    custom_llm = MyCustomLLM(model="my-custom-model")
    agent = Agent(llm=custom_llm)

    result = agent.run("你好")
    print(f"回答: {result.content}")


if __name__ == "__main__":
    main()
```

**Step 5: Commit**

```bash
git add examples/
git commit -m "feat: 添加示例代码"
```

---

## Task 15: 运行全量测试并验证

**Step 1: 运行全部测试**

```bash
pytest tests/ -v --tb=short
```

Expected: 所有测试通过

**Step 2: 运行测试覆盖率**

```bash
pytest tests/ --cov=pure_agent_loop --cov-report=term-missing
```

Expected: 覆盖率 > 80%

**Step 3: 验证包可安装**

```bash
pip install -e .
python -c "import pure_agent_loop; print(pure_agent_loop.__version__)"
```

Expected: `0.1.0`

**Step 4: 最终 Commit**

```bash
git add .
git commit -m "chore: 完成 v0.1.0 初始实现"
```

---

## 实现顺序总结

| 顺序 | Task | 模块 | 依赖 |
|------|------|------|------|
| 1 | 项目骨架 | pyproject.toml, 目录 | 无 |
| 2 | errors.py | 自定义异常 | 无 |
| 3 | llm/types.py | 类型定义 | 无 |
| 4 | events.py | 事件系统 | 无 |
| 5 | limits.py | 终止控制 | 无 |
| 6 | retry.py | 重试机制 | 无 |
| 7 | tool.py | 工具系统 | 无 |
| 8 | llm/base.py | LLM 抽象接口 | llm/types |
| 9 | llm/openai_client.py | OpenAI 客户端 | llm/base, llm/types |
| 10 | renderer.py | 渲染器 | events |
| 11 | loop.py | 循环引擎 | 全部基础模块 |
| 12 | agent.py | Agent 入口 | 全部模块 |
| 13 | __init__.py | API 导出 | 全部模块 |
| 14 | examples/ | 示例代码 | 全部模块 |
| 15 | 全量测试 | 验证 | 全部模块 |
