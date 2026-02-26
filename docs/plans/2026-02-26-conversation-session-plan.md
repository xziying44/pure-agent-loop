# Conversation 多轮会话实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为 pure-agent-loop 添加 `Conversation` 类，支持多轮对话自动续接消息历史。

**Architecture:** `Conversation` 是 `Agent` 的薄包装器，内部维护 `list[dict]` 消息历史。每次 `send` 时将历史传给现有的 `agent.arun_stream(task, messages=...)`，从 LOOP_END 事件提取更新后的消息。Agent/ReactLoop 核心零修改。

**Tech Stack:** Python 3.10+, pytest, pytest-asyncio

---

### Task 1: 创建 Conversation 类 — 测试先行

**Files:**
- Create: `tests/test_conversation.py`
- Create: `src/pure_agent_loop/conversation.py`

**Step 1: 编写 Conversation 的失败测试**

在 `tests/test_conversation.py` 中编写以下测试代码。使用与 `test_agent.py` 相同的 MockLLM 模式：

```python
"""Conversation 多轮会话测试"""

import asyncio
import pytest
from pure_agent_loop.agent import Agent, AgentResult
from pure_agent_loop.conversation import Conversation
from pure_agent_loop.events import Event, EventType
from pure_agent_loop.llm.base import BaseLLMClient
from pure_agent_loop.llm.types import LLMResponse, ToolCall, TokenUsage


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


class TestConversationCreation:
    """Conversation 创建测试"""

    def test_agent_conversation_factory(self):
        """Agent.conversation() 应返回 Conversation 实例"""
        mock_llm = MockLLM([_text_response("你好")])
        agent = Agent(llm=mock_llm)
        conv = agent.conversation()
        assert isinstance(conv, Conversation)

    def test_initial_messages_empty(self):
        """新建 Conversation 的消息历史应为空"""
        mock_llm = MockLLM([_text_response("你好")])
        agent = Agent(llm=mock_llm)
        conv = agent.conversation()
        assert conv.messages == []

    def test_multiple_conversations_independent(self):
        """同一 Agent 创建的多个 Conversation 应互相独立"""
        mock_llm = MockLLM([_text_response("你好")])
        agent = Agent(llm=mock_llm)
        conv1 = agent.conversation()
        conv2 = agent.conversation()
        assert conv1 is not conv2
        assert conv1.messages == []
        assert conv2.messages == []
```

**Step 2: 运行测试验证失败**

运行: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest tests/test_conversation.py -v`
预期: FAIL — `ModuleNotFoundError: No module named 'pure_agent_loop.conversation'`

**Step 3: 创建 Conversation 类的最小实现**

创建 `src/pure_agent_loop/conversation.py`：

```python
"""多轮对话会话

提供 Conversation 类，自动维护消息历史，支持多轮对话续接。
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Iterator, TYPE_CHECKING

from .events import Event, EventType

if TYPE_CHECKING:
    from .agent import Agent, AgentResult


class Conversation:
    """多轮对话会话

    由 Agent.conversation() 创建，自动维护消息历史。
    同一 Agent 可创建多个独立 Conversation。

    使用示例:
        agent = Agent(llm=my_llm)
        conv = agent.conversation()

        # 多轮对话自动续接
        async for event in conv.send_stream("第一个问题"):
            ...
        async for event in conv.send_stream("追问"):
            ...
    """

    def __init__(self, agent: Agent):
        self._agent = agent
        self._messages: list[dict[str, Any]] = []

    @property
    def messages(self) -> list[dict[str, Any]]:
        """当前消息历史（只读副本）"""
        return list(self._messages)

    def reset(self) -> None:
        """清空消息历史，开始新对话"""
        self._messages = []
```

在 `src/pure_agent_loop/agent.py` 的 `Agent` 类中添加工厂方法（在 `_build_result` 方法之前插入）：

```python
    def conversation(self) -> "Conversation":
        """创建一个新的多轮对话会话

        Returns:
            Conversation: 独立的对话会话实例
        """
        from .conversation import Conversation
        return Conversation(self)
```

**Step 4: 运行测试验证通过**

运行: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest tests/test_conversation.py::TestConversationCreation -v`
预期: 3 passed

**Step 5: 提交**

```bash
cd /Users/xziying/project/github/pure-agent-loop
git add src/pure_agent_loop/conversation.py tests/test_conversation.py src/pure_agent_loop/agent.py
git commit -m "feat(conversation): 添加 Conversation 类骨架和 Agent.conversation() 工厂方法"
```

---

### Task 2: 实现异步流式发送 send_stream

**Files:**
- Modify: `tests/test_conversation.py` — 添加 send_stream 测试
- Modify: `src/pure_agent_loop/conversation.py` — 实现 send_stream

**Step 1: 编写 send_stream 的失败测试**

在 `tests/test_conversation.py` 中添加新的测试类：

```python
class TestConversationSendStream:
    """send_stream 异步流式发送测试"""

    @pytest.mark.asyncio
    async def test_send_stream_yields_events(self):
        """send_stream 应产出事件流"""
        mock_llm = MockLLM([_text_response("你好")])
        agent = Agent(llm=mock_llm)
        conv = agent.conversation()

        events = []
        async for event in conv.send_stream("打个招呼"):
            events.append(event)

        types = [e.type for e in events]
        assert EventType.LOOP_START in types
        assert EventType.LOOP_END in types

    @pytest.mark.asyncio
    async def test_send_stream_updates_messages(self):
        """send_stream 完成后应更新内部消息历史"""
        mock_llm = MockLLM([_text_response("你好")])
        agent = Agent(llm=mock_llm)
        conv = agent.conversation()

        assert conv.messages == []
        async for _ in conv.send_stream("打个招呼"):
            pass
        assert len(conv.messages) > 0

    @pytest.mark.asyncio
    async def test_send_stream_multi_turn(self):
        """两轮 send_stream 后，第二轮应包含第一轮的消息历史"""
        # 第一轮和第二轮共享同一个 MockLLM，需要两个响应
        mock_llm = MockLLM([
            _text_response("第一轮回答"),
            _text_response("第二轮回答"),
        ])
        agent = Agent(llm=mock_llm)
        conv = agent.conversation()

        # 第一轮
        async for _ in conv.send_stream("第一个问题"):
            pass
        first_count = len(conv.messages)

        # 第二轮
        async for _ in conv.send_stream("追问"):
            pass
        second_count = len(conv.messages)

        # 第二轮消息应比第一轮多（至少多一个 user + 一个 assistant）
        assert second_count > first_count

    @pytest.mark.asyncio
    async def test_reset_clears_messages(self):
        """reset 后应清空消息历史"""
        mock_llm = MockLLM([
            _text_response("你好"),
            _text_response("重新开始"),
        ])
        agent = Agent(llm=mock_llm)
        conv = agent.conversation()

        async for _ in conv.send_stream("打个招呼"):
            pass
        assert len(conv.messages) > 0

        conv.reset()
        assert conv.messages == []
```

**Step 2: 运行测试验证失败**

运行: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest tests/test_conversation.py::TestConversationSendStream -v`
预期: FAIL — `AttributeError: 'Conversation' object has no attribute 'send_stream'`

**Step 3: 实现 send_stream**

在 `src/pure_agent_loop/conversation.py` 的 `Conversation` 类中添加：

```python
    async def send_stream(self, task: str) -> AsyncIterator[Event]:
        """发送消息并流式返回事件

        自动将内部维护的消息历史传递给 Agent，
        并在 LOOP_END 事件中捕获更新后的消息历史。

        Args:
            task: 任务描述

        Yields:
            Event: 执行过程中的结构化事件
        """
        # 首次调用时 messages 为空 → 传 None（等同于新对话）
        # 后续调用时 messages 非空 → 传入历史实现续接
        msgs = self._messages if self._messages else None

        async for event in self._agent.arun_stream(task, messages=msgs):
            # 捕获 LOOP_END 事件，更新内部消息历史
            if event.type == EventType.LOOP_END:
                self._messages = event.data.get("messages", [])
            yield event
```

**Step 4: 运行测试验证通过**

运行: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest tests/test_conversation.py::TestConversationSendStream -v`
预期: 4 passed

**Step 5: 提交**

```bash
cd /Users/xziying/project/github/pure-agent-loop
git add tests/test_conversation.py src/pure_agent_loop/conversation.py
git commit -m "feat(conversation): 实现 send_stream 异步流式多轮对话"
```

---

### Task 3: 实现异步阻塞发送 send

**Files:**
- Modify: `tests/test_conversation.py` — 添加 send 测试
- Modify: `src/pure_agent_loop/conversation.py` — 实现 send

**Step 1: 编写 send 的失败测试**

在 `tests/test_conversation.py` 中添加：

```python
class TestConversationSend:
    """send 异步阻塞发送测试"""

    @pytest.mark.asyncio
    async def test_send_returns_agent_result(self):
        """send 应返回 AgentResult"""
        mock_llm = MockLLM([_text_response("你好")])
        agent = Agent(llm=mock_llm)
        conv = agent.conversation()

        result = await conv.send("打个招呼")
        assert isinstance(result, AgentResult)
        assert result.content == "你好"
        assert result.stop_reason == "completed"

    @pytest.mark.asyncio
    async def test_send_updates_messages(self):
        """send 完成后应更新内部消息历史"""
        mock_llm = MockLLM([_text_response("你好")])
        agent = Agent(llm=mock_llm)
        conv = agent.conversation()

        await conv.send("打个招呼")
        assert len(conv.messages) > 0

    @pytest.mark.asyncio
    async def test_send_multi_turn(self):
        """两轮 send 应实现多轮对话续接"""
        mock_llm = MockLLM([
            _text_response("第一轮"),
            _text_response("第二轮"),
        ])
        agent = Agent(llm=mock_llm)
        conv = agent.conversation()

        r1 = await conv.send("问题1")
        r2 = await conv.send("追问")

        assert r1.content == "第一轮"
        assert r2.content == "第二轮"
        assert len(r2.messages) > len(r1.messages)
```

**Step 2: 运行测试验证失败**

运行: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest tests/test_conversation.py::TestConversationSend -v`
预期: FAIL — `AttributeError: 'Conversation' object has no attribute 'send'`

**Step 3: 实现 send**

在 `src/pure_agent_loop/conversation.py` 的 `Conversation` 类中添加：

```python
    async def send(self, task: str) -> AgentResult:
        """发送消息并返回完整结果

        Args:
            task: 任务描述

        Returns:
            AgentResult: 执行结果
        """
        events: list[Event] = []
        async for event in self.send_stream(task):
            events.append(event)
        return self._agent._build_result(events)
```

**Step 4: 运行测试验证通过**

运行: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest tests/test_conversation.py::TestConversationSend -v`
预期: 3 passed

**Step 5: 提交**

```bash
cd /Users/xziying/project/github/pure-agent-loop
git add tests/test_conversation.py src/pure_agent_loop/conversation.py
git commit -m "feat(conversation): 实现 send 异步阻塞多轮对话"
```

---

### Task 4: 实现同步方法 send_sync / send_stream_sync

**Files:**
- Modify: `tests/test_conversation.py` — 添加同步测试
- Modify: `src/pure_agent_loop/conversation.py` — 实现同步方法

**Step 1: 编写同步方法的失败测试**

在 `tests/test_conversation.py` 中添加：

```python
class TestConversationSync:
    """同步方法测试"""

    def test_send_sync_returns_result(self):
        """send_sync 应返回 AgentResult"""
        mock_llm = MockLLM([_text_response("你好")])
        agent = Agent(llm=mock_llm)
        conv = agent.conversation()

        result = conv.send_sync("打个招呼")
        assert isinstance(result, AgentResult)
        assert result.content == "你好"

    def test_send_stream_sync_yields_events(self):
        """send_stream_sync 应产出事件"""
        mock_llm = MockLLM([_text_response("你好")])
        agent = Agent(llm=mock_llm)
        conv = agent.conversation()

        events = list(conv.send_stream_sync("打个招呼"))
        types = [e.type for e in events]
        assert EventType.LOOP_START in types
        assert EventType.LOOP_END in types

    def test_send_sync_multi_turn(self):
        """同步多轮对话应正常续接"""
        mock_llm = MockLLM([
            _text_response("第一轮"),
            _text_response("第二轮"),
        ])
        agent = Agent(llm=mock_llm)
        conv = agent.conversation()

        r1 = conv.send_sync("问题1")
        r2 = conv.send_sync("追问")

        assert r1.content == "第一轮"
        assert r2.content == "第二轮"
        assert len(r2.messages) > len(r1.messages)
```

**Step 2: 运行测试验证失败**

运行: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest tests/test_conversation.py::TestConversationSync -v`
预期: FAIL

**Step 3: 实现同步方法**

在 `src/pure_agent_loop/conversation.py` 的 `Conversation` 类中添加。参考 `agent.py` 中 `run_stream` 的同步实现模式：

```python
    def send_stream_sync(self, task: str) -> Iterator[Event]:
        """同步流式发送

        Args:
            task: 任务描述

        Yields:
            Event: 执行过程中的结构化事件
        """
        try:
            asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                new_loop = asyncio.new_event_loop()

                async def _collect():
                    events = []
                    async for event in self.send_stream(task):
                        events.append(event)
                    return events

                future = pool.submit(new_loop.run_until_complete, _collect())
                events = future.result()
                new_loop.close()
        except RuntimeError:
            async def _collect():
                events = []
                async for event in self.send_stream(task):
                    events.append(event)
                return events

            events = asyncio.run(_collect())

        yield from events

    def send_sync(self, task: str) -> AgentResult:
        """同步发送

        Args:
            task: 任务描述

        Returns:
            AgentResult: 执行结果
        """
        events = list(self.send_stream_sync(task))
        return self._agent._build_result(events)
```

**Step 4: 运行测试验证通过**

运行: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest tests/test_conversation.py::TestConversationSync -v`
预期: 3 passed

**Step 5: 提交**

```bash
cd /Users/xziying/project/github/pure-agent-loop
git add tests/test_conversation.py src/pure_agent_loop/conversation.py
git commit -m "feat(conversation): 实现同步 send_sync/send_stream_sync 方法"
```

---

### Task 5: 导出 Conversation 并更新 __init__.py

**Files:**
- Modify: `src/pure_agent_loop/__init__.py` — 添加导出

**Step 1: 编写导入测试**

在 `tests/test_conversation.py` 中添加：

```python
class TestConversationImport:
    """导入路径测试"""

    def test_import_from_package(self):
        """应能从包顶层导入 Conversation"""
        from pure_agent_loop import Conversation
        assert Conversation is not None
```

**Step 2: 运行测试验证失败**

运行: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest tests/test_conversation.py::TestConversationImport -v`
预期: FAIL — `ImportError: cannot import name 'Conversation'`

**Step 3: 更新 __init__.py**

在 `src/pure_agent_loop/__init__.py` 中添加：

1. 在导入区域（`from .agent import ...` 附近）添加：
```python
from .conversation import Conversation
```

2. 在 `__all__` 列表的 `# 核心入口` 区域添加：
```python
    "Conversation",
```

**Step 4: 运行测试验证通过**

运行: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest tests/test_conversation.py::TestConversationImport -v`
预期: PASS

**Step 5: 运行全量测试确保无回归**

运行: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest -v`
预期: 全部 PASS

**Step 6: 提交**

```bash
cd /Users/xziying/project/github/pure-agent-loop
git add src/pure_agent_loop/__init__.py tests/test_conversation.py
git commit -m "feat(conversation): 导出 Conversation 到包顶层"
```

---

### Task 6: 添加使用示例

**Files:**
- Create: `examples/conversation.py`

**Step 1: 创建示例文件**

```python
"""多轮对话示例: 使用 Conversation 实现会话续接

使用前请先安装依赖并配置环境变量:
    pip install python-dotenv
    cp .env.example .env
    # 编辑 .env 填入实际的 API 密钥
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from pure_agent_loop import Agent, Renderer

# 加载 examples/.env 配置
load_dotenv(Path(__file__).parent / ".env")


async def main():
    agent = Agent(
        name="对话助手",
        model=os.getenv("MODEL", "deepseek-chat"),
        api_key=os.environ["API_KEY"],
        base_url=os.getenv("BASE_URL", "https://api.deepseek.com/v1"),
        system_prompt="你是一个友好的对话助手。",
    )
    renderer = Renderer()

    # 创建会话 — 多轮对话自动续接
    conv = agent.conversation()

    print("=" * 50)
    print("第一轮对话")
    print("=" * 50)

    async for event in conv.send_stream("Python 是什么语言？"):
        output = renderer.render(event)
        if output:
            print(output)

    print(f"\n当前消息历史长度: {len(conv.messages)}")

    print("\n" + "=" * 50)
    print("第二轮对话（自动续接上下文）")
    print("=" * 50)

    async for event in conv.send_stream("它有哪些主要的应用领域？"):
        output = renderer.render(event)
        if output:
            print(output)

    print(f"\n当前消息历史长度: {len(conv.messages)}")

    # 创建全新对话（不带历史）
    print("\n" + "=" * 50)
    print("全新对话（独立 Conversation）")
    print("=" * 50)

    conv2 = agent.conversation()
    async for event in conv2.send_stream("1 + 1 等于多少？"):
        output = renderer.render(event)
        if output:
            print(output)


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 2: 提交**

```bash
cd /Users/xziying/project/github/pure-agent-loop
git add examples/conversation.py
git commit -m "docs: 添加多轮对话 Conversation 使用示例"
```

---

### Task 7: 最终验证

**Step 1: 运行全量测试**

运行: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest -v`
预期: 全部 PASS，无回归

**Step 2: 运行带覆盖率测试**

运行: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest --cov=pure_agent_loop --cov-report=term-missing tests/test_conversation.py`
预期: `conversation.py` 覆盖率 >= 80%
