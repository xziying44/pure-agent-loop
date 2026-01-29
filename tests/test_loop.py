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
