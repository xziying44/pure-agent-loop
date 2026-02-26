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
