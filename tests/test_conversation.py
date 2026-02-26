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
