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
