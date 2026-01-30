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
