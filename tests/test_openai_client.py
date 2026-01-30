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
