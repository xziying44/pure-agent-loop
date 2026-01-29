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
