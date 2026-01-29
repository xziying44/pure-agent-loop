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
