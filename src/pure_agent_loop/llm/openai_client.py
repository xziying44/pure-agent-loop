"""OpenAI 兼容客户端

基于 openai SDK 实现的 LLM 客户端，兼容所有 OpenAI 兼容 API。
"""

import json
import logging
from typing import Any, Literal

from openai import AsyncOpenAI

from .base import BaseLLMClient
from .types import LLMResponse, ToolCall, TokenUsage

logger = logging.getLogger(__name__)

# 思考深度类型
ThinkingLevel = Literal["off", "low", "medium", "high"]


class OpenAIClient(BaseLLMClient):
    """OpenAI 兼容 LLM 客户端

    支持 OpenAI、Azure OpenAI、DeepSeek、通义千问等所有兼容 API。

    Args:
        model: 模型名称
        api_key: API 密钥（默认读取 OPENAI_API_KEY 环境变量）
        base_url: API 基础地址（用于接入其他兼容服务）
        thinking_level: 思考深度（off/low/medium/high），默认 off
        **kwargs: 传递给 AsyncOpenAI 的额外参数
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        thinking_level: ThinkingLevel = "off",
        **kwargs: Any,
    ):
        self.model = model
        self.thinking_level = thinking_level
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    def _is_openai_reasoning_model(self) -> bool:
        """检测是否为 OpenAI 原生推理模型（o1/o3/o4 系列）"""
        reasoning_prefixes = ("o1", "o3", "o4")
        return any(self.model.startswith(p) for p in reasoning_prefixes)

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

        # 根据 thinking_level 注入思考模式参数
        if self.thinking_level != "off":
            if self._is_openai_reasoning_model():
                # OpenAI o1/o3: 使用 reasoning_effort 参数
                request_kwargs["reasoning_effort"] = self.thinking_level
            else:
                # DeepSeek/智谱等: 使用 extra_body.thinking 参数
                request_kwargs["extra_body"] = {
                    "thinking": {"type": "enabled"}
                }

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

        # 解析 reasoning_content（从 message 属性或 model_extra 中获取）
        reasoning_content = getattr(message, "reasoning_content", None)

        # 解析详细 token 统计
        reasoning_tokens = None
        cached_tokens = None
        usage = response.usage

        if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
            reasoning_tokens = getattr(
                usage.completion_tokens_details, "reasoning_tokens", None
            )
        if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
            cached_tokens = getattr(
                usage.prompt_tokens_details, "cached_tokens", None
            )

        # 构建 TokenUsage
        token_usage = TokenUsage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            reasoning_tokens=reasoning_tokens,
            cached_tokens=cached_tokens,
        )

        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            usage=token_usage,
            raw=response,
            reasoning_content=reasoning_content,
        )
